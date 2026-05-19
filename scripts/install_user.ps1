[CmdletBinding()]
param(
    [string]$InstallRoot = (Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'PNP'),
    [ValidateSet('mock', 'ollama', 'openai_compatible', 'anthropic')]
    [string]$Provider = 'mock',
    [string]$ModelId = 'local-model',
    [string]$LocalToken = '',
    [switch]$CreateDesktopShortcuts,
    [switch]$NoDesktopShortcuts,
    [switch]$SkipDependencyInstall,
    [switch]$InstallTrainingDeps,
    [string]$SyncModelBase = '',
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Resolve-FullPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path (Get-Location).Path $Path))
}

function Quote-Yaml {
    param([AllowEmptyString()][string]$Value)
    return "'" + ($Value -replace "'", "''") + "'"
}

function Quote-PS {
    param([AllowEmptyString()][string]$Value)
    return "'" + ($Value -replace "'", "''") + "'"
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$ArgumentList = @()
    )
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $FilePath $($ArgumentList -join ' ')"
    }
}

function New-DesktopShortcut {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$ScriptPath
    )
    $desktop = [Environment]::GetFolderPath('DesktopDirectory')
    if (-not $desktop) {
        return $null
    }
    $shortcutPath = Join-Path $desktop "$Name.lnk"
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
    $shortcut.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$ScriptPath`""
    $shortcut.WorkingDirectory = (Split-Path -Parent $ScriptPath)
    $shortcut.IconLocation = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe,0"
    $shortcut.Save()
    return $shortcutPath
}

function Copy-AppSource {
    param(
        [Parameter(Mandatory = $true)][string]$SourceRoot,
        [Parameter(Mandatory = $true)][string]$AppRoot
    )

    if ((Test-Path -LiteralPath $AppRoot) -and $Force) {
        Remove-Item -LiteralPath $AppRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $AppRoot | Out-Null

    $tracked = @()
    Push-Location $SourceRoot
    try {
        $tracked = & git ls-files --cached --others --exclude-standard
        if ($LASTEXITCODE -ne 0 -or -not $tracked) {
            $tracked = @()
        }
    }
    catch {
        $tracked = @()
    }
    finally {
        Pop-Location
    }

    if ($tracked.Count -eq 0) {
        $excluded = @('.git', '.venv', 'venv', 'ENV', 'data', '.smoke', '.install-smoke', '.pytest_cache', '__pycache__', 'dist', 'build')
        $files = Get-ChildItem -LiteralPath $SourceRoot -Recurse -File | Where-Object {
            $relative = [System.IO.Path]::GetRelativePath($SourceRoot, $_.FullName)
            $parts = $relative -split '[\\/]'
            -not ($parts | Where-Object { $excluded -contains $_ })
        }
        foreach ($file in $files) {
            $relative = [System.IO.Path]::GetRelativePath($SourceRoot, $file.FullName)
            $destination = Join-Path $AppRoot $relative
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
            Copy-Item -LiteralPath $file.FullName -Destination $destination -Force
        }
        return
    }

    foreach ($relative in $tracked) {
        if ($relative -match '^(data|\.venv|venv|ENV|\.smoke|\.install-smoke|__pycache__|\.pytest_cache|dist|build)/') {
            continue
        }
        $source = Join-Path $SourceRoot $relative
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
            continue
        }
        $destination = Join-Path $AppRoot $relative
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
        Copy-Item -LiteralPath $source -Destination $destination -Force
    }
}

$SourceRoot = Resolve-FullPath (Join-Path $PSScriptRoot '..')
$InstallRoot = Resolve-FullPath $InstallRoot
$AppRoot = Join-Path $InstallRoot 'app'
$DataRoot = Join-Path $InstallRoot 'data'
$ConfigRoot = Join-Path $InstallRoot 'config'
$LogRoot = Join-Path $InstallRoot 'logs'
$VenvRoot = Join-Path $InstallRoot '.venv'
$InstalledConfig = Join-Path $ConfigRoot 'installed.yaml'
$VenvPython = Join-Path $VenvRoot 'Scripts\python.exe'

if (-not $LocalToken) {
    $LocalToken = [guid]::NewGuid().ToString('N')
}
$SyncBackend = if ($InstallTrainingDeps) { 'peft_lora' } else { 'context_pack' }

New-Item -ItemType Directory -Force -Path $InstallRoot, $DataRoot, $ConfigRoot, $LogRoot | Out-Null
Copy-AppSource -SourceRoot $SourceRoot -AppRoot $AppRoot

$configText = @"
base_model: $(Quote-Yaml $ModelId)
model_id: $(Quote-Yaml $ModelId)
inference_provider: $(Quote-Yaml $Provider)

providers:
  anthropic:
    api_key_env: 'ANTHROPIC_API_KEY'
  openai_compatible:
    api_base: 'http://127.0.0.1:8001/v1'
    api_key_env: 'OPENAI_API_KEY'
  ollama:
    api_base: 'http://127.0.0.1:11434'

heartbeat_interval_ms: 100
consolidation_idle_threshold_seconds: 30

data_dir: $(Quote-Yaml $DataRoot)
working_memory_capacity: 8192
embedding_dimensions: 128
episodic_db_path: $(Quote-Yaml (Join-Path $DataRoot 'episodic.db'))
semantic_db_path: $(Quote-Yaml (Join-Path $DataRoot 'semantic.db'))
embed_path: $(Quote-Yaml (Join-Path $DataRoot 'embeddings.pkl'))
adapter_path: $(Quote-Yaml (Join-Path $DataRoot 'adapter'))
core_state_path: $(Quote-Yaml (Join-Path $DataRoot 'core_state.json'))
journal_path: $(Quote-Yaml (Join-Path $DataRoot 'events.jsonl'))
project_data_dir: $(Quote-Yaml (Join-Path $DataRoot 'projects'))
project_archive_dir: $(Quote-Yaml (Join-Path $DataRoot 'project_archives'))

consolidation_batch_size: 20
episodic_decay_rate_per_hour: 0.02
episodic_consolidation_threshold: 0.15
consolidation_model_id: $(Quote-Yaml $ModelId)
consolidation_max_tokens: 1000

adapter_checkpoint_interval: 100
adapter_max_deltas_in_memory: 1000
adapter_rank: 8
adapter_alpha: 8.0
adapter_learning_rate: 0.05
adapter_training_epochs: 80
adapter_auto_train: true
adapter_train_min_deltas: 1
adapter_train_interval: 1
adapter_seed: 'pnp'
sync_model_enabled: true
sync_model_adapter_backend: $(Quote-Yaml $SyncBackend)
sync_model_base_model: $(Quote-Yaml $SyncModelBase)
peft_training_epochs: 1
peft_batch_size: 1
peft_max_length: 512
peft_learning_rate: 0.0002

api_host: '127.0.0.1'
api_port: 8000
api_log_level: 'info'
local_api_token_env: 'PNP_LOCAL_TOKEN'
local_api_token: $(Quote-Yaml $LocalToken)
"@
$configText | Set-Content -LiteralPath $InstalledConfig -Encoding UTF8

if (-not $SkipDependencyInstall) {
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        $pythonCandidates = @('py', 'python')
        $created = $false
        foreach ($candidate in $pythonCandidates) {
            try {
                if ($candidate -eq 'py') {
                    & py -3.12 -m venv $VenvRoot
                }
                else {
                    & python -m venv $VenvRoot
                }
                if ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $VenvPython)) {
                    $created = $true
                    break
                }
            }
            catch {
                $created = $false
            }
        }
        if (-not $created) {
            throw "Could not create venv. Install Python 3.12 or ensure python is on PATH."
        }
    }

    Invoke-Checked -FilePath $VenvPython -ArgumentList @('-m', 'pip', 'install', '--upgrade', 'pip')
    Invoke-Checked -FilePath $VenvPython -ArgumentList @('-m', 'pip', 'install', '-r', (Join-Path $AppRoot 'requirements.txt'))
    Invoke-Checked -FilePath $VenvPython -ArgumentList @('-m', 'pip', 'install', '-e', $AppRoot)
    if ($InstallTrainingDeps) {
        Invoke-Checked -FilePath $VenvPython -ArgumentList @('-m', 'pip', 'install', '-e', "$AppRoot[train]")
    }
}

$quotedConfig = Quote-PS $InstalledConfig
$quotedToken = Quote-PS $LocalToken
$quotedProvider = Quote-PS $Provider
$quotedModel = Quote-PS $ModelId
$quotedAppRoot = Quote-PS $AppRoot
$quotedPython = Quote-PS $VenvPython

$startLauncher = Join-Path $InstallRoot 'PNP-Start.ps1'
@"
`$ErrorActionPreference = 'Stop'
`$env:PNP_CONFIG_PATH = $quotedConfig
`$env:PNP_LOCAL_TOKEN = $quotedToken
if (-not `$env:PNP_INFERENCE_PROVIDER) { `$env:PNP_INFERENCE_PROVIDER = $quotedProvider }
if (-not `$env:PNP_MODEL_ID) { `$env:PNP_MODEL_ID = $quotedModel }
Set-Location $quotedAppRoot
& $quotedPython 'main.py'
exit `$LASTEXITCODE
"@ | Set-Content -LiteralPath $startLauncher -Encoding UTF8

$supervisorLauncher = Join-Path $InstallRoot 'PNP-Supervisor.ps1'
@"
`$ErrorActionPreference = 'Stop'
`$env:PNP_CONFIG_PATH = $quotedConfig
`$env:PNP_LOCAL_TOKEN = $quotedToken
if (-not `$env:PNP_INFERENCE_PROVIDER) { `$env:PNP_INFERENCE_PROVIDER = $quotedProvider }
if (-not `$env:PNP_MODEL_ID) { `$env:PNP_MODEL_ID = $quotedModel }
Set-Location $quotedAppRoot
& $quotedPython -m daemon.supervisor --repo-root $quotedAppRoot --host '127.0.0.1' --port 8000
exit `$LASTEXITCODE
"@ | Set-Content -LiteralPath $supervisorLauncher -Encoding UTF8

$smokeLauncher = Join-Path $InstallRoot 'PNP-Smoke.ps1'
@"
`$ErrorActionPreference = 'Stop'
`$env:PNP_CONFIG_PATH = $quotedConfig
`$env:PNP_LOCAL_TOKEN = $quotedToken
Set-Location $quotedAppRoot
& $quotedPython -m compileall -q .
if (`$LASTEXITCODE -ne 0) { exit `$LASTEXITCODE }
& $quotedPython 'scripts\smoke_api.py'
if (`$LASTEXITCODE -ne 0) { exit `$LASTEXITCODE }
& $quotedPython 'scripts\smoke_supervisor.py'
exit `$LASTEXITCODE
"@ | Set-Content -LiteralPath $smokeLauncher -Encoding UTF8

$statusLauncher = Join-Path $InstallRoot 'PNP-Status.ps1'
@"
`$ErrorActionPreference = 'Stop'
`$status = [ordered]@{
  install_root = $(Quote-PS $InstallRoot)
  app_root = $(Quote-PS $AppRoot)
  config = $(Quote-PS $InstalledConfig)
  docs_url = 'http://127.0.0.1:8000/docs'
}
try {
  `$status.health = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/' -TimeoutSec 3
}
catch {
  `$status.health = 'offline'
}
`$status | ConvertTo-Json -Depth 8
"@ | Set-Content -LiteralPath $statusLauncher -Encoding UTF8

$docsLauncher = Join-Path $InstallRoot 'PNP-Docs.ps1'
@"
Start-Process 'http://127.0.0.1:8000/docs'
"@ | Set-Content -LiteralPath $docsLauncher -Encoding UTF8

$sourceCommit = ''
Push-Location $SourceRoot
try {
    $sourceCommit = (& git rev-parse HEAD 2>$null)
    if ($LASTEXITCODE -ne 0) {
        $sourceCommit = ''
    }
}
catch {
    $sourceCommit = ''
}
finally {
    Pop-Location
}

$shortcuts = @()
if (-not $NoDesktopShortcuts -or $CreateDesktopShortcuts) {
    $shortcuts += New-DesktopShortcut -Name 'PNP Start' -ScriptPath $startLauncher
    $shortcuts += New-DesktopShortcut -Name 'PNP Supervisor' -ScriptPath $supervisorLauncher
    $shortcuts += New-DesktopShortcut -Name 'PNP Status' -ScriptPath $statusLauncher
    $shortcuts += New-DesktopShortcut -Name 'PNP Docs' -ScriptPath $docsLauncher
}

$manifest = [ordered]@{
    app = 'PNP'
    installed_at = (Get-Date).ToString('o')
    install_root = $InstallRoot
    app_root = $AppRoot
    data_root = $DataRoot
    config_path = $InstalledConfig
    venv_python = $VenvPython
    provider = $Provider
    model_id = $ModelId
    training_dependencies = [bool]$InstallTrainingDeps
    sync_model_base = $SyncModelBase
    source_root = $SourceRoot
    source_commit = $sourceCommit
    launchers = [ordered]@{
        start = $startLauncher
        supervisor = $supervisorLauncher
        smoke = $smokeLauncher
        status = $statusLauncher
        docs = $docsLauncher
    }
    desktop_shortcuts = @($shortcuts | Where-Object { $_ })
}
$manifestPath = Join-Path $InstallRoot 'install_manifest.json'
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

[ordered]@{
    ok = $true
    install_root = $InstallRoot
    manifest = $manifestPath
    start = $startLauncher
    supervisor = $supervisorLauncher
    smoke = $smokeLauncher
    status = $statusLauncher
    config = $InstalledConfig
    token_source = 'config/local_api_token or PNP_LOCAL_TOKEN'
    dependency_install = -not $SkipDependencyInstall
    desktop_shortcuts = @($shortcuts | Where-Object { $_ })
} | ConvertTo-Json -Depth 8
