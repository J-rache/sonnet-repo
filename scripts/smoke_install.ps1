param(
    [string]$InstallRoot = (Join-Path (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path '.install-smoke\PNP'),
    [switch]$Keep
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$installRootFull = [System.IO.Path]::GetFullPath($InstallRoot)
$expectedParent = [System.IO.Path]::GetFullPath((Join-Path $repoRoot '.install-smoke'))

if (-not $installRootFull.StartsWith($expectedParent, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to remove or overwrite install root outside .install-smoke: $installRootFull"
}

if (Test-Path -LiteralPath $installRootFull) {
    Remove-Item -LiteralPath $installRootFull -Recurse -Force
}

$installer = Join-Path $repoRoot 'install.ps1'
& $installer `
    -InstallRoot $installRootFull `
    -Provider mock `
    -ModelId mock-model `
    -LocalToken smoke-install-token `
    -NoDesktopShortcuts `
    -Force

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$smoke = Join-Path $installRootFull 'PNP-Smoke.ps1'
& $smoke
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$manifestPath = Join-Path $installRootFull 'install_manifest.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json

$result = [ordered]@{
    ok = $true
    install_root = $installRootFull
    manifest = $manifestPath
    app_root = $manifest.app_root
    venv_python = $manifest.venv_python
    smoke_launcher = $manifest.launchers.smoke
}
$result | ConvertTo-Json -Depth 6

if (-not $Keep) {
    Remove-Item -LiteralPath $installRootFull -Recurse -Force
}
