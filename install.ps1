param(
    [string]$InstallRoot = (Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'PNP'),
    [ValidateSet('mock', 'ollama', 'openai_compatible', 'anthropic')]
    [string]$Provider = 'mock',
    [string]$ModelId = 'local-model',
    [string]$LocalToken = '',
    [switch]$CreateDesktopShortcuts,
    [switch]$NoDesktopShortcuts,
    [switch]$SkipDependencyInstall,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$installer = Join-Path $PSScriptRoot 'scripts\install_user.ps1'
& $installer @PSBoundParameters
exit $LASTEXITCODE
