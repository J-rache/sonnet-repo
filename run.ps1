param(
    [string]$InstallRoot = (Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'PNP')
)

$ErrorActionPreference = 'Stop'
$launcher = Join-Path $InstallRoot 'PNP-Start.ps1'
if (-not (Test-Path -LiteralPath $launcher)) {
    throw "PNP is not installed at $InstallRoot. Run .\install.ps1 first."
}

& $launcher
exit $LASTEXITCODE
