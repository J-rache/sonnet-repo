param(
    [string]$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path,
    [string]$HostName = '127.0.0.1',
    [int]$Port = 8000
)

$python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    $python = 'python'
}

& $python -m daemon.supervisor --repo-root $RepoRoot --host $HostName --port $Port
