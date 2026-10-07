param(
    [string]$ConfigPath = ""
)

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $ConfigPath) {
    $ConfigPath = Join-Path $RepoRoot "automation\config\local.toml"
}

& python (Join-Path $RepoRoot "automation\local_watcher.py") --config $ConfigPath
exit $LASTEXITCODE
