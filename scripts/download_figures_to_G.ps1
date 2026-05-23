# Download superconductor_seed0 figures from server ta2 to local G:\
# Run in Windows PowerShell (outside Cursor remote session):
#   powershell -ExecutionPolicy Bypass -File .\download_figures_to_G.ps1
#
# Edit $RemoteHost if you connect via jump host / another IP or alias.

param(
    [string]$RemoteHost = "xk@ta2",
    [string]$LocalDir = "G:\superconductor_seed0",
    [switch]$UseArchive
)

$RemoteFiguresDir = "/data/xk/zyh_dfgo/DUO/results/figures/superconductor_seed0"
$RemoteArchive = "/data/xk/zyh_dfgo/DUO/results/figures/superconductor_seed0.tar.gz"

New-Item -ItemType Directory -Force -Path $LocalDir | Out-Null

if ($UseArchive) {
    Write-Host "[scp] archive -> $LocalDir\superconductor_seed0.tar.gz"
    scp "${RemoteHost}:${RemoteArchive}" "$LocalDir\superconductor_seed0.tar.gz"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "[done] extract with: tar -xzf $LocalDir\superconductor_seed0.tar.gz -C $LocalDir"
} else {
    Write-Host "[scp] folder -> $LocalDir"
    scp -r "${RemoteHost}:${RemoteFiguresDir}" $LocalDir
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "[done] figures under $LocalDir\superconductor_seed0"
}
