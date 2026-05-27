# sync_to_website.ps1
# Copies generated chart HTML files from docs/ to the website directory.
# Run after GitHub Actions pulls fresh charts, or after running scripts locally.

$src  = "$PSScriptRoot\..\docs"
$dest = "C:\Users\cwhit\Documents\Alphaline Research\Website\New"

$charts = @(
    "eth_stable_model_compact.html",
    "eth_combined_tvl.html",
    "eth_stacked_tvl.html",
    "eth_proxy_signal.html",
    "eth_tvl_momentum.html",
    "eth_combined_components_tvl.html",
    "eth_model_zscore.html",
    "eth_stacked_tvl_mcap.html",
    "btc_rv7_rv30_proxy_signal.html",
    "mstr_pnav_heatmap.html",
    "btc_rsi_weekly.html",
    "eth_rsi_weekly.html",
    "btc_rsi_annual.html",
    "eth_rsi_annual.html",
    "btc_cost_momentum.html"
)

$copied = 0
foreach ($file in $charts) {
    $srcFile  = Join-Path $src  $file
    $destFile = Join-Path $dest $file
    if (Test-Path $srcFile) {
        Copy-Item -Path $srcFile -Destination $destFile -Force
        Write-Host "  copied $file"
        $copied++
    } else {
        Write-Warning "  missing: $srcFile"
    }
}

Write-Host ""
Write-Host "$copied / $($charts.Count) files synced to website."
