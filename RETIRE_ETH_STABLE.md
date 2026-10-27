# Retiring the ETH stablecoin-only model (eth_stable_model_compact)

The website no longer shows this chart (done on the website side). Old links to `charts.html#eth-security` now land on ETH Model Price. Charts-repo checklist:

1. **Workflow — do both in the same commit, or the daily run fails for every chart:**
   - Remove the step `Run ETH Stable Model Compact chart`.
   - Remove `docs/eth_stable_model_compact.html` from the `git add` list. (`git add` on a missing path errors and stops the commit step.)
2. **Keep the public URL working.** Older posts link to `https://alphalineresearch.github.io/alphaline-charts/eth_stable_model_compact.html`. Replace that file with a small redirect page (meta refresh + link) to `https://alphalineresearch.com/charts#eth-combined`, rather than deleting it. Commit it once; it no longer needs regenerating.
3. **Manifest:** exclude `eth_stable_model_compact.html` in `generate_manifest.py` so it doesn't appear in the chart list on the GitHub Pages index.
4. **Thumbnail:** delete `docs/thumbs/eth_stable_model_compact.webp`.
5. **Script:** delete or move `scripts/generate_eth_stable_model_compact.py` to an `archive/` folder.
6. Nothing in `generate_signals.py` uses this chart — no change needed there. Run `generate_manifest.py` after and confirm 6 `signals.items` / 6 `signals.thumbs`.
