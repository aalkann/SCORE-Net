# Legacy code

Kept for provenance only. Nothing here is imported by `utils/`, `scripts/` or the
notebooks, and nothing here produces a table or figure in the manuscript.

## `plot_utils_legacy.py`

The original `utils/plot_utils.py` (1,443 lines). Six of its seven plotting
helpers were written for exploratory analysis and are not referenced anywhere in
the repository:

| Function | Status |
|---|---|
| `plot_exact_feature_impact` | unused |
| `plot_simulated_feature_impact` | unused |
| `plot_global_feature_importance` | unused |
| `plot_feature_rule_cooccurrence_dashboard` | unused |
| `plot_rule_complexity_and_impact_dashboard` | unused |
| `plot_scorecard_quantization_dashboard` | unused |
| `plot_exact_actionable_recourse` | **live** — moved to `utils/plot_utils.py` |

The live function was moved to `utils/plot_utils.py` (307 lines) so the public
API reflects what the paper actually uses. If you need one of the retired
helpers, copy it back rather than importing from this folder.
