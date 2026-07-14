# Notebooks

EDA and final evaluation figures only. Keep training / CV loops in `src/` and `scripts/` so notebook cell order cannot introduce look-ahead leakage.

| File | Purpose |
|------|---------|
| `01_EDA_stylised_facts.ipynb` | Stylised facts, macro/cross-asset overview, regime overlays |
| `02_Model_Results_and_Evaluation.ipynb` | Ablation, thesis vs control, bootstrap/DSR, costs, holdout, conclusions |

Regenerate figures by running notebooks after `reports/` artefacts exist (see root `README.md`).
