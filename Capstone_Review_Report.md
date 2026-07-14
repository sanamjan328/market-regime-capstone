# Capstone Review Report  
## Predict Market Regimes before Price Prediction

| | |
|---|---|
| **Author** | Sanam Jan |
| **Project** | Market Regime Capstone |
| **Universe** | SPY (S&P 500 ETF), daily bars, 2000–2024 |
| **Report date** | July 2026 |
| **Primary sources** | `notebooks/02_Model_Results_and_Evaluation.ipynb`, `reports/*.csv`, `reports/*.json` |

---

## 1. Executive Summary

This project tests whether **regime conditioning via Feature-wise Linear Modulation (FiLM)** improves PatchTST return/volatility forecasting relative to an identical **regime-agnostic control**. Regimes are estimated with a **causal 3-state HMM**; forecasts drive a **Kelly-lite**, cost-aware strategy evaluated with purged walk-forward CV and a locked final holdout (2022–2024).

**Thesis verdict: not supported.** On the official ablation (OOS 2011–2021) and the final holdout, PatchTST + FiLM underperforms PatchTST without regimes. Buy & Hold remains the strongest simple benchmark on OOS Sharpe; on holdout, the no-regime PatchTST is best among tested models. This is a scientifically useful **negative result**: methodology and evaluation design are stronger than the thesis claim on this dataset.

---

## 2. Project Scope and Architecture

### 2.1 Research question

> Does soft, posterior-driven FiLM conditioning of PatchTST improve out-of-sample risk-adjusted performance versus the same Transformer without regime inputs?

### 2.2 Pipeline (end-to-end)

1. **Data** — SPY daily panel (plus macro / cross-asset features per configuration)  
2. **Features** — engineered returns, volatility, and related predictors; fold-wise scaling  
3. **Regimes** — 3-state HMM on (return, realized vol) with **filtered** (causal) posteriors only  
4. **Forecasting** — PatchTST variants (± FiLM / hard switch), DLinear, classical baselines  
5. **Strategy** — Kelly-lite sizing (vol target ≈ 10%, leverage cap 2×), next-open execution, commissions  
6. **Evaluation** — purged expanding CV + embargo; bootstrap CIs; Deflated Sharpe (DSR); cost sensitivity; holdout once  

### 2.3 Correctness practices (strength of the project)

- Filtered HMM posteriors (no smoothed look-ahead)  
- Refit of regimes/scalers on **train only** per fold  
- Purged walk-forward CV with embargo; **2022–2024 holdout touched once**  
- Transaction costs and next-bar execution  
- Ablation heart: identical backbone with vs without FiLM  
- Leakage sanity check (shuffled labels)  

These practices are appropriate for a finance ML capstone and follow Lopez de Prado–style discipline.

---

## 3. Methodology Recap

### 3.1 Regime model

| State | Label | Economic reading |
|------:|-------|------------------|
| 0 | Low-vol bull | Strong trend, tight realized vol |
| 1 | Transition | Range / moderate vol |
| 2 | High-vol bear | Stress, crash/recovery dynamics |

FiLM applies per-block affine modulation \(h \leftarrow \gamma \cdot \mathrm{LN}(h) + \beta\), with \(\gamma,\beta\) driven by the HMM soft posterior.

### 3.2 Model set (ablation)

| Role | Model |
|------|--------|
| Economic floor | Buy & Hold |
| Null signal | Zero forecast |
| Econometrics | ARIMA–GARCH |
| Tabular ML | LightGBM |
| Strong linear TS baseline | DLinear |
| **Control** | PatchTST / Transformer **no regime** |
| **Thesis** | PatchTST / Transformer **+ FiLM** |
| Hard conditioning | PatchTST hard-switch |

---

## 4. Results Review

### 4.1 Walk-forward OOS ablation (2011–2021)

Primary metrics from `reports/ablation_table.csv` (≈5 bps costs, Kelly-lite, *n* ≈ 2,693 bars):

| Rank | Model | Sharpe | Sortino | Max DD | Ann. return | Hit rate |
|-----:|-------|-------:|--------:|-------:|------------:|---------:|
| 1 | Buy & Hold | **0.782** | 0.896 | −35.7% | 13.3% | 0.557 |
| 2 | ARIMA–GARCH | 0.497 | 0.554 | ≈0%* | ≈0%* | 0.540 |
| 3 | DLinear | 0.488 | 0.545 | −1.6% | 0.5% | 0.525 |
| 4 | PatchTST no-regime (**control**) | **0.410** | 0.405 | −18.0% | 2.8% | 0.540 |
| 5 | PatchTST + FiLM (**thesis**) | **0.290** | 0.288 | −17.6% | 1.9% | 0.555 |
| 6 | LightGBM | 0.199 | 0.273 | −1.2% | 0.1% | 0.476 |
| 7 | Zero | 0.000 | 0.000 | 0% | 0% | — |
| 8 | Hard-switch | **−0.060** | −0.067 | −13.3% | −0.3% | 0.508 |

\*ARIMA–GARCH’s near-zero absolute P&amp;L with moderate Sharpe reflects very low exposure / scale, not a deployable equity-like return stream.

**Architecture delta (official ablation):**  
\(\Delta = \mathrm{Sharpe}_{\mathrm{FiLM}} - \mathrm{Sharpe}_{\mathrm{control}} = 0.290 - 0.410 = \mathbf{-0.120}\).

FiLM improves hit rate slightly vs the control but **worsens** Sharpe, Sortino, and Calmar.

### 4.2 Cost sensitivity

From `reports/sensitivity_costs.csv` (reconstructed paths at 0 / 5 / 10 / 20 bps):

| Model | Survives 10 bps? | Comment |
|-------|------------------|---------|
| Buy & Hold | Yes | Most robust |
| PatchTST + FiLM | Yes (in this reconstruction) | Edge decays with cost but remains positive at 10 bps |
| PatchTST no-regime | Yes | Similar cost slope to FiLM |
| DLinear / LightGBM / Hard-switch | No | Fragile to realistic costs |

**Review note:** Cost-sensitivity Sharpes are not identical to the ablation table (different reconstruction/path assumptions). Treat the **ablation table + holdout** as the thesis ground truth; use cost curves for robustness of *edge vs friction*, not for re-ranking the thesis.

### 4.3 Statistical rigor

From notebook bootstrap CIs and `reports/eval_summary.json`:

- Stationary bootstrap CIs are wide for all active strategies → **high estimation uncertainty**.  
- **Deflated Sharpe (n_trials ≈ 40)** is near zero for most models → observed Sharpes do not survive a serious multiple-testing penalty.  
- Shuffled-label check drives Sharpe ≈ 0 → **no smoking-gun leakage**, which strengthens trust in the *negative* thesis result.  
- Monte Carlo drawdown percentile ≈ 0.40 → realized DD is not extreme vs a null of scrambled returns.

**Implication for grading/interpretation:** Even “winning” models are statistically fragile after DSR. Capstone value lies in the ablation design and honesty of the holdout, not in claiming a production alpha.

### 4.4 Per-regime attribution (FiLM OOS)

From `reports/regime_attribution.csv`:

| Regime | % time | Sharpe | Hit rate | Reading |
|--------|-------:|-------:|---------:|---------|
| Low-vol bull | ~6% | −0.36 | 0.47 | Poor in calm bull |
| Transition | ~55% | ~0.03 | 0.49 | Near flat |
| High-vol bear | ~38% | +0.19 | 0.52 | Mild positive |

FiLM’s relative contribution appears concentrated in stress regimes, but not enough to beat the control overall. The strategy spends most time in **transition**, where signal is weak.

### 4.5 FiLM interpretability (γ / β)

From `reports/film_params.csv` and `reports/film_divergence.csv`:

- Mean γ / β **divergence across regimes** is material (≈0.10–0.15 by block) → FiLM is **not** stuck at identity.  
- Later blocks show large negative γ means and distinct norms → conditioning is active.  

**Interpretation:** The mechanism *fires*; the problem is more likely **information quality of regimes** and **generalization**, not a dead FiLM layer.

### 4.6 Final holdout 2022–2024 (locked)

From `reports/final_holdout_summary.csv` / `final_holdout.json` (*n* ≈ 751–752):

| Rank | Model | Sharpe | 95% CI (approx.) | Ann. return |
|-----:|-------|-------:|------------------:|------------:|
| 1 | PatchTST no-regime | **0.724** | [−0.36, 1.87] | 0.50% |
| 2 | Buy & Hold | 0.684 | [−0.50, 1.84] | 0.47% |
| 3 | ARIMA–GARCH | 0.467 | [−0.58, 1.87] | 0.11% |
| 4 | DLinear | 0.035 | [−1.23, 1.09] | ≈0% |
| 5 | Zero | 0.000 | — | 0% |
| 6 | PatchTST + FiLM | **−0.145** | [−1.32, 0.98] | ≈0% |
| 7 | LightGBM | −0.174 | [−1.42, 0.87] | −0.09% |

**Holdout thesis delta:** \(0.724 - (-0.145) \Rightarrow\) FiLM trails the control by **≈ 0.87 Sharpe**.  
Period includes 2022 bear, 2023–24 AI-driven bull — a hard stress test for regimes fit on 2000–2021 structure.

Holdout regime labels collapse toward a single reported state in the summary export (all bars labelled low-vol bull). That is itself a warning: **downstream regime assignment / export may be oversimplified or non-stationary relative to training**, which can hurt FiLM more than a regime-agnostic model.

---

## 5. Critical Assessment

### 5.1 What worked well

1. **Experimental design** — purged CV, embargo, single-touch holdout, causal regimes.  
2. **Fair ablation** — control vs FiLM with shared backbone.  
3. **Honest negative result** — thesis rejected on both OOS ablation and holdout.  
4. **Broader baselines** — Buy & Hold, ARIMA–GARCH, LightGBM, DLinear prevent “Transformer-only” storytelling.  
5. **Interpretability** — γ/β analysis shows FiLM is learning *something* regime-specific.  
6. **Cost and leakage checks** — raise the bar above typical course projects.

### 5.2 Limitations and failure modes

1. **Thesis hypothesis fails** on the official metrics; soft conditioning adds parameters without OOS benefit.  
2. **Buy & Hold dominates OOS Sharpe**; ML strategies do not clearly beat the economic baseline after costs and sizing.  
3. **DSR ≈ 0** — statistical significance after multiple testing is lacking.  
4. **Hard switch is destructive** — confirms that discrete expert switching needs more data/regularization than available here.  
5. **Holdout regime representation** looks degenerate in the exported attribution — potential mismatch between live causal regimes and evaluation packaging.  
6. **Single-asset daily SPY** — limited diversity; regime signal may be too coarse or too noisy for FiLM.  
7. **Small OOS sample** (~2.7k bars) for a parameter-rich Transformer + FiLM.  
8. Metric inconsistency between ablation paths and cost-sensitivity reconstructions should be reconciled before publication.

### 5.3 Reviewer judgment

| Criterion | Rating | Comment |
|-----------|--------|---------|
| Problem motivation | Strong | Regime-before-price is a clear ML+finance story |
| Engineering / leakage control | Strong | Pipeline rules are professional |
| Empirical honesty | Excellent | Negative result clearly stated |
| Thesis support | Not supported | Control &gt; FiLM on OOS and holdout |
| Statistical power | Weak–moderate | Wide CIs, near-zero DSR |
| Deployability | Low | Does not beat Buy & Hold after costs in a convincing, DSR-safe way |

**Overall:** A **high-quality methodological capstone** with a **rejected product thesis**. That combination is defensible and teachable if discussed as such.

---

## 6. Suggestions (Immediate / Near-Term)

### 6.1 Analysis & reporting

1. **Reconcile Sharpe sources** — document why `ablation_table.csv` and `sensitivity_costs.csv` disagree; pick one as official and regener ate all figures from it.  
2. **Fix / audit holdout regime attribution** — verify filtered posteriors over 2022–2024 are exported correctly; if HMM collapses, report that as a scientific finding.  
3. **Lead with DSR and CIs** in the abstract so Sharpe rankings are not oversold.  
4. **Add Diebold–Mariano / SPA-style tests** for forecast loss, separate from trading Sharpe, so “predictive skill” and “strategy skill” are not conflated.  
5. **Report turnover and capacity** next to Sharpe (turnover already appears in `backtest_metrics.json`).

### 6.2 Regime modelling

1. Try **2- and 4-state HMMs**, plus BIC/AIC selection on train folds only.  
2. Condition on richer observations (VIX, credit spreads, breadth) rather than only (return, RV).  
3. Compare HMM to **change-point**, **volatility thresholds**, or **supervised regimes** (labelled crisis windows).  
4. Use **confidence gating**: apply FiLM only when max posterior &gt; threshold; otherwise fall back to the no-regime path.

### 6.3 Strategy layer

1. Replace Kelly-lite with simpler **sign(μ)/σ** or fixed risk to isolate forecast quality.  
2. Test **long-only** and **overlay** (always invested + regime tilt) vs full long/short.  
3. Explicitly compare against **Buy & Hold with volatility targeting** — often the fairest economic control.

---

## 7. Model Improvements

### 7.1 Architecture

| Idea | Rationale |
|------|-----------|
| Keep **no-regime PatchTST** as production candidate | Best neural model on holdout |
| Replace FiLM with **cross-attention to regime embedding** (already have config hooks) | Soft, lower-capacity conditioning |
| **Pinball / quantile heads** | Better σ for sizing; already contemplated in configs |
| **Adapter / LoRA-style regime adapters** | Fewer free parameters than full FiLM γ/β per block |
| **Mixture-of-experts with load balancing** | Soft experts without hard-switch collapse |
| **Multi-horizon PatchTST** | Stabilize μ/σ for strategy |
| Ensemble: **no-regime + FiLM gated by posterior entropy** | Use FiLM only when regimes are confident |

### 7.2 Training

1. Stronger regularization on FiLM parameters (L2 toward γ=1, β=0).  
2. Curriculum: pretrain no-regime, then freeze backbone and train FiLM only.  
3. Fold-wise early stopping on purged validation Sharpe **and** forecast CRPS/NLL.  
4. Balance loss across regimes (or oversample high-vol bars) to fight dominance of transition state.

### 7.3 Features

1. PIT-safe macro (rates, inflation surprise, liquidity).  
2. Options-implied regime proxies (VVIX, skew) if available under license.  
3. Fractional differentiation carefully purged per fold (already in spirit of the repo).

---

## 8. Future Work

### Short horizon (next 1–2 months)

- BTC hourly robustness study (`scripts/btc_robustness.py`) — different asset, different regime persistence.  
- Multi-asset panel (sector ETFs or futures) sharing a global regime encoder.  
- Publish a short “negative results” note: *when soft regime conditioning hurts PatchTST*.  
- Unit tests already exist for leakage — extend them to **holdout regime export** and metric alignment.

### Medium horizon

- Online / Bayesian HMM with recursive updates so 2022–2024 inflation/AI shifts are absorbed.  
- Causal validation: does regime *t* help forecast returns at *t+h* after controlling for RV alone? (If no, FiLM cannot help.)  
- Portfolio construction under turnover constraints; capacity study for SPY-like liquidity.

### Longer horizon

- Market microstructure: regime-conditioned execution / limit-order simulation.  
- Reinforcement learning that treats regime posterior as state — only after predictive regimes are validated.  
- Public leaderboard-style evaluation pack so future models reuse the same holdout protocol.

---

## 9. Recommended Narrative for Submission

Use language that matches the evidence:

1. **Claim the pipeline and evaluation design**, not a live trading edge.  
2. State the hypothesis clearly, then: *results do not support FiLM over the control*.  
3. Emphasize that **soft FiLM &gt; hard switch**, but **no-regime ≥ FiLM** on this sample.  
4. Frame FiLM γ/β divergence as evidence the model *responds* to regimes while still failing to improve decisions.  
5. Stress DSR / CI fragility as a lesson in quant ML evaluation.  
6. End with a concrete research agenda (Section 8) so the negative result looks like a starting line, not a dead end.

**Suggested one-sentence abstract:**  
*Using causal HMM regimes and FiLM-conditioned PatchTST on SPY 2000–2024, we find that regime soft-conditioning does not improve walk-forward or locked 2022–2024 Sharpe versus an identical unconditional PatchTST; a carefully designed negative result under purged CV, costs, and Deflated Sharpe.*

---

## 10. Conclusion

The capstone delivers a credible, leakage-aware research system and a clear experimental answer: **FiLM regime conditioning does not beat the no-regime PatchTST control on the official OOS ablation or the final holdout.** Buy & Hold remains hard to beat on OOS risk-adjusted terms; neural models show promise mainly in specific settings (e.g., holdout no-regime PatchTST), but not with statistical comfort after Deflated Sharpe.

**Suggestions and future work** should prioritize (i) auditing regime quality and holdout regime exports, (ii) lower-capacity / gated conditioning instead of full FiLM, (iii) fairer economic benchmarks (vol-targeted Buy & Hold), and (iv) multi-asset / BTC robustness. Improvement of models should focus less on stacking depth and more on **when regimes are usable** and **how few parameters** are needed to condition on them.

---

### Appendix A — File map for evidence

| Artifact | Use |
|----------|-----|
| `reports/ablation_table.csv` | Official OOS model ranking |
| `reports/sensitivity_costs.csv` | Cost robustness |
| `reports/regime_attribution.csv` | Per-regime P&amp;L |
| `reports/film_params.csv` / `film_divergence.csv` | FiLM interpretability |
| `reports/final_holdout.json` | Locked 2022–2024 results |
| `reports/eval_summary.json` | Bootstrap / DSR / shuffle checks |
| `notebooks/02_Model_Results_and_Evaluation.ipynb` | Narrative + figures |
| `notebooks/01_EDA_stylised_facts.ipynb` | Stylised facts / regime EDA |

### Appendix B — Thesis scorecard

| Check | Result |
|-------|--------|
| FiLM &gt; no-regime on OOS Sharpe | Fail (−0.12) |
| FiLM &gt; no-regime on holdout Sharpe | Fail (−0.87) |
| Soft FiLM &gt; hard switch | Pass |
| FiLM parameters diverge by regime | Pass (mechanism active) |
| Leakage (shuffled labels) | Pass (≈0 Sharpe) |
| Survive ~10 bps (sensitivity reconstruction) | Pass for FiLM & control |
| Beat Buy & Hold after costs (OOS) | Fail for thesis model |

---

*Report prepared from project outputs for capstone review. Methodology aligned with Lopez de Prado (2018), Advances in Financial Machine Learning.*
