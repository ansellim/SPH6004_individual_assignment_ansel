# SPH6004 Individual Assignment 1 — Predicting ICU Discharge with Machine Learning

Author: Ansel Lim. Code and analysis for SPH6004 Assignment 1, comparing classical and ensemble ML models for predicting ICU discharge status on a MIMIC-IV–derived cohort.

## Overview

Each record in the working dataset is a temporally-flattened snapshot of a patient at ICU admission (demographics, SOFA component scores, vitals, glucose, GCS, hematology, biochemistry, coagulation, and arterial blood gas). The binary outcome is ICU discharge vs. in-ICU death (`icu_death_flag`). Accurate prediction supports resource planning, step-down-care decisions, and prognostication at the bedside.

The pipeline covers preprocessing, four feature-selection strategies, six model families, and a head-to-head evaluation on a held-out test set.

## Data pipeline

Source: MIMIC-IV extract (`data/Assignment1_mimic dataset.csv`) — not included in the repo.

1. **Cohort filtering.** Restrict to the eight recognized ICU types (MICU, CVICU, MICU/SICU, SICU, TSICU, CCU, Neuro SICU, general ICU); drop records with missing discharge timestamps.
2. **Column pruning.** Remove identifiers (`subject_id`, `hadm_id`, `stay_id`), care-unit and timing variables, outcome-leakage variables (`los`, `deathtime`, `hospital_expire_flag`), and ethically-excluded variables (`insurance`, `language`, `marital_status`).
3. **Race recoding.** Collapse 33 original categories into 7 groups (Asian, Black, Hispanic/Latino, Native American, White, Other, Unknown).
4. **Split.** 70/30 stratified train/test split.
5. **Missingness filter.** Drop features with >10% missingness in the training set (applied identically to test).
6. **Imputation and scaling.** Median imputation and z-score standardization fit on the training set only, then applied to test.
7. **Class balancing.** SMOTENC on the training set to upsample the minority class (ICU death); handles mixed numeric/categorical features by interpolating numerics and majority-voting categoricals among k-NN.

## Feature selection

Rationale: reduce multicollinearity (e.g., correlated SOFA components), mitigate overfitting, lower inference cost for clinical deployment, and improve interpretability.

Four strategies were applied, all using logistic regression as the base:

| Method | Mechanism |
|---|---|
| Forward stepwise | Greedy AIC-minimizing addition of features from a null model |
| Lasso (L1) | L1-penalized logistic regression; shrinks unimportant coefficients to zero |
| Elastic Net (L1+L2) | Combined penalty with mixing parameter chosen by cross-validation |
| Boruta | Random-forest wrapper comparing feature importance against permuted shadow features |

**What got selected.** SOFA scores and GCS subscores were retained universally, consistent with their established role as predictors of organ dysfunction and neurological status. Stepwise and lasso were the most aggressive, dropping gender and the maximum values of several vitals (diastolic BP, glucose, SpO2; lasso additionally dropped maxima of MAP, systolic BP, and temperature) as redundant with other summary statistics. Elastic net and Boruta retained all candidate features.

## Models

Six model families were trained on each of five feature sets (all candidate features + four selected subsets):

- **Logistic regression** — linear baseline, assumes linearity of log-odds and feature independence.
- **Decision tree** — axis-aligned recursive partitioning.
- **SVM** — kernelized maximum-margin classifier.
- **AdaBoost** — sequential reweighting of weak stumps (bias-reduction ensemble).
- **XGBoost** — regularized, sub-sampled gradient-boosted trees.
- **Random Forest** — bagged ensemble of independently grown trees with random feature subsets (variance-reduction ensemble).

## Findings

Performance on the 30% held-out test set, averaged across feature sets:

- **Logistic regression** — AUC ~0.87, Sensitivity ~0.80, Specificity ~0.77, PPV ~0.97, NPV ~0.29. Performance was essentially unchanged across feature sets, meaning stepwise (32 features) and lasso (26 features) preserved full-model accuracy with a much smaller input surface.
- **Decision tree** — lower AUC than logistic regression but consistently higher sensitivity and NPV; the gain comes at a cost of lower specificity (more false discharges predicted).
- **SVM, AdaBoost, XGBoost** — same feature-set pattern as above: aggressive feature reduction does not meaningfully degrade performance.
- **Boosted ensembles (AdaBoost, XGBoost)** — outperform both logistic regression and single decision trees on AUC and sensitivity. The largest gain is in **NPV** (AdaBoost ~0.53, XGBoost ~0.64 vs. logistic regression ~0.28 and decision tree ~0.35), i.e., boosted models are substantially better at correctly identifying patients who will die in the ICU — the clinically safer error mode.
- **Random Forest** — best AUC, sensitivity, specificity, and NPV overall, suggesting that variance reduction via bagging is the dominant benefit on this cohort.

**Clinical takeaway.** Model choice should track the clinical use case: logistic regression gives a parsimonious, interpretable baseline; a single decision tree maximizes sensitivity for discharge; boosted and bagged ensembles offer the strongest overall performance, with Random Forest the most promising target for further hyperparameter tuning.

**Caveat.** The dataset is a single-snapshot, temporally-flattened view at ICU admission; it cannot capture trajectory-dependent signals that evolve over an ICU stay.

## Repository structure

```
00_eda.R                                     # exploratory data analysis
01_data_split.R                              # cohort filter, pruning, split, imputation, scaling, SMOTENC
02_feature_selection_stepwise_logistic.R     # forward stepwise (AIC)
03_feature_selection_lasso_logistic.R        # L1
04_feature_selection_elastic_logistic.R      # elastic net
05_feature_selection_boruta.R                # Boruta
06_logistic_regression.R                     # logistic regression, 5 feature sets
07_decision_tree.R / 07_decision_tree_candidate.R
08_adaboost.R
09_xgboost.R
10_svm.R
11_random_forest.R
12_model_comparison.R                        # aggregates per-model CSVs, ROC plots
helpers.R                                    # shared utilities
report.Rmd                                   # full write-up with tables and figures
renv.lock                                    # locked R package versions
run_on_server.sh                             # headless execution on a compute server
AI_USE_LOG.md                                # record of AI assistance
```

Scripts are numbered `01`–`11` and must be run in order; `12` assumes all per-model result CSVs are already written.

## Reproducing the analysis

1. Place the MIMIC-IV extract at `data/Assignment1_mimic dataset.csv`.
2. Restore R dependencies: `renv::restore()` (uses `renv.lock`).
3. Run the scripts in numbered order (`00_eda.R` through `11_random_forest.R`), then `12_model_comparison.R`.
4. Knit `report.Rmd` to produce the PDF/HTML write-up with tables and figures.

On a headless server, `run_on_server.sh` runs the full pipeline end-to-end.

## AI use

AI tools (Anthropic Claude Chat and Claude Code) were used for understanding documentation, generating repetitive or boilerplate code from author-written analytical plans, formatting, and language editing. All analytical decisions and the analysis plan are the author's own. See `AI_USE_LOG.md` for details.
