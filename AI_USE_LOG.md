# AI Use Log — SPH6004 Individual Assignment

**Student:** Ansel Lim
**Assignment:** SPH6004 Individual Assignment — Predicting ICU Discharge Using Machine Learning
**AI Tool Used:** Claude Code (Model: `claude-sonnet-4-6`, Anthropic), accessed via the Claude Code CLI
**Date Range:** March 2026

---

## Overview

This log documents the use of generative AI (Claude Code) during the development of this assignment. Claude Code was used as a coding assistant to help write, debug, and refactor R scripts, and to assist with structuring the written report. All modelling decisions, interpretation of results, and written analysis were reviewed and verified by the student.

---

## Log of AI Interactions

### 1. Project Setup and Preprocessing (`01_data_split.R`)

**Purpose:** Design and implement the data preprocessing pipeline.

**Representative prompts used:**
- *"Create an R script that reads the MIMIC-IV dataset, filters to ICU patients only, removes identifier columns, leakage variables, and ethically sensitive variables (insurance, language, marital status), then does a stratified 70/30 train-test split."*
- *"After the split, fit median imputation and z-score standardisation on the training set only and apply to both train and test. Then apply SMOTENC on the training set to address class imbalance, oversampling the minority class (DiedinICU) to a 1:1 ratio."*
- *"Recode the race variable into 7 broad groups before splitting."*

**AI contribution:** Generated the full preprocessing pipeline including stratified splitting with `caret`, median imputation, z-score scaling, and SMOTENC via the `themis` package. The student verified that no data leakage occurred (preprocessing fit only on training data).

---

### 2. Feature Selection Scripts (`02`–`05`)

**Purpose:** Implement four feature selection strategies: forward stepwise (AIC), LASSO, elastic net, and Boruta.

**Representative prompts used:**
- *"Write an R script that does forward stepwise logistic regression using AIC on the SMOTENC-balanced training data and saves the selected feature names (original column names, not dummy-expanded names) to an RDS file."*
- *"Write a LASSO logistic regression feature selection script using `glmnet`. Use 5-fold CV to find the best lambda. Extract the non-zero coefficient names and collapse dummy variable names back to the original column name (e.g., 'raceBlack' → 'race')."*
- *"Do the same for elastic net but also tune the alpha parameter via cross-validation."*
- *"Implement Boruta feature selection using the `Boruta` package on the training data and save confirmed features."*

**AI contribution:** Generated all four feature selection scripts. The student reviewed the dummy-collapsing logic (`gsub` pattern) to ensure it correctly mapped dummy columns back to original factor columns for compatibility with downstream modelling scripts.

---

### 3. Model Training Scripts (`06`–`11`)

**Purpose:** Implement logistic regression, decision tree (CART), AdaBoost, XGBoost, SVM, and random forest models across all five feature sets.

**Representative prompts used:**
- *"Write an R script that trains a logistic regression model using `glm` for each of the 5 feature sets (all, stepwise, lasso, elastic, boruta) and evaluates on the test set. Report AUC, sensitivity, specificity, PPV, and NPV at threshold 0.5."*
- *"Write a decision tree script using `rpart`. Tune the complexity parameter (cp) using 5-fold CV on the training data."*
- *"Implement AdaBoost using the `adabag` package. Tune mfinal and tree depth using 5-fold CV on a 10,000-row subsample of training data due to computational cost. Train the final model on the full training set."*
- *"Implement XGBoost with a 2-stage hyperparameter tuning approach: first tune tree structure parameters, then learning rate. Use 5-fold CV."*
- *"Implement SVM with radial basis function kernel using `e1071`. Tune cost and gamma via 5-fold CV on a 15,000-row subsample."*
- *"Add a random forest model using `randomForest`. Tune `mtry` and `ntree` via CV."*
- *"Refactor all model scripts to use a shared `helpers.R` file containing common metric computation and result-saving functions."*

**AI contribution:** Generated all model training scripts. The student specified the design decisions (CV strategy, subsampling approach for computationally expensive models, threshold for classification). The student reviewed and approved the metric computation logic, particularly the direction of predicted probabilities (`P(Discharged)` vs `P(DiedinICU)`).

---

### 4. Debugging and Refactoring

**Purpose:** Fix errors that arose during execution on the compute server and improve code quality.

**Representative prompts used:**
- *"The logistic regression script is failing because some feature sets include 'elastic' which selected all 38 features, the same as 'all'. Fix the script to handle duplicate feature sets gracefully."*
- *"The AdaBoost script is predicting P(DiedinICU) but we need P(Discharged) for AUC to be above 0.5. Fix this."*
- *"Refactor all model scripts to remove duplicated metric-computation code into a shared helpers.R."*
- *"The SVM predict output does not include probabilities by default. Set `probability = TRUE` in `svm()` and use `attr(pred, 'probabilities')` to extract them."*

**AI contribution:** Identified and fixed bugs, refactored duplicated code into `helpers.R`. The student tested the fixes by re-running scripts on the server.

---

### 5. Model Comparison (`12_model_comparison.R`)

**Purpose:** Aggregate results from all models and feature sets into a single comparison table and generate ROC plots.

**Representative prompts used:**
- *"Write a script that reads all model result CSVs from the results/ folder, joins them, and outputs a model_comparison.csv with one row per model × feature set combination."*
- *"Add a ROC curve comparison plot that overlays ROC curves for all models on the best feature set."*

**AI contribution:** Generated the aggregation and plotting script. The student reviewed the output table and confirmed the results matched expectations from individual model runs.

---

### 6. Report Writing (`report-anselnus.Rmd`)

**Purpose:** Assist with structuring and drafting sections of the written report.

**Representative prompts used:**
- *"Help me write a methods section explaining the SMOTENC oversampling approach and why it was applied only to the training set."*
- *"Write a paragraph explaining the difference between LASSO, elastic net, and stepwise feature selection, and justify why all three were included."*
- *"Summarise the model comparison results in a results section. The best model is XGBoost with LASSO features (AUC ~0.92)."*
- *"Add a limitations section discussing the static snapshot nature of the dataset and the use of SMOTENC."*

**AI contribution:** Drafted sections of the report which the student then reviewed, edited, and verified for accuracy. The student made final decisions on all interpretive claims. AI-generated text was revised to reflect the student's own voice and understanding.

---

## Summary

| Task | AI Used? | Student Oversight |
|---|---|---|
| Preprocessing pipeline design | Yes (code generation) | Reviewed for data leakage |
| Feature selection implementation | Yes (code generation) | Verified dummy-collapsing logic |
| Model training scripts | Yes (code generation) | Specified all design decisions |
| Debugging | Yes (debugging assistance) | Tested all fixes |
| Model comparison & visualisation | Yes (code generation) | Verified outputs |
| Report writing | Yes (drafting assistance) | Reviewed, edited, and approved all text |
| Result interpretation | No | Student only |
| Modelling decisions | No | Student only |

---

## Statement

All AI-generated code and text was reviewed by the student before use. The student takes responsibility for the correctness and appropriateness of all content in this submission. AI was used as a productivity tool to accelerate implementation, not to substitute for the student's own understanding of the methods.
