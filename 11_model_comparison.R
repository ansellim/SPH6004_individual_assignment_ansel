# ============================================================================
# 11_model_comparison.R
# collect results from all model CSVs, produce final comparison table
#
# inputs: results/{logistic_regression,decision_tree,adaboost,xgboost,svm}_results.csv
# outputs: results/model_comparison.csv
#   - one row per (model x feature_set) combo
#   - plus one AVERAGE row per model accross the 4 feature sets
# ============================================================================

library(dplyr)

# ---- 1. load and label each model's results ----
metric_cols <- c("AUC", "Sensitivity", "Specificity", "PPV", "NPV")
keep_cols   <- c("model", "feature_set", "n_features", metric_cols)

read_results <- function(path, model_name) {
  df <- read.csv(path, stringsAsFactors = FALSE)
  df$model <- model_name
  df[, keep_cols]
}

all_results <- bind_rows(
  read_results("results/logistic_regression_results.csv", "Logistic Regression"),
  read_results("results/decision_tree_results.csv",       "Decision Tree"),
  read_results("results/adaboost_results.csv",            "AdaBoost"),
  read_results("results/xgboost_results.csv",             "XGBoost"),
  read_results("results/svm_results.csv",                 "SVM")
)

# 2. average metrics per model across feature sets
avg_results <- all_results %>%
  group_by(model) %>%
  summarise(
    feature_set = "AVERAGE",
    n_features  = round(mean(n_features), 1),
    AUC         = round(mean(AUC),         4),
    Sensitivity = round(mean(Sensitivity), 4),
    Specificity = round(mean(Specificity), 4),
    PPV         = round(mean(PPV),         4),
    NPV         = round(mean(NPV),         4),
    .groups = "drop"
  )

# --- 3. combine and order ---
model_order <- c("Logistic Regression", "Decision Tree", "AdaBoost", "XGBoost", "SVM")
fs_order    <- c("stepwise", "lasso", "elastic", "boruta", "AVERAGE")

final_table <- bind_rows(all_results, avg_results) %>%
  mutate(model       = factor(model,       levels = model_order),
         feature_set = factor(feature_set, levels = fs_order)) %>%
  arrange(model, feature_set) %>%
  mutate(model = as.character(model), feature_set = as.character(feature_set))

# 4. print and save
print(as.data.frame(final_table[, keep_cols]), row.names = FALSE)

dir.create("results", showWarnings = FALSE)
write.csv(final_table[, keep_cols], "results/model_comparison.csv", row.names = FALSE)

# quick best-combination summary
best <- all_results[which.max(all_results$AUC), ]

best <- all_results[which.max(all_results$Sensitivity), ]

best_avg <- avg_results[which.max(avg_results$AUC), ]
