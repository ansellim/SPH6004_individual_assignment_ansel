# ============================================================================
# 07_decision_tree.R
# decision tree (CART via rpart) evaluated on each of 4 feature sets
#
# workflow (repeated for stepwise/lasso/elastic/boruta):
#   1. hyperparameter tuning - 5 fold CV on df_train (cp x maxdepth grid)
#   2. final model fit on full df_train w/ best hyperparams
#   3. evaluate on df_test: AUC, Sensitivity, Specificity, PPV, NPV
#   4. save results to results/decision_tree_results.csv
# ============================================================================

start_time <- Sys.time()

library(rpart)
library(pROC)

set.seed(42)

target  <- "icu_death_flag"
N_FOLDS <- 5

# --------- 1. load data and feature sets ---------
df_train <- readRDS("data/train.rds")
df_test  <- readRDS("data/test.rds")

feature_sets <- list(
  stepwise = readRDS("data/selected_features_stepwise.rds"),
  lasso    = readRDS("data/selected_features_lasso.rds"),
  elastic  = readRDS("data/selected_features_elastic.rds"),
  boruta   = readRDS("data/selected_features_boruta.rds"),
  all      = readRDS("data/candidate_features.rds")
)


# helper
compute_metrics <- function(actual, prob, threshold = 0.5) {
  roc_obj <- roc(actual, prob, quiet = TRUE)
  auc_val <- as.numeric(auc(roc_obj))
  pred <- ifelse(prob >= threshold, 1L, 0L)
  tp <- sum(pred == 1 & actual == 1); tn <- sum(pred == 0 & actual == 0)
  fp <- sum(pred == 1 & actual == 0); fn <- sum(pred == 0 & actual == 1)
  list(AUC         = round(auc_val,        4),
       Sensitivity = round(tp/(tp+fn),     4),
       Specificity = round(tn/(tn+fp),     4),
       PPV         = round(tp/(tp+fp),     4),
       NPV         = round(tn/(tn+fn),     4))
}

dt_cp_grid <- c(0.0001, 0.001, 0.005, 0.01, 0.05)
dt_depth_grid  <- c(5, 10, 20, 30)

results <- data.frame()

# progress tracking
n_fs     <- length(feature_sets)
n_combos <- length(dt_cp_grid) * length(dt_depth_grid)
total_cv_fits  <- n_fs * n_combos * N_FOLDS
total_fits     <- total_cv_fits + n_fs
fit_counter    <- 0
loop_start     <- Sys.time()

# --- 2. loop over feature sets ---
for (fs_i in seq_along(feature_sets)) {
  fs_name  <- names(feature_sets)[fs_i]
  features <- feature_sets[[fs_name]]

  cat(sprintf("\n=== Feature set %d/%d: %s (%d features) ===\n",
              fs_i, n_fs, fs_name, length(features)))

  cols  <- c(features, target)
  df_tr <- df_train[, cols]
  df_te <- df_test[,  cols]

  # 5-fold CV tuning
  fold_idx <- sample(rep(1:N_FOLDS, length.out = nrow(df_tr)))
  tune_res <- data.frame()
  combo_i  <- 0

  for (cp_val in dt_cp_grid) {
    for (md_val in dt_depth_grid) {
      combo_i <- combo_i + 1
      fold_aucs <- numeric(N_FOLDS)
      for (f in 1:N_FOLDS) {
        fit <- rpart(as.formula(paste(target, "~ .")),
                     data    = df_tr[fold_idx != f, ],
                     method  = "class",
                     control = rpart.control(cp = cp_val, maxdepth = md_val))
        prob         <- predict(fit, df_tr[fold_idx == f, ], type = "prob")[, "Discharged"]
        actual_f     <- as.integer(df_tr[fold_idx == f, target] == "Discharged")
        fold_aucs[f] <- as.numeric(auc(roc(actual_f, prob, quiet = TRUE)))
        fit_counter  <- fit_counter + 1
      }
      elapsed <- as.numeric(difftime(Sys.time(), loop_start, units = "mins"))
      eta     <- elapsed / fit_counter * (total_fits - fit_counter)
      cat(sprintf("  [%s] combo %d/%d (cp=%.4f, maxdepth=%d) CV_AUC=%.4f | fit %d/%d  elapsed %.1f min  ETA %.1f min\n",
                  fs_name, combo_i, n_combos, cp_val, md_val, mean(fold_aucs),
                  fit_counter, total_fits, elapsed, eta))

      tune_res <- rbind(tune_res,
        data.frame(cp = cp_val, maxdepth = md_val, CV_AUC = mean(fold_aucs)))
    }
  }
  best <- tune_res[which.max(tune_res$CV_AUC), ]
  cat(sprintf("  >> best: cp=%.4f, maxdepth=%d, CV_AUC=%.4f\n",
              best$cp, best$maxdepth, best$CV_AUC))

  # final model on full train set
  cat(sprintf("  fitting final model on %d rows...\n", nrow(df_tr)))
  final_fit <- rpart(as.formula(paste(target, "~ .")),
                     data    = df_tr,
                     method  = "class",
                     control = rpart.control(cp = best$cp, maxdepth = best$maxdepth))
  fit_counter <- fit_counter + 1

  prob   <- predict(final_fit, df_te, type = "prob")[, "Discharged"]
  actual <- as.integer(df_te[[target]] == "Discharged")
  m      <- compute_metrics(actual, prob)
  cat(sprintf("  >> test AUC=%.4f  Sens=%.4f  Spec=%.4f\n", m$AUC, m$Sensitivity, m$Specificity))


  results <- rbind(results, data.frame(
    feature_set = fs_name,
    n_features  = length(features),
    features    = paste(features, collapse = ";"),
    best_cp     = best$cp,
    best_depth  = best$maxdepth,
    AUC         = m$AUC,
    Sensitivity = m$Sensitivity,
    Specificity = m$Specificity,
    PPV         = m$PPV,
    NPV         = m$NPV,
    stringsAsFactors = FALSE
  ))
}

# 3. save
dir.create("results", showWarnings = FALSE)
write.csv(results, "results/decision_tree_results.csv", row.names = FALSE)
print(results[, c("feature_set", "n_features", "AUC", "Sensitivity",
                  "Specificity", "PPV", "NPV")])

print(Sys.time() - start_time)
