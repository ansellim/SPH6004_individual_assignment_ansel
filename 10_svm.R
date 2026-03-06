# =========================================================================
# 10_svm.R
# SVM (radial kernel) evaluated on 4 feature sets
#
# workflow (stepwise/lasso/elastic/boruta):
#   1. hyperparam tuning - 5fold CV on subsample (cost x gamma)
#   2. final model on same subsample with best hyperparams
#   3. eval on full df_test
#   4. save results
#
# SVM is too slow on 70k+ rows so we subsample SVM_TRAIN_N rows
# =========================================================================

start_time <- Sys.time()

library(e1071)
library(pROC)

set.seed(42)

target      <- "icu_death_flag"
N_FOLDS     <- 3
SVM_TRAIN_N <- 10000

# --- 1. load data and feature sets
#     Use partially-balanced train set (over_ratio=0.5) — smaller, faster for SVM
df_train <- readRDS("data/train_small.rds")
df_test  <- readRDS("data/test.rds")

feature_sets <- list(
  all      = readRDS("data/candidate_features.rds"),
  stepwise = readRDS("data/selected_features_stepwise.rds"),
  lasso    = readRDS("data/selected_features_lasso.rds"),
  elastic  = readRDS("data/selected_features_elastic.rds"),
  boruta   = readRDS("data/selected_features_boruta.rds")
)


# helper
compute_metrics <- function(actual, prob, threshold = 0.5) {
  roc_obj <- roc(actual, prob, quiet = TRUE)
  auc_val <- as.numeric(auc(roc_obj))
  pred <- ifelse(prob >= threshold, 1L, 0L)
  tp <- sum(pred == 1 & actual == 1); tn <- sum(pred == 0 & actual == 0)
  fp <- sum(pred == 1 & actual == 0); fn <- sum(pred == 0 & actual == 1)
  list(AUC         = round(auc_val,    4),
       Sensitivity = round(tp/(tp+fn), 4),
       Specificity = round(tn/(tn+fp), 4),
       PPV         = round(tp/(tp+fp), 4),
       NPV         = round(tn/(tn+fn), 4))
}

svm_cost_grid  <- c(0.1, 1, 10)
svm_gamma_grid <- c(0.01, 0.1)

# ------------- Runtime estimate (times 1 CV fit + 1 final fit, then extrapolates)
cat("=== SVM Runtime Estimate ===\n")
est_fs   <- feature_sets[[1]]
est_cols <- c(est_fs, target)
set.seed(99)
est_idx  <- sample(nrow(df_train[, est_cols]), SVM_TRAIN_N)
est_df   <- df_train[est_idx, est_cols]
est_fold <- sample(rep(1:N_FOLDS, length.out = nrow(est_df)))

t0 <- Sys.time()
svm(as.formula(paste(target, "~ .")), data = est_df[est_fold != 1, ],
    kernel = "radial", cost = 1, gamma = 0.1, probability = FALSE, scale = FALSE)
cv_fit_sec <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

t0 <- Sys.time()
svm(as.formula(paste(target, "~ .")), data = est_df,
    kernel = "radial", cost = 1, gamma = 0.1, probability = TRUE, scale = FALSE)
final_fit_sec <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

n_cv_total    <- length(feature_sets) * length(svm_cost_grid) * length(svm_gamma_grid) * N_FOLDS
n_final_total <- length(feature_sets)
est_min       <- (cv_fit_sec * n_cv_total + final_fit_sec * n_final_total) / 60

cat(sprintf("  CV fit: %.1f sec | final fit: %.1f sec\n", cv_fit_sec, final_fit_sec))
cat(sprintf("  Total fits: %d CV + %d final = %d\n", n_cv_total, n_final_total, n_cv_total + n_final_total))
cat(sprintf("  Estimated runtime: %.1f min (%.1f hrs)\n", est_min, est_min / 60))
cat("=== Starting full run ===\n\n")

rm(est_fs, est_cols, est_idx, est_df, est_fold, cv_fit_sec, final_fit_sec,
   n_cv_total, n_final_total, est_min)

results <- data.frame()

# progress tracking
n_fs     <- length(feature_sets)
n_combos <- length(svm_cost_grid) * length(svm_gamma_grid)
total_cv_fits  <- n_fs * n_combos * N_FOLDS
total_fits     <- total_cv_fits + n_fs   # cv fits + final fits
fit_counter    <- 0
loop_start     <- Sys.time()

# ------------- 2. loop over feature sets
for (fs_i in seq_along(feature_sets)) {
  fs_name  <- names(feature_sets)[fs_i]
  features <- feature_sets[[fs_name]]

  cat(sprintf("\n=== Feature set %d/%d: %s (%d features) ===\n",
              fs_i, n_fs, fs_name, length(features)))

  cols  <- c(features, target)
  df_tr <- df_train[, cols]
  df_te <- df_test[,  cols]

  # fixed subsample + fold assignments
  set.seed(42)
  sub_idx  <- sample(nrow(df_tr), SVM_TRAIN_N)
  df_svm   <- df_tr[sub_idx, ]
  fold_idx <- sample(rep(1:N_FOLDS, length.out = nrow(df_svm)))

  tune_res <- data.frame()
  combo_i  <- 0

  for (c_val in svm_cost_grid) {
    for (g_val in svm_gamma_grid) {
      combo_i <- combo_i + 1
      fold_aucs <- numeric(N_FOLDS)
      for (f in 1:N_FOLDS) {
        # probability=FALSE + decision.values for CV: ~2x faster (skips Platt scaling)
        # scale=FALSE: data already z-scored in 01_data_split.R
        fit      <- svm(as.formula(paste(target, "~ .")),
                        data        = df_svm[fold_idx != f, ],
                        kernel      = "radial", cost = c_val, gamma = g_val,
                        probability = FALSE, scale = FALSE)
        pred_val     <- predict(fit, df_svm[fold_idx == f, ], decision.values = TRUE)
        dv           <- attr(pred_val, "decision.values")[, 1]
        actual_f     <- as.integer(df_svm[fold_idx == f, target] == "Discharged")
        fold_aucs[f] <- as.numeric(auc(roc(actual_f, dv, quiet = TRUE)))
        fit_counter  <- fit_counter + 1
      }
      # progress for this combo
      elapsed <- as.numeric(difftime(Sys.time(), loop_start, units = "mins"))
      eta     <- elapsed / fit_counter * (total_fits - fit_counter)
      cat(sprintf("  [%s] combo %d/%d (cost=%.2f, gamma=%.2f) CV_AUC=%.4f | fit %d/%d  elapsed %.1f min  ETA %.1f min\n",
                  fs_name, combo_i, n_combos, c_val, g_val, mean(fold_aucs),
                  fit_counter, total_fits, elapsed, eta))

      tune_res <- rbind(tune_res,
        data.frame(cost = c_val, gamma = g_val, CV_AUC = mean(fold_aucs)))
    }
  }
  best <- tune_res[which.max(tune_res$CV_AUC), ]
  cat(sprintf("  >> best: cost=%.2f, gamma=%.2f, CV_AUC=%.4f\n",
              best$cost, best$gamma, best$CV_AUC))

  # final model on full subsample
  cat(sprintf("  fitting final model on %d rows...\n", nrow(df_svm)))
  final_fit <- svm(as.formula(paste(target, "~ .")),
                   data        = df_svm,
                   kernel      = "radial", cost = best$cost, gamma = best$gamma,
                   probability = TRUE, scale = FALSE)
  fit_counter <- fit_counter + 1

  pred_te <- predict(final_fit, df_te, probability = TRUE)
  prob    <- attr(pred_te, "probabilities")[, "Discharged"]
  actual  <- as.integer(df_te[[target]] == "Discharged")
  m       <- compute_metrics(actual, prob)
  cat(sprintf("  >> test AUC=%.4f  Sens=%.4f  Spec=%.4f\n", m$AUC, m$Sensitivity, m$Specificity))


  results <- rbind(results, data.frame(
    feature_set = fs_name,
    n_features  = length(features),
    features    = paste(features, collapse = ";"),
    best_cost   = best$cost,
    best_gamma  = best$gamma,
    AUC         = m$AUC,
    Sensitivity = m$Sensitivity,
    Specificity = m$Specificity,
    PPV         = m$PPV,
    NPV         = m$NPV,
    stringsAsFactors = FALSE
  ))
}

# ----- 3. save
dir.create("results", showWarnings = FALSE)
write.csv(results, "results/svm_results.csv", row.names = FALSE)
print(results[, c("feature_set", "n_features", "AUC", "Sensitivity",
                  "Specificity", "PPV", "NPV")])

print(Sys.time() - start_time)
