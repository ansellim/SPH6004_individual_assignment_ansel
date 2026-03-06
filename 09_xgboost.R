# ==============================================================================
# 09_xgboost.R
# XGBoost evaluated on 4 feature sets
#
# workflow (stepwise/lasso/elastic/boruta):
#   1. stage 1: tune max_depth x eta via xgb.cv (5fold, early stopping)
#   2. stage 2: tune alpha(L1) x lambda(L2) via xgb.cv
#   3. final model on full df_train w/ best hyperparams
#   4. evaluate on df_test
#   5. save results
# ==============================================================================

start_time <- Sys.time()

library(xgboost)
library(pROC)

set.seed(42)

target <- "icu_death_flag"

# ---- 1. Load data and feature sets
df_train <- readRDS("data/train.rds")
df_test  <- readRDS("data/test.rds")

feature_sets <- list(
  stepwise = readRDS("data/selected_features_stepwise.rds"),
  lasso    = readRDS("data/selected_features_lasso.rds"),
  elastic  = readRDS("data/selected_features_elastic.rds")
  ,boruta   = readRDS("data/selected_features_boruta.rds")
)


# helpers
make_xgb_matrix <- function(df, tgt) {
  X <- model.matrix(as.formula(paste("~", paste(setdiff(names(df), tgt), collapse = "+"))),
                    data = df)[, -1]
  y <- as.integer(df[[tgt]] == "Discharged")
  xgb.DMatrix(data = X, label = y)
}

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

xgb_depth_grid <- c(3, 5, 7)
xgb_eta_grid   <- c(0.01, 0.05, 0.1)
xgb_alpha_grid <- c(0, 0.1, 1)
xgb_lambda_grid <- c(0.1, 1, 10)
xgb_nrounds <- 500

results <- data.frame()

# progress tracking -- xgb.cv counts as 1 "fit" each, plus 1 final xgb.train per feature set
n_fs        <- length(feature_sets)
n_s1_combos <- length(xgb_depth_grid) * length(xgb_eta_grid)
n_s2_combos <- length(xgb_alpha_grid) * length(xgb_lambda_grid)
total_fits  <- n_fs * (n_s1_combos + n_s2_combos + 1)
fit_counter <- 0
loop_start  <- Sys.time()

# 2. loop over feature sets
for (fs_i in seq_along(feature_sets)) {
  fs_name  <- names(feature_sets)[fs_i]
  features <- feature_sets[[fs_name]]

  cat(sprintf("\n=== Feature set %d/%d: %s (%d features) ===\n",
              fs_i, n_fs, fs_name, length(features)))

  cols   <- c(features, target)
  dtrain <- make_xgb_matrix(df_train[, cols], target)
  dtest  <- make_xgb_matrix(df_test[,  cols], target)

  # stage 1: max_depth x eta
  s1_res  <- data.frame()
  combo_i <- 0
  cat("  Stage 1: depth x eta\n")
  for (d_val in xgb_depth_grid) {
    for (e_val in xgb_eta_grid) {
      combo_i <- combo_i + 1
      cv <- xgb.cv(
        params = list(objective="binary:logistic", eval_metric="auc",
                      max_depth=d_val, eta=e_val, subsample=0.7,
                      min_child_weight=20, alpha=0, lambda=1),
        data = dtrain, nfold = 5, nrounds = xgb_nrounds,
        early_stopping_rounds = 20, verbose = 0)
      best_it <- cv$early_stop$best_iteration
      cv_auc  <- cv$evaluation_log$test_auc_mean[best_it]
      fit_counter <- fit_counter + 1
      elapsed <- as.numeric(difftime(Sys.time(), loop_start, units = "mins"))
      eta_rem <- elapsed / fit_counter * (total_fits - fit_counter)
      cat(sprintf("    combo %d/%d (depth=%d, eta=%.2f) CV_AUC=%.4f nrounds=%d | fit %d/%d  elapsed %.1f min  ETA %.1f min\n",
                  combo_i, n_s1_combos, d_val, e_val, cv_auc, best_it,
                  fit_counter, total_fits, elapsed, eta_rem))
      s1_res  <- rbind(s1_res, data.frame(max_depth=d_val, eta=e_val,
                                          best_iter=best_it, CV_AUC=cv_auc))
    }
  }
  best_s1 <- s1_res[which.max(s1_res$CV_AUC), ]
  cat(sprintf("  >> S1 best: depth=%d, eta=%.2f, CV_AUC=%.4f\n",
              best_s1$max_depth, best_s1$eta, best_s1$CV_AUC))

  # stage 2: alpha(L1) x lambda(L2)
  s2_res  <- data.frame()
  combo_i <- 0
  cat("  Stage 2: alpha x lambda\n")
  for (a_val in xgb_alpha_grid) {
    for (l_val in xgb_lambda_grid) {
      combo_i <- combo_i + 1
      cv <- xgb.cv(
        params = list(objective="binary:logistic", eval_metric="auc",
                      max_depth=best_s1$max_depth, eta=best_s1$eta,
                      subsample=0.7, min_child_weight=20,
                      alpha=a_val, lambda=l_val),
        data = dtrain, nfold = 5, nrounds = xgb_nrounds,
        early_stopping_rounds = 20, verbose = 0)
      best_it <- cv$early_stop$best_iteration
      cv_auc  <- cv$evaluation_log$test_auc_mean[best_it]
      fit_counter <- fit_counter + 1
      elapsed <- as.numeric(difftime(Sys.time(), loop_start, units = "mins"))
      eta_rem <- elapsed / fit_counter * (total_fits - fit_counter)
      cat(sprintf("    combo %d/%d (alpha=%.1f, lambda=%.1f) CV_AUC=%.4f nrounds=%d | fit %d/%d  elapsed %.1f min  ETA %.1f min\n",
                  combo_i, n_s2_combos, a_val, l_val, cv_auc, best_it,
                  fit_counter, total_fits, elapsed, eta_rem))
      s2_res  <- rbind(s2_res, data.frame(alpha=a_val, lambda=l_val,
                                          best_iter=best_it, CV_AUC=cv_auc))
    }
  }
  best_s2 <- s2_res[which.max(s2_res$CV_AUC), ]
  cat(sprintf("  >> S2 best: alpha=%.1f, lambda=%.1f, CV_AUC=%.4f\n",
              best_s2$alpha, best_s2$lambda, best_s2$CV_AUC))

  # Final model
  cat("  fitting final model...\n")
  final_fit <- xgb.train(
    params  = list(objective="binary:logistic", eval_metric="auc",
                   max_depth=best_s1$max_depth, eta=best_s1$eta,
                   subsample=0.7, min_child_weight=20,
                   alpha=best_s2$alpha, lambda=best_s2$lambda),
    data    = dtrain,
    nrounds = best_s2$best_iter,
    verbose = 0)
  fit_counter <- fit_counter + 1

  prob   <- predict(final_fit, dtest)
  actual <- as.integer(getinfo(dtest, "label"))
  m      <- compute_metrics(actual, prob)
  cat(sprintf("  >> test AUC=%.4f  Sens=%.4f  Spec=%.4f\n", m$AUC, m$Sensitivity, m$Specificity))


  results <- rbind(results, data.frame(
    feature_set  = fs_name,
    n_features   = length(features),
    features     = paste(features, collapse = ";"),
    best_depth   = best_s1$max_depth,
    best_eta     = best_s1$eta,
    best_alpha   = best_s2$alpha,
    best_lambda  = best_s2$lambda,
    best_nrounds = best_s2$best_iter,
    AUC          = m$AUC,
    Sensitivity  = m$Sensitivity,
    Specificity  = m$Specificity,
    PPV          = m$PPV,
    NPV          = m$NPV,
    stringsAsFactors = FALSE
  ))
}

# 3. save
dir.create("results", showWarnings = FALSE)
write.csv(results, "results/xgboost_results.csv", row.names = FALSE)
print(results[, c("feature_set", "n_features", "AUC", "Sensitivity",
                  "Specificity", "PPV", "NPV")])

print(Sys.time() - start_time)
