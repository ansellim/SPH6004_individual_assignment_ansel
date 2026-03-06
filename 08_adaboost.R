# =============================================================================
# 08_adaboost.R
# AdaBoost evaluated on 5 feature sets
#
# workflow (candidate/stepwise/lasso/elastic/boruta):
#   1. hyperparam tuning - 3-fold CV on subsample (mfinal x maxdepth)
#   2. final model on full train_small with best hyperparams
#   3. eval on df_test
#   4. save results
#
# Uses train_small.rds (smaller oversampling ratio) for speed
# =============================================================================

start_time <- Sys.time()

library(adabag)
library(rpart)
library(pROC)

set.seed(42)

target    <- "icu_death_flag"
N_FOLDS   <- 3
ADA_SUB_N <- 10000

# ------ 1. load data and feature sets ------
df_train <- readRDS("data/train_small.rds")
df_test  <- readRDS("data/test.rds")

feature_sets <- list(
  all      = readRDS("data/candidate_features.rds"),
  stepwise = readRDS("data/selected_features_stepwise.rds"),
  lasso    = readRDS("data/selected_features_lasso.rds"),
  elastic  = readRDS("data/selected_features_elastic.rds"),
  boruta   = readRDS("data/selected_features_boruta.rds")
)


# Helper fn
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

ada_mfinal_grid <- c(50, 100)
ada_depth_grid  <- c(1, 3)

results <- data.frame()

# progress tracking
n_fs     <- length(feature_sets)
n_combos <- length(ada_mfinal_grid) * length(ada_depth_grid)
total_cv_fits  <- n_fs * n_combos * N_FOLDS
total_fits     <- total_cv_fits + n_fs
fit_counter    <- 0

# ------------- Runtime estimate -------------
cat("=== AdaBoost Runtime Estimate ===\n")
est_fs   <- feature_sets[[1]]
est_cols <- c(est_fs, target)
set.seed(99)
est_idx  <- sample(nrow(df_train[, est_cols]), min(ADA_SUB_N, nrow(df_train)))
est_df   <- df_train[est_idx, est_cols]
est_fold <- sample(rep(1:N_FOLDS, length.out = nrow(est_df)))

t0 <- Sys.time()
boosting(as.formula(paste(target, "~ .")),
         data = est_df[est_fold != 1, ],
         boos = TRUE, mfinal = 50,
         control = rpart.control(maxdepth = 1))
cv_fit_sec <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

t0 <- Sys.time()
boosting(as.formula(paste(target, "~ .")),
         data = est_df,
         boos = TRUE, mfinal = 50,
         control = rpart.control(maxdepth = 1))
final_fit_sec <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

n_cv_total    <- total_cv_fits
n_final_total <- n_fs
est_min       <- (cv_fit_sec * n_cv_total + final_fit_sec * n_final_total) / 60

cat(sprintf("  CV fit: %.1f sec | final fit: %.1f sec\n", cv_fit_sec, final_fit_sec))
cat(sprintf("  Total fits: %d CV + %d final = %d\n", n_cv_total, n_final_total, n_cv_total + n_final_total))
cat(sprintf("  Estimated runtime: %.1f min (%.1f hrs)\n", est_min, est_min / 60))
cat("=== Starting full run ===\n\n")

rm(est_fs, est_cols, est_idx, est_df, est_fold, cv_fit_sec, final_fit_sec,
   n_cv_total, n_final_total, est_min)

loop_start <- Sys.time()

# ---- 2. Loop over feature sets
for (fs_i in seq_along(feature_sets)) {
  fs_name  <- names(feature_sets)[fs_i]
  features <- feature_sets[[fs_name]]

  cat(sprintf("\n=== Feature set %d/%d: %s (%d features) ===\n",
              fs_i, n_fs, fs_name, length(features)))

  cols  <- c(features, target)
  df_tr <- df_train[, cols]
  df_te <- df_test[,  cols]

  # subsample once per feature set, fixed folds across hyperparam combos
  set.seed(42)
  sub_idx  <- sample(nrow(df_tr), min(ADA_SUB_N, nrow(df_tr)))
  df_sub   <- df_tr[sub_idx, ]
  fold_idx <- sample(rep(1:N_FOLDS, length.out = nrow(df_sub)))

  tune_res <- data.frame()
  combo_i  <- 0

  for (mf_val in ada_mfinal_grid) {
    for (md_val in ada_depth_grid) {
      combo_i <- combo_i + 1
      fold_aucs <- numeric(N_FOLDS)
      for (f in 1:N_FOLDS) {
        fit      <- boosting(as.formula(paste(target, "~ .")),
                             data    = df_sub[fold_idx != f, ],
                             boos    = TRUE, mfinal = mf_val,
                             control = rpart.control(maxdepth = md_val))
        prob         <- { p <- predict(fit, df_sub[fold_idx == f, ]); if (!is.null(colnames(p$prob))) p$prob[, "Discharged"] else p$prob[, 1] }
        actual_f     <- as.integer(df_sub[fold_idx == f, target] == "Discharged")
        fold_aucs[f] <- as.numeric(auc(roc(actual_f, prob, quiet = TRUE)))
        fit_counter  <- fit_counter + 1
      }
      elapsed <- as.numeric(difftime(Sys.time(), loop_start, units = "mins"))
      eta     <- elapsed / fit_counter * (total_fits - fit_counter)
      cat(sprintf("  [%s] combo %d/%d (mfinal=%d, maxdepth=%d) CV_AUC=%.4f | fit %d/%d  elapsed %.1f min  ETA %.1f min\n",
                  fs_name, combo_i, n_combos, mf_val, md_val, mean(fold_aucs),
                  fit_counter, total_fits, elapsed, eta))

      tune_res <- rbind(tune_res,
        data.frame(mfinal = mf_val, maxdepth = md_val, CV_AUC = mean(fold_aucs)))
    }
  }
  best <- tune_res[which.max(tune_res$CV_AUC), ]
  cat(sprintf("  >> best: mfinal=%d, maxdepth=%d, CV_AUC=%.4f\n",
              best$mfinal, best$maxdepth, best$CV_AUC))

  # final model on full train_small
  cat(sprintf("  fitting final model on %d rows...\n", nrow(df_tr)))
  final_fit <- boosting(as.formula(paste(target, "~ .")),
                        data    = df_tr,
                        boos    = TRUE, mfinal = best$mfinal,
                        control = rpart.control(maxdepth = best$maxdepth))
  fit_counter <- fit_counter + 1

  prob   <- { p <- predict(final_fit, df_te); if (!is.null(colnames(p$prob))) p$prob[, "Discharged"] else p$prob[, 1] }
  actual <- as.integer(df_te[[target]] == "Discharged")
  m      <- compute_metrics(actual, prob)
  cat(sprintf("  >> test AUC=%.4f  Sens=%.4f  Spec=%.4f\n", m$AUC, m$Sensitivity, m$Specificity))


  results <- rbind(results, data.frame(
    feature_set  = fs_name,
    n_features   = length(features),
    features     = paste(features, collapse = ";"),
    best_mfinal  = best$mfinal,
    best_depth   = best$maxdepth,
    AUC          = m$AUC,
    Sensitivity  = m$Sensitivity,
    Specificity  = m$Specificity,
    PPV          = m$PPV,
    NPV          = m$NPV,
    stringsAsFactors = FALSE
  ))
}

# ---------- 3. Save ----------
dir.create("results", showWarnings = FALSE)
write.csv(results, "results/adaboost_results.csv", row.names = FALSE)
print(results[, c("feature_set", "n_features", "AUC", "Sensitivity",
                  "Specificity", "PPV", "NPV")])

print(Sys.time() - start_time)
