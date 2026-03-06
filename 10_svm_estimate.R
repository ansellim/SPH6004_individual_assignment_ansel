# 10_svm_estimate.R
# Times a single SVM fit and extrapolates total runtime for 10_svm.R

library(e1071)
library(pROC)

set.seed(42)

target      <- "icu_death_flag"
SVM_TRAIN_N <- 15000
N_FOLDS     <- 5

df_train <- readRDS("data/train.rds")

# use the smallest feature set (lasso) for a lower bound, boruta for upper
feature_sets <- list(
  lasso  = readRDS("data/selected_features_lasso.rds"),
  boruta = readRDS("data/selected_features_boruta.rds")
)

n_cost  <- 3   # cost grid: 0.1, 1, 10
n_gamma <- 3   # gamma grid: 0.01, 0.1, 1
n_fs    <- 4   # feature sets in full script

cat("=== SVM Runtime Estimator ===\n\n")

for (fs_name in names(feature_sets)) {
  features <- feature_sets[[fs_name]]
  cols  <- c(features, target)
  df_tr <- df_train[, cols]

  sub_idx  <- sample(nrow(df_tr), SVM_TRAIN_N)
  df_svm   <- df_tr[sub_idx, ]
  fold_idx <- sample(rep(1:N_FOLDS, length.out = nrow(df_svm)))

  # Time a single fold fit
  t0 <- Sys.time()
  fit <- svm(as.formula(paste(target, "~ .")),
             data        = df_svm[fold_idx != 1, ],
             kernel      = "radial", cost = 1, gamma = 0.1,
             probability = TRUE, scale = TRUE)
  t1 <- Sys.time()

  single_fit_sec <- as.numeric(difftime(t1, t0, units = "secs"))

  # Total fits: (n_cost * n_gamma * N_FOLDS) per feature set + 1 final fit per feature set
  fits_per_fs    <- n_cost * n_gamma * N_FOLDS + 1
  total_fits     <- fits_per_fs * n_fs
  est_total_sec  <- single_fit_sec * total_fits
  est_total_min  <- est_total_sec / 60

  cat(sprintf("Feature set: %s (%d features)\n", fs_name, length(features)))
  cat(sprintf("  Single fold fit:  %.1f sec\n", single_fit_sec))
  cat(sprintf("  Fits per feature set: %d (CV) + 1 (final) = %d\n",
              n_cost * n_gamma * N_FOLDS, fits_per_fs))
  cat(sprintf("  Total fits (4 feature sets): %d\n", total_fits))
  cat(sprintf("  Estimated total time: %.1f min (%.1f hrs)\n\n",
              est_total_min, est_total_min / 60))
}

cat("Note: This is a rough estimate. Actual time may vary by ~20%%\n")
cat("depending on feature set complexity and gamma/cost values.\n")
