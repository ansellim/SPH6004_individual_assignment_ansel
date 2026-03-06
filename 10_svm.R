# SVM with radial kernel - CV tuning of cost and gamma
# DECLARATION: AI tools (Anthropic Claude Code) were used in the editing and development of this code.

start_time <- Sys.time()

library(e1071)
library(pROC)

set.seed(42)

target      <- "icu_death_flag"
N_FOLDS     <- 3
SVM_TRAIN_N <- 10000

df_train <- readRDS("data/train_small.rds")
df_test  <- readRDS("data/test.rds")

feature_sets <- list(
  all      = readRDS("data/candidate_features.rds"),
  stepwise = readRDS("data/selected_features_stepwise.rds"),
  lasso    = readRDS("data/selected_features_lasso.rds"),
  elastic  = readRDS("data/selected_features_elastic.rds"),
  boruta   = readRDS("data/selected_features_boruta.rds")
)

source("helpers.R")

svm_cost_grid  <- c(0.1, 1, 10)
svm_gamma_grid <- c(0.01, 0.1)

results <- data.frame()

for (fs_i in seq_along(feature_sets)) {
  fs_name  <- names(feature_sets)[fs_i]
  features <- feature_sets[[fs_name]]

  cols  <- c(features, target)
  df_tr <- df_train[, cols]
  df_te <- df_test[,  cols]

  set.seed(42)
  sub_idx  <- sample(nrow(df_tr), SVM_TRAIN_N)
  df_svm   <- df_tr[sub_idx, ]
  fold_idx <- sample(rep(1:N_FOLDS, length.out = nrow(df_svm)))

  tune_res <- data.frame()

  for (c_val in svm_cost_grid) {
    for (g_val in svm_gamma_grid) {
      fold_aucs <- numeric(N_FOLDS)
      for (f in 1:N_FOLDS) {
        # use decision values instead of probabilities for CV - faster
        fit      <- svm(as.formula(paste(target, "~ .")),
                        data        = df_svm[fold_idx != f, ],
                        kernel      = "radial", cost = c_val, gamma = g_val,
                        probability = FALSE, scale = FALSE)
        pred_val     <- predict(fit, df_svm[fold_idx == f, ], decision.values = TRUE)
        dv           <- attr(pred_val, "decision.values")[, 1]
        actual_f     <- as.integer(df_svm[fold_idx == f, target] == "Discharged")
        fold_aucs[f] <- as.numeric(auc(roc(actual_f, dv, quiet = TRUE)))
      }
      tune_res <- rbind(tune_res,
        data.frame(cost = c_val, gamma = g_val, CV_AUC = mean(fold_aucs)))
    }
  }

  best <- tune_res[which.max(tune_res$CV_AUC), ]

  final_fit <- svm(as.formula(paste(target, "~ .")),
                   data        = df_svm,
                   kernel      = "radial", cost = best$cost, gamma = best$gamma,
                   probability = TRUE, scale = FALSE)

  pred_te <- predict(final_fit, df_te, probability = TRUE)
  prob    <- attr(pred_te, "probabilities")[, "Discharged"]
  actual  <- as.integer(df_te[[target]] == "Discharged")
  m       <- compute_metrics(actual, prob)

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

dir.create("results", showWarnings = FALSE)
write.csv(results, "results/svm_results.csv", row.names = FALSE)
print(results[, c("feature_set", "n_features", "AUC", "Sensitivity",
                  "Specificity", "PPV", "NPV")])

print(Sys.time() - start_time)
