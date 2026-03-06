# AdaBoost - tune mfinal and maxdepth via 3-fold CV on a subsample
# DECLARATION: AI tools (Anthropic Claude Code) were used in the editing and development of this code.

start_time <- Sys.time()

library(adabag)
library(rpart)
library(pROC)

set.seed(42)

target    <- "icu_death_flag"
N_FOLDS   <- 3
ADA_SUB_N <- 10000

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

ada_mfinal_grid <- c(50, 100)
ada_depth_grid  <- c(1, 3)

results <- data.frame()

for (fs_i in seq_along(feature_sets)) {
  fs_name  <- names(feature_sets)[fs_i]
  features <- feature_sets[[fs_name]]

  cols  <- c(features, target)
  df_tr <- df_train[, cols]
  df_te <- df_test[,  cols]

  # adaboost is slow so subsample for CV
  set.seed(42)
  sub_idx  <- sample(nrow(df_tr), min(ADA_SUB_N, nrow(df_tr)))
  df_sub   <- df_tr[sub_idx, ]
  fold_idx <- sample(rep(1:N_FOLDS, length.out = nrow(df_sub)))

  tune_res <- data.frame()

  for (mf_val in ada_mfinal_grid) {
    for (md_val in ada_depth_grid) {
      fold_aucs <- numeric(N_FOLDS)
      for (f in 1:N_FOLDS) {
        fit      <- boosting(as.formula(paste(target, "~ .")),
                             data    = df_sub[fold_idx != f, ],
                             boos    = TRUE, mfinal = mf_val,
                             control = rpart.control(maxdepth = md_val))
        prob         <- { p <- predict(fit, df_sub[fold_idx == f, ]); if (!is.null(colnames(p$prob))) p$prob[, "Discharged"] else p$prob[, 1] }
        actual_f     <- as.integer(df_sub[fold_idx == f, target] == "Discharged")
        fold_aucs[f] <- as.numeric(auc(roc(actual_f, prob, quiet = TRUE)))
      }
      tune_res <- rbind(tune_res,
        data.frame(mfinal = mf_val, maxdepth = md_val, CV_AUC = mean(fold_aucs)))
    }
  }

  best <- tune_res[which.max(tune_res$CV_AUC), ]

  final_fit <- boosting(as.formula(paste(target, "~ .")),
                        data    = df_tr,
                        boos    = TRUE, mfinal = best$mfinal,
                        control = rpart.control(maxdepth = best$maxdepth))

  prob   <- { p <- predict(final_fit, df_te); if (!is.null(colnames(p$prob))) p$prob[, "Discharged"] else p$prob[, 1] }
  actual <- as.integer(df_te[[target]] == "Discharged")
  m      <- compute_metrics(actual, prob)

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

dir.create("results", showWarnings = FALSE)
write.csv(results, "results/adaboost_results.csv", row.names = FALSE)
print(results[, c("feature_set", "n_features", "AUC", "Sensitivity",
                  "Specificity", "PPV", "NPV")])

print(Sys.time() - start_time)
