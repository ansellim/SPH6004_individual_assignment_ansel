# 09_xgboost.R
# DECLARATION: AI tools (Anthropic Claude Code) were used in the editing and development of this code.
# XGBoost evaluated on 5 feature sets
# 2-stage CV tuning: (1) max_depth x eta, (2) alpha x lambda

start_time <- Sys.time()

library(xgboost)
library(pROC)

set.seed(42)

target <- "icu_death_flag"

df_train <- readRDS("data/train.rds")
df_test  <- readRDS("data/test.rds")

feature_sets <- list(
  candidate = readRDS("data/candidate_features.rds"),
  stepwise  = readRDS("data/selected_features_stepwise.rds"),
  lasso     = readRDS("data/selected_features_lasso.rds"),
  elastic   = readRDS("data/selected_features_elastic.rds"),
  boruta    = readRDS("data/selected_features_boruta.rds")
)

make_xgb_matrix <- function(df, tgt) {
  X <- model.matrix(as.formula(paste("~", paste(setdiff(names(df), tgt), collapse = "+"))),
                    data = df)[, -1]
  y <- as.integer(df[[tgt]] == "Discharged")
  xgb.DMatrix(data = X, label = y)
}

source("helpers.R")

xgb_depth_grid  <- c(3, 5, 7)
xgb_eta_grid    <- c(0.01, 0.05, 0.1)
xgb_alpha_grid  <- c(0, 0.1, 1)
xgb_lambda_grid <- c(0.1, 1, 10)
xgb_nrounds     <- 500

results <- data.frame()

for (fs_i in seq_along(feature_sets)) {
  fs_name  <- names(feature_sets)[fs_i]
  features <- feature_sets[[fs_name]]

  cols   <- c(features, target)
  dtrain <- make_xgb_matrix(df_train[, cols], target)
  dtest  <- make_xgb_matrix(df_test[,  cols], target)

  # stage 1: tune max_depth x eta
  s1_res <- data.frame()
  for (d_val in xgb_depth_grid) {
    for (e_val in xgb_eta_grid) {
      cv <- xgb.cv(
        params = list(objective="binary:logistic", eval_metric="auc",
                      max_depth=d_val, eta=e_val, subsample=0.7,
                      min_child_weight=20, alpha=0, lambda=1),
        data = dtrain, nfold = 5, nrounds = xgb_nrounds,
        early_stopping_rounds = 20, verbose = 0)
      best_it <- cv$early_stop$best_iteration
      cv_auc  <- cv$evaluation_log$test_auc_mean[best_it]
      s1_res  <- rbind(s1_res, data.frame(max_depth=d_val, eta=e_val,
                                          best_iter=best_it, CV_AUC=cv_auc))
    }
  }
  best_s1 <- s1_res[which.max(s1_res$CV_AUC), ]

  # stage 2: tune alpha x lambda
  s2_res <- data.frame()
  for (a_val in xgb_alpha_grid) {
    for (l_val in xgb_lambda_grid) {
      cv <- xgb.cv(
        params = list(objective="binary:logistic", eval_metric="auc",
                      max_depth=best_s1$max_depth, eta=best_s1$eta,
                      subsample=0.7, min_child_weight=20,
                      alpha=a_val, lambda=l_val),
        data = dtrain, nfold = 5, nrounds = xgb_nrounds,
        early_stopping_rounds = 20, verbose = 0)
      best_it <- cv$early_stop$best_iteration
      cv_auc  <- cv$evaluation_log$test_auc_mean[best_it]
      s2_res  <- rbind(s2_res, data.frame(alpha=a_val, lambda=l_val,
                                          best_iter=best_it, CV_AUC=cv_auc))
    }
  }
  best_s2 <- s2_res[which.max(s2_res$CV_AUC), ]

  final_fit <- xgb.train(
    params  = list(objective="binary:logistic", eval_metric="auc",
                   max_depth=best_s1$max_depth, eta=best_s1$eta,
                   subsample=0.7, min_child_weight=20,
                   alpha=best_s2$alpha, lambda=best_s2$lambda),
    data    = dtrain,
    nrounds = best_s2$best_iter,
    verbose = 0)

  prob   <- predict(final_fit, dtest)
  actual <- as.integer(getinfo(dtest, "label"))
  m      <- compute_metrics(actual, prob)

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

dir.create("results", showWarnings = FALSE)
write.csv(results, "results/xgboost_results.csv", row.names = FALSE)
print(results[, c("feature_set", "n_features", "AUC", "Sensitivity",
                  "Specificity", "PPV", "NPV")])

print(Sys.time() - start_time)
