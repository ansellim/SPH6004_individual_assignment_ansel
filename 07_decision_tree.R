# CART decision tree with 5-fold CV tuning (cp and maxdepth)
# DECLARATION: AI tools (Anthropic Claude Code) were used in the editing and development of this code.

start_time <- Sys.time()

library(rpart)
library(pROC)

set.seed(42)

target  <- "icu_death_flag"
N_FOLDS <- 5

df_train <- readRDS("data/train.rds")
df_test  <- readRDS("data/test.rds")

feature_sets <- list(
  stepwise = readRDS("data/selected_features_stepwise.rds"),
  lasso    = readRDS("data/selected_features_lasso.rds"),
  elastic  = readRDS("data/selected_features_elastic.rds"),
  boruta   = readRDS("data/selected_features_boruta.rds"),
  all      = readRDS("data/candidate_features.rds")
)

source("helpers.R")

dt_cp_grid    <- c(0.0001, 0.001, 0.005, 0.01, 0.05)
dt_depth_grid <- c(5, 10, 20, 30)

results <- data.frame()

for (fs_i in seq_along(feature_sets)) {
  fs_name  <- names(feature_sets)[fs_i]
  features <- feature_sets[[fs_name]]

  cols  <- c(features, target)
  df_tr <- df_train[, cols]
  df_te <- df_test[,  cols]

  fold_idx <- sample(rep(1:N_FOLDS, length.out = nrow(df_tr)))
  tune_res <- data.frame()

  for (cp_val in dt_cp_grid) {
    for (md_val in dt_depth_grid) {
      fold_aucs <- numeric(N_FOLDS)
      for (f in 1:N_FOLDS) {
        fit <- rpart(as.formula(paste(target, "~ .")),
                     data    = df_tr[fold_idx != f, ],
                     method  = "class",
                     control = rpart.control(cp = cp_val, maxdepth = md_val))
        prob         <- predict(fit, df_tr[fold_idx == f, ], type = "prob")[, "Discharged"]
        actual_f     <- as.integer(df_tr[fold_idx == f, target] == "Discharged")
        fold_aucs[f] <- as.numeric(auc(roc(actual_f, prob, quiet = TRUE)))
      }
      tune_res <- rbind(tune_res,
        data.frame(cp = cp_val, maxdepth = md_val, CV_AUC = mean(fold_aucs)))
    }
  }

  best <- tune_res[which.max(tune_res$CV_AUC), ]

  final_fit <- rpart(as.formula(paste(target, "~ .")),
                     data    = df_tr,
                     method  = "class",
                     control = rpart.control(cp = best$cp, maxdepth = best$maxdepth))

  prob   <- predict(final_fit, df_te, type = "prob")[, "Discharged"]
  actual <- as.integer(df_te[[target]] == "Discharged")
  m      <- compute_metrics(actual, prob)

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

dir.create("results", showWarnings = FALSE)
write.csv(results, "results/decision_tree_results.csv", row.names = FALSE)
print(results[, c("feature_set", "n_features", "AUC", "Sensitivity",
                  "Specificity", "PPV", "NPV")])

print(Sys.time() - start_time)
