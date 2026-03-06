# 11_random_forest.R - tune ntree, mtry, nodesize
# DECLARATION: AI tools (Anthropic Claude Code) were used in the editing and development of this code.

start_time <- Sys.time()

library(randomForest)
library(pROC)

set.seed(42)

target   <- "icu_death_flag"
N_FOLDS  <- 3
RF_SUB_N <- 10000

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

ntree_grid    <- c(100, 300)
nodesize_grid <- c(1, 5, 10)
# mtry will be derived from sqrt(p) inside the loop

results <- data.frame()

for (fs_i in seq_along(feature_sets)) {
  fs_name  <- names(feature_sets)[fs_i]
  features <- feature_sets[[fs_name]]

  cols  <- c(features, target)
  df_tr <- df_train[, cols]
  df_te <- df_test[,  cols]

  p         <- length(features)
  mtry_grid <- unique(pmax(1L, c(floor(sqrt(p) / 2), floor(sqrt(p)), floor(2 * sqrt(p)))))

  set.seed(42)
  sub_idx  <- sample(nrow(df_tr), min(RF_SUB_N, nrow(df_tr)))
  df_sub   <- df_tr[sub_idx, ]
  fold_idx <- sample(rep(1:N_FOLDS, length.out = nrow(df_sub)))

  tune_res <- data.frame()

  for (nt_val in ntree_grid) {
    for (mt_val in mtry_grid) {
      for (ns_val in nodesize_grid) {
        fold_aucs <- numeric(N_FOLDS)
        for (f in 1:N_FOLDS) {
          fit          <- randomForest(as.formula(paste(target, "~ .")),
                                       data = df_sub[fold_idx != f, ],
                                       ntree = nt_val, mtry = mt_val, nodesize = ns_val)
          prob         <- predict(fit, df_sub[fold_idx == f, ], type = "prob")[, "Discharged"]
          actual_f     <- as.integer(df_sub[fold_idx == f, target] == "Discharged")
          fold_aucs[f] <- as.numeric(auc(roc(actual_f, prob, quiet = TRUE)))
        }
        tune_res <- rbind(tune_res,
          data.frame(ntree = nt_val, mtry = mt_val, nodesize = ns_val,
                     CV_AUC = mean(fold_aucs)))
      }
    }
  }

  best <- tune_res[which.max(tune_res$CV_AUC), ]

  final_fit <- randomForest(as.formula(paste(target, "~ .")),
                            data = df_tr,
                            ntree = best$ntree, mtry = best$mtry, nodesize = best$nodesize)

  prob   <- predict(final_fit, df_te, type = "prob")[, "Discharged"]
  actual <- as.integer(df_te[[target]] == "Discharged")
  m      <- compute_metrics(actual, prob)

  results <- rbind(results, data.frame(
    feature_set   = fs_name,
    n_features    = length(features),
    features      = paste(features, collapse = ";"),
    best_ntree    = best$ntree,
    best_mtry     = best$mtry,
    best_nodesize = best$nodesize,
    AUC           = m$AUC,
    Sensitivity   = m$Sensitivity,
    Specificity   = m$Specificity,
    PPV           = m$PPV,
    NPV           = m$NPV,
    stringsAsFactors = FALSE
  ))
}

dir.create("results", showWarnings = FALSE)
write.csv(results, "results/random_forest_results.csv", row.names = FALSE)
print(results[, c("feature_set", "n_features", "AUC", "Sensitivity",
                  "Specificity", "PPV", "NPV")])

print(Sys.time() - start_time)
