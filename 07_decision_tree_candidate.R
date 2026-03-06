# 07_decision_tree_candidate.R
# Run decision tree on candidate features only, append to existing results

start_time <- Sys.time()

library(rpart)
library(pROC)

set.seed(42)

target  <- "icu_death_flag"
N_FOLDS <- 5

df_train <- readRDS("data/train.rds")
df_test  <- readRDS("data/test.rds")

features <- readRDS("data/candidate_features.rds")

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

dt_cp_grid    <- c(0.0001, 0.001, 0.005, 0.01, 0.05)
dt_depth_grid <- c(5, 10, 20, 30)

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

new_row <- data.frame(
  feature_set = "candidate",
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
)

# append to existing results
dir.create("results", showWarnings = FALSE)
existing <- read.csv("results/decision_tree_results.csv", stringsAsFactors = FALSE)
combined <- rbind(new_row, existing)  # candidate first
write.csv(combined, "results/decision_tree_results.csv", row.names = FALSE)

print(combined[, c("feature_set", "n_features", "AUC", "Sensitivity",
                    "Specificity", "PPV", "NPV")])

print(Sys.time() - start_time)
