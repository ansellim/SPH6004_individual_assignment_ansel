# ===========================================================================
# 06_logistic_regression.R
# plain logistic regression evaluated on each of the 4 feature sets
#
# workflow (repeated for stepwise / lasso / elastic / boruta):
#   1. fit glm(binomial) on df_train restricted to selected features
#   2. evaluate on df_test: AUC, Sens, Spec, PPV, NPV
#   3. save combined results to results/logistic_regression_results.csv
# ===========================================================================

start_time <- Sys.time()

library(pROC)

set.seed(42)

target <- "icu_death_flag"

# ---------- 1. Load data and feature sets ----------
df_train <- readRDS("data/train.rds")
df_test  <- readRDS("data/test.rds")

feature_sets <- list(
  stepwise = readRDS("data/selected_features_stepwise.rds"),
  lasso    = readRDS("data/selected_features_lasso.rds"),
  elastic  = readRDS("data/selected_features_elastic.rds"),
  boruta   = readRDS("data/selected_features_boruta.rds"),
  all      = readRDS("data/candidate_features.rds")
)


# Helper: compute metrics from confusion matrix
compute_metrics <- function(actual, prob, threshold = 0.5) {
  roc_obj <- roc(actual, prob, quiet = TRUE)
  auc_val <- as.numeric(auc(roc_obj))

  pred   <- ifelse(prob >= threshold, 1L, 0L)
  tp <- sum(pred == 1 & actual == 1)
  tn <- sum(pred == 0 & actual == 0)
  fp <- sum(pred == 1 & actual == 0)
  fn <- sum(pred == 0 & actual == 1)

  list(
    AUC         = round(auc_val,          4),
    Sensitivity = round(tp / (tp + fn),   4),
    Specificity = round(tn / (tn + fp),   4),
    PPV         = round(tp / (tp + fp),   4),
    NPV         = round(tn / (tn + fn),   4)
  )
}

# ---- 2. Fit and evaluate for each feature set ----
results <- data.frame()

for (fs_name in names(feature_sets)) {
  features <- feature_sets[[fs_name]]

  cols   <- c(features, target)
  df_tr  <- df_train[, cols]
  df_te  <- df_test[,  cols]

  # fit plain logistic regression (no regularization)
  fit <- glm(as.formula(paste(target, "~ .")),
             data   = df_tr,
             family = binomial)

  # glm models P(DiedinICU) - 2nd factor level; invert for P(Discharged)
  prob   <- 1 - predict(fit, newdata = df_te, type = "response")
  actual <- as.integer(df_te[[target]] == "Discharged")

  m <- compute_metrics(actual, prob)

  results <- rbind(results, data.frame(
    feature_set = fs_name,
    n_features  = length(features),
    features    = paste(features, collapse = ";"),
    AUC         = m$AUC,
    Sensitivity = m$Sensitivity,
    Specificity = m$Specificity,
    PPV         = m$PPV,
    NPV         = m$NPV,
    stringsAsFactors = FALSE
  ))
}

# ------ 3. Save ------
dir.create("results", showWarnings = FALSE)
write.csv(results, "results/logistic_regression_results.csv", row.names = FALSE)
print(results[, c("feature_set", "n_features", "AUC", "Sensitivity",
                  "Specificity", "PPV", "NPV")])

print(Sys.time() - start_time)
