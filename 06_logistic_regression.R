# logistic regression baseline - no regularization
# DECLARATION: AI tools (Anthropic Claude Code) were used in the editing and development of this code.

start_time <- Sys.time()

library(pROC)

set.seed(42)

target <- "icu_death_flag"

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

results <- data.frame()

for (fs_name in names(feature_sets)) {
  features <- feature_sets[[fs_name]]

  cols   <- c(features, target)
  df_tr  <- df_train[, cols]
  df_te  <- df_test[,  cols]

  fit <- glm(as.formula(paste(target, "~ .")),
             data   = df_tr,
             family = binomial)

  # glm models P(DiedinICU); invert to get P(Discharged)
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

dir.create("results", showWarnings = FALSE)
write.csv(results, "results/logistic_regression_results.csv", row.names = FALSE)
print(results[, c("feature_set", "n_features", "AUC", "Sensitivity",
                  "Specificity", "PPV", "NPV")])

print(Sys.time() - start_time)
