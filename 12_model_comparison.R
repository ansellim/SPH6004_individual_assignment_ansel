# 12_model_comparison.R - combine results from all models and generate ROC plot
# DECLARATION: AI tools (Anthropic Claude Code) were used in the editing and development of this code.

library(dplyr)

metric_cols <- c("AUC", "Sensitivity", "Specificity", "PPV", "NPV")
keep_cols   <- c("model", "feature_set", "n_features", metric_cols)

read_results <- function(path, model_name) {
  df <- read.csv(path, stringsAsFactors = FALSE)
  df$model <- model_name
  df[, keep_cols]
}

all_results <- bind_rows(
  read_results("results/logistic_regression_results.csv", "Logistic Regression"),
  read_results("results/decision_tree_results.csv",       "Decision Tree"),
  read_results("results/adaboost_results.csv",            "AdaBoost"),
  read_results("results/xgboost_results.csv",             "XGBoost"),
  read_results("results/svm_results.csv",                 "SVM"),
  read_results("results/random_forest_results.csv",       "Random Forest")
)

# average across feature sets
avg_results <- all_results %>%
  group_by(model) %>%
  summarise(
    feature_set = "AVERAGE",
    n_features  = round(mean(n_features), 1),
    AUC         = round(mean(AUC),         4),
    Sensitivity = round(mean(Sensitivity), 4),
    Specificity = round(mean(Specificity), 4),
    PPV         = round(mean(PPV),         4),
    NPV         = round(mean(NPV),         4),
    .groups = "drop"
  )

model_order <- c("Logistic Regression", "Decision Tree", "AdaBoost", "XGBoost", "SVM", "Random Forest")
fs_order    <- c("stepwise", "lasso", "elastic", "boruta", "AVERAGE")

final_table <- bind_rows(all_results, avg_results) %>%
  mutate(model       = factor(model,       levels = model_order),
         feature_set = factor(feature_set, levels = fs_order)) %>%
  arrange(model, feature_set) %>%
  mutate(model = as.character(model), feature_set = as.character(feature_set))

print(as.data.frame(final_table[, keep_cols]), row.names = FALSE)

dir.create("results", showWarnings = FALSE)
write.csv(final_table[, keep_cols], "results/model_comparison.csv", row.names = FALSE)

# ROC curves using lasso feature set
library(pROC)
library(rpart)

df_train <- readRDS("data/train.rds")
df_test  <- readRDS("data/test.rds")
lasso_features <- readRDS("data/selected_features_lasso.rds")

target <- "icu_death_flag"
cols   <- c(lasso_features, target)
actual <- as.integer(df_test[[target]] == "Discharged")

lr_fit  <- glm(as.formula(paste(target, "~ .")),
               data = df_train[, cols], family = binomial)
lr_prob <- 1 - predict(lr_fit, newdata = df_test[, cols], type = "response")
roc_lr  <- roc(actual, lr_prob, quiet = TRUE)

dt_res   <- read.csv("results/decision_tree_results.csv", stringsAsFactors = FALSE)
dt_lasso <- dt_res[dt_res$feature_set == "lasso", ]
dt_fit   <- rpart(as.formula(paste(target, "~ .")),
                  data = df_train[, cols], method = "class",
                  control = rpart.control(cp = dt_lasso$best_cp,
                                          maxdepth = dt_lasso$best_depth))
dt_prob <- predict(dt_fit, df_test[, cols], type = "prob")[, "Discharged"]
roc_dt  <- roc(actual, dt_prob, quiet = TRUE)

library(e1071)
svm_res   <- read.csv("results/svm_results.csv", stringsAsFactors = FALSE)
svm_lasso <- svm_res[svm_res$feature_set == "lasso", ]
df_train_small <- readRDS("data/train_small.rds")
set.seed(42)
svm_sub <- df_train_small[sample(nrow(df_train_small), 10000), cols]
svm_fit <- svm(as.formula(paste(target, "~ .")),
               data = svm_sub, kernel = "radial",
               cost = svm_lasso$best_cost, gamma = svm_lasso$best_gamma,
               probability = TRUE, scale = FALSE)
svm_pred <- predict(svm_fit, df_test[, cols], probability = TRUE)
svm_prob <- attr(svm_pred, "probabilities")[, "Discharged"]
roc_svm  <- roc(actual, svm_prob, quiet = TRUE)

library(adabag)
ada_res   <- read.csv("results/adaboost_results.csv", stringsAsFactors = FALSE)
ada_lasso <- ada_res[ada_res$feature_set == "lasso", ]
ada_fit   <- boosting(as.formula(paste(target, "~ .")),
                      data = df_train_small[, cols],
                      boos = TRUE, mfinal = ada_lasso$best_mfinal,
                      control = rpart.control(maxdepth = ada_lasso$best_depth))
ada_pred <- predict(ada_fit, df_test[, cols])
ada_prob <- if (!is.null(colnames(ada_pred$prob))) ada_pred$prob[, "Discharged"] else ada_pred$prob[, 1]
roc_ada  <- roc(actual, ada_prob, quiet = TRUE)

library(xgboost)
make_xgb_matrix <- function(df, tgt) {
  X <- model.matrix(as.formula(paste("~", paste(setdiff(names(df), tgt), collapse = "+"))),
                    data = df)[, -1]
  y <- as.integer(df[[tgt]] == "Discharged")
  xgb.DMatrix(data = X, label = y)
}
dtrain_xgb <- make_xgb_matrix(df_train_small[, cols], target)
dtest_xgb  <- make_xgb_matrix(df_test[, cols], target)
xgb_fit <- xgb.train(
  params  = list(objective = "binary:logistic", eval_metric = "auc",
                 max_depth = 6, eta = 0.3, subsample = 0.7, nthread = 4),
  data    = dtrain_xgb, nrounds = 100, verbose = 0)
xgb_prob <- predict(xgb_fit, dtest_xgb)
roc_xgb  <- roc(actual, xgb_prob, quiet = TRUE)

library(randomForest)
rf_res   <- read.csv("results/random_forest_results.csv", stringsAsFactors = FALSE)
rf_lasso <- rf_res[rf_res$feature_set == "lasso", ]
rf_fit   <- randomForest(as.formula(paste(target, "~ .")),
                         data     = df_train_small[, cols],
                         ntree    = rf_lasso$best_ntree,
                         mtry     = rf_lasso$best_mtry,
                         nodesize = rf_lasso$best_nodesize)
rf_prob  <- predict(rf_fit, df_test[, cols], type = "prob")[, "Discharged"]
roc_rf   <- roc(actual, rf_prob, quiet = TRUE)

model_colours <- c("steelblue", "tomato", "forestgreen", "darkorange", "purple", "deeppink")
dir.create("figures", showWarnings = FALSE)
png("figures/roc_comparison_lasso.png", width = 6, height = 5, units = "in", res = 300)
plot(roc_lr,  col = model_colours[1], lwd = 2, main = "")
plot(roc_dt,  col = model_colours[2], lwd = 2, add = TRUE)
plot(roc_svm, col = model_colours[3], lwd = 2, add = TRUE)
plot(roc_ada, col = model_colours[4], lwd = 2, add = TRUE)
plot(roc_xgb, col = model_colours[5], lwd = 2, add = TRUE)
plot(roc_rf,  col = model_colours[6], lwd = 2, add = TRUE)
legend("bottomright",
       legend = c(sprintf("Logistic Regression (AUC = %.3f)", auc(roc_lr)),
                  sprintf("Decision Tree (AUC = %.3f)", auc(roc_dt)),
                  sprintf("SVM (AUC = %.3f)", auc(roc_svm)),
                  sprintf("AdaBoost (AUC = %.3f)", auc(roc_ada)),
                  sprintf("XGBoost (AUC = %.3f)", auc(roc_xgb)),
                  sprintf("Random Forest (AUC = %.3f)", auc(roc_rf))),
       col = model_colours, lwd = 2, cex = 0.8)
dev.off()
