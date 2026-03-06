# =============================================================================
# 05_feature_selection_elastic_logistic.R
# Elastic net logistic regression, alpha tuned via CV
#
# workflow:
#   1. load 70/30 split
#   2. for each alpha in {0.0, 0.1, ..., 1.0}:
#        - cv.glmnet on df_train (5-fold CV, AUC) -> select lambda, record CV AUC
#   3. pick best_alpha = argmax(CV AUC)
#   4. refit final model on df_train with best_alpha, eval on df_test
#   5. save model, selected features, results
# =============================================================================

start_time <- Sys.time()

library(glmnet)   # elastic net
library(pROC)     # roc / auc
library(ggplot2)  # ROC curve plot

set.seed(42)

target <- "icu_death_flag"

# 1. load data
df_train <- readRDS("data/train.rds")
df_test  <- readRDS("data/test.rds")


# -----------------------------------------------------------------------------
# 2. Prepare model matrices
#    glmnet needs numeric matrix
#model.matrix does one-hot encoding
# -----------------------------------------------------------------------------
make_xy <- function(df, tgt) {
  X <- model.matrix(as.formula(paste("~", paste(setdiff(names(df), tgt), collapse = "+"))),
                    data = df)[, -1]   # drop intercept column
  y <- as.integer(df[[tgt]] == "Discharged")
  list(X = X, y = y)
}

train_xy <- make_xy(df_train, target)
test_xy  <- make_xy(df_test,  target)

# 3. alpha tuning via 5-fold CV on df_train
#    for each alpha, cv.glmnet picks best lambda and records CV AUC
#    alpha=0 -> Ridge; alpha=1 -> Lasso; in between -> elastic net
alpha_grid <- seq(0, 1, by = 0.1)
cv_auc     <- numeric(length(alpha_grid))
cv_fits    <- vector("list", length(alpha_grid))


for (i in seq_along(alpha_grid)) {
  cv_fits[[i]] <- cv.glmnet(
    train_xy$X, train_xy$y,
    family       = "binomial",
    alpha        = alpha_grid[i],
    type.measure = "auc",
    nfolds       = 5
  )
  cv_auc[i] <- max(cv_fits[[i]]$cvm)   # best CV AUC for this alpha
}

best_idx   <- which.max(cv_auc)
best_alpha <- alpha_grid[best_idx]


alpha_summary <- data.frame(alpha = alpha_grid, CV_AUC = round(cv_auc, 4))
print(alpha_summary, row.names = FALSE)

# ---- 4. final model: refit on full df_train with best_alpha

cv_final <- cv_fits[[best_idx]]   # already fitted on df_train with best_alpha

best_lambda <- cv_final$lambda.min

# selected features (nonzero coefficients, excl intercept)
elastic_coefs <- coef(cv_final, s = "lambda.min")
nonzero_idx   <- which(elastic_coefs[-1] != 0)   # drop intercept row
coef_names    <- rownames(elastic_coefs)[-1][nonzero_idx]

# map dummy names back to orignal column names
selected_features_elastic <- unique(gsub("(race|gender).*", "\\1", coef_names))

print(selected_features_elastic)

# 5. evaluate on test set
log_loss <- function(actual, prob, eps = 1e-15) {
  prob <- pmax(pmin(prob, 1 - eps), eps)
  -mean(actual * log(prob) + (1 - actual) * log(1 - prob))
}

test_prob   <- predict(cv_final, newx = test_xy$X,
                       s = "lambda.min", type = "response")[, 1]
test_actual <- test_xy$y

ll      <- log_loss(test_actual, test_prob)
roc_obj <- roc(test_actual, test_prob, quiet = TRUE)
auc_val <- as.numeric(auc(roc_obj))

pred_class   <- factor(ifelse(test_prob >= 0.5, "Discharged", "DiedinICU"),
                        levels = c("Discharged", "DiedinICU"))
actual_class <- factor(ifelse(test_actual == 1, "Discharged", "DiedinICU"),
                        levels = c("Discharged", "DiedinICU"))

cm <- table(Predicted = pred_class, Actual = actual_class)

tp <- cm["Discharged", "Discharged"]
tn <- cm["DiedinICU",  "DiedinICU"]
fp <- cm["Discharged", "DiedinICU"]
fn <- cm["DiedinICU",  "Discharged"]

accuracy    <- (tp + tn) / sum(cm)
sensitivity <- tp / (tp + fn)
specificity <- tn / (tn + fp)
ppv         <- tp / (tp + fp)

print(cm)

# ------ 6. save
dir.create("data", showWarnings = FALSE)

saveRDS(cv_final,                   "data/elastic_model.rds")
saveRDS(selected_features_elastic,  "data/selected_features_elastic.rds")

elastic_results <- data.frame(
  Model       = paste0("Elastic Net (alpha=", best_alpha, ")"),
  Log_Loss    = round(ll,          4),
  AUC_ROC     = round(auc_val,     4),
  Accuracy    = round(accuracy,    4),
  Sensitivity = round(sensitivity, 4),
  Specificity = round(specificity, 4),
  PPV         = round(ppv,         4),
  stringsAsFactors = FALSE
)
saveRDS(elastic_results, "data/elastic_results.rds")

# 7. ROC curve
dir.create("figures", showWarnings = FALSE)

roc_df <- data.frame(
  FPR = 1 - roc_obj$specificities,
  TPR = roc_obj$sensitivities
)

p_roc <- ggplot(roc_df, aes(x = FPR, y = TPR)) +
  geom_line(colour = "darkorange", linewidth = 0.9) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey50") +
  annotate("text", x = 0.65, y = 0.15,
           label = paste0("AUC = ", round(auc_val, 4)),
           size = 4.5, colour = "darkorange") +
  labs(title = paste0("ROC Curve — Elastic Net Logistic Regression (alpha=", best_alpha, ")"),
       x = "1 - Specificity (FPR)", y = "Sensitivity (TPR)") +
  theme_bw()

ggsave("figures/roc_elastic_logistic.png", p_roc,
       width = 6, height = 5, dpi = 150)

print(Sys.time() - start_time)
