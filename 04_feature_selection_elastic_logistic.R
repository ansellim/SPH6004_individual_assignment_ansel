# 04_feature_selection_elastic_logistic.R
# DECLARATION: AI tools (Anthropic Claude Code) were used in the editing and development of this code.
# Elastic net logistic regression for feature selection
# Alpha is tuned via CV over a grid of 0 to 1 in steps of 0.1 (alpha=0 is Ridge, alpha=1 is LASSO)
# Best alpha chosen by maximising CV AUC; lambda selected by lambda.min within that alpha

start_time <- Sys.time()

library(glmnet)   # elastic net
library(pROC)     # roc / auc
library(ggplot2)  # ROC curve plot

set.seed(42)

target <- "icu_death_flag"

# 1. load data
df_train <- readRDS("data/train.rds")
df_test  <- readRDS("data/test.rds")


# glmnet requires a numeric matrix - model.matrix handles one-hot encoding of factors
make_xy <- function(df, tgt) {
  X <- model.matrix(as.formula(paste("~", paste(setdiff(names(df), tgt), collapse = "+"))),
                    data = df)[, -1]   # drop intercept column
  y <- as.integer(df[[tgt]] == "Discharged")
  list(X = X, y = y)
}

train_xy <- make_xy(df_train, target)
test_xy  <- make_xy(df_test,  target)

# 3. alpha tuning - for each alpha value, cv.glmnet picks the best lambda and records its CV AUC
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

# 4. use the already-fitted cv object for the best alpha - no need to refit

cv_final <- cv_fits[[best_idx]]   # already fitted on df_train with best_alpha

best_lambda <- cv_final$lambda.min

elastic_coefs <- coef(cv_final, s = "lambda.min")
nonzero_idx   <- which(elastic_coefs[-1] != 0)
coef_names    <- rownames(elastic_coefs)[-1][nonzero_idx]
# collapse dummy names (e.g. "raceBlack") back to original column name
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
