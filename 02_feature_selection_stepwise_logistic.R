# =============================================================================
# 03_stepwise.R
# Logistic regression with stepwise forward feature selection
# Evaluation on held-out test set
# Assumes that we have already partitioned the data into train and test sets, and that we have already performed 
# =============================================================================

start_time <- Sys.time()

library(MASS)     # stepAIC
library(pROC)     # AUC-ROC
library(ggplot2)  # ROC curve plot

# Set seed for reproducibility
set.seed(42)

# 1. load data
df_train <- readRDS("data/train.rds")
df_test  <- readRDS("data/test.rds")

target <- "icu_death_flag"

print(prop.table(table(df_train[[target]])))

# -----------------------------------------------------------------------------
# 2. Define null and full models for stepwise
#    null model: intercept only
#    full model: all 38 candidate features
#    direction: "forward" - start from null, add one feature at a time
#    criterion: AIC (penalizes model complexity via +2k per param)
#    note: scope uses fitted model objects not formulas, so "." resolves correctly
# -----------------------------------------------------------------------------
null_model <- glm(as.formula(paste(target, "~ 1")),
                  data = df_train, family = binomial)
full_model <- glm(as.formula(paste(target, "~ .")),
                  data = df_train, family = binomial)

# Variables in the full model, excluding the intercept
full_vars <- names(coef(full_model))[-1]


# Stepwise forward selection
step_model <- stepAIC(
  null_model,
  scope     = list(lower = null_model, upper = full_model),
  direction = "forward",
  trace     = TRUE    # set to FALSE to suppress per-step output
)

# ----- 3. selected features -----
selected_vars <- names(coef(step_model))[-1]   # drop "(Intercept)"

# for factor variables stepAIC returns eg "raceBlack" - need to extract base name
selected_features <- unique(gsub("(race|gender).*", "\\1", selected_vars))

print(selected_features)

print(summary(step_model))

print(setdiff(full_vars,selected_features))

# 4. helper: log loss
log_loss <- function(actual, prob, eps = 1e-15) {
  prob <- pmax(pmin(prob, 1 - eps), eps)
  -mean(actual * log(prob) + (1 - actual) * log(1 - prob))
}

# ---- 5. evaluate on test set
# predicted probs for positive class (Discharged)
# glm models P(second factor lvl) = P(DiedinICU); invert to get P(Discharged)
test_prob <- 1 - predict(step_model, newdata = df_test, type = "response")
test_actual <- as.integer(df_test[[target]] == "Discharged")

# Log loss
ll <- log_loss(test_actual, test_prob)

# AUC-ROC
roc_obj <- roc(test_actual, test_prob, quiet = TRUE)

# confusion matrix at 0.5 cutoff
test_pred_class <- ifelse(test_prob >= 0.5, "Discharged", "DiedinICU")
test_pred_class <- factor(test_pred_class, levels = levels(df_test[[target]]))

cm <- table(Predicted = test_pred_class, Actual = df_test[[target]])
print(cm)

tp <- cm["Discharged", "Discharged"]
tn <- cm["DiedinICU",  "DiedinICU"]
fp <- cm["Discharged", "DiedinICU"]
fn <- cm["DiedinICU",  "Discharged"]

accuracy    <- (tp + tn) / sum(cm)
sensitivity <- tp / (tp + fn)
specificity <- tn / (tn + fp)
ppv         <- tp / (tp + fp)


# 6. save model and selected features
dir.create("data", showWarnings = FALSE)
saveRDS(step_model,       "data/logistic_model.rds")

# save selected features from stepwise
saveRDS(selected_features, "data/selected_features_stepwise.rds")


# Save evaluation results for model comparison (11_model_comparison.R)
logistic_results <- data.frame(
  Model       = "Logistic Regression (Stepwise)",
  Log_Loss    = round(ll, 4),
  AUC_ROC     = round(auc(roc_obj), 4),
  Accuracy    = round(accuracy, 4),
  Sensitivity = round(sensitivity, 4),
  Specificity = round(specificity, 4),
  PPV         = round(ppv, 4),
  stringsAsFactors = FALSE
)
saveRDS(logistic_results, "data/logistic_results.rds")

# ------- 7. ROC curve
dir.create("figures", showWarnings = FALSE)

roc_df <- data.frame(
  FPR = 1 - roc_obj$specificities,
  TPR = roc_obj$sensitivities
)

p_roc <- ggplot(roc_df, aes(x = FPR, y = TPR)) +
  geom_line(colour = "steelblue", linewidth = 0.9) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey50") +
  annotate("text", x = 0.65, y = 0.15,
           label = paste0("AUC = ", round(auc(roc_obj), 4)),
           size = 4.5, colour = "steelblue") +
  labs(title = "ROC Curve — Stepwise Logistic Regression",
       x = "1 - Specificity (FPR)", y = "Sensitivity (TPR)") +
  theme_bw()

ggsave("figures/roc_stepwise_logistic.png", p_roc,
       width = 6, height = 5, dpi = 150)

print(Sys.time() - start_time)
