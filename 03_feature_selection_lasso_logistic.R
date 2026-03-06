# =============================================================================
# 03_feature_selection_lasso_logistic.R
# LASSO (L1) logistic regression for feature selection
#
# Workflow:
#   1. Load train data
#   2. Fit cv.glmnet with alpha=1 (LASSO), lambda selected by CV AUC
#   3. Extract nonzero coef features (lambda.1se - more parsimonious)
#   4. collapse model.matrix dummy names back to original col names
#   5. Save selected_features_lasso.rds
# =============================================================================

start_time <- Sys.time()

library(glmnet)

set.seed(42)

target <- "icu_death_flag"

# 1. load data
df_train <- readRDS("data/train.rds")

# ---- 2. prepare model matrix
# glmnet needs a numeric matrix, model.matrix one-hot encodes factors
x_train <- model.matrix(
  as.formula(paste("~", paste(setdiff(names(df_train), target), collapse = "+"))),
  data = df_train
)[, -1]   # drop intercept column

y_train <- as.integer(df_train[[target]] == "Discharged")

# 3. fit LASSO via CV (alpha=1, criterion=AUC)
#    using lambda.1se - most parsimonious lambda within 1SE of min CV error
lasso_cv <- cv.glmnet(x_train, y_train,
                      family       = "binomial",
                      alpha        = 1,
                      type.measure = "auc",
                      nfolds       = 5)


# --- 4. Extract selected feautres and collapse dummy names to orignal column names
lasso_coefs <- coef(lasso_cv, s = "lambda.1se")
coef_names  <- rownames(lasso_coefs)[which(lasso_coefs != 0)]

# map model.matrix dummy names (eg "raceBlack") back to original col names
selected_features_lasso <- unique(gsub("(race|gender).*", "\\1", coef_names))
selected_features_lasso <- selected_features_lasso[selected_features_lasso != "(Intercept)"]

print(selected_features_lasso)

# 5. save
dir.create("data", showWarnings = FALSE)
saveRDS(selected_features_lasso, "data/selected_features_lasso.rds")

print(Sys.time() - start_time)
