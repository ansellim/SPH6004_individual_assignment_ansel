# 03_feature_selection_lasso_logistic.R
# DECLARATION: AI tools (Anthropic Claude Code) were used in the editing and development of this code.
# LASSO (L1) logistic regression for feature selection
#
# Workflow:
#   1. Load train data
#   2. Fit cv.glmnet with alpha=1 (LASSO), lambda selected by CV AUC
#   3. Extract nonzero coef features (lambda.1se - more parsimonious)
#   4. collapse model.matrix dummy names back to original col names
#   5. Save selected_features_lasso.rds

start_time <- Sys.time()

library(glmnet)

set.seed(42)

target <- "icu_death_flag"

# 1. load data
df_train <- readRDS("data/train.rds")

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


lasso_coefs <- coef(lasso_cv, s = "lambda.1se")
coef_names  <- rownames(lasso_coefs)[which(lasso_coefs != 0)]

# map model.matrix dummy names (eg "raceBlack") back to original col names
selected_features_lasso <- unique(gsub("(race|gender).*", "\\1", coef_names))
selected_features_lasso <- selected_features_lasso[selected_features_lasso != "(Intercept)"]

print(selected_features_lasso)

# 5. save
dir.create("data", showWarnings = FALSE)
saveRDS(selected_features_lasso, "data/selected_features_lasso.rds")

# 6. CV curve plot with number of features retained at each lambda
dir.create("figures", showWarnings = FALSE)

# compute number of non-zero coefficients (excluding intercept) at each lambda
n_features <- colSums(as.matrix(lasso_cv$glmnet.fit$beta) != 0)
# align to the lambdas evaluated by cv.glmnet
lambda_seq   <- lasso_cv$lambda
cvm          <- lasso_cv$cvm          # mean CV AUC
cvsd         <- lasso_cv$cvsd         # SD of CV AUC
lambda_min   <- lasso_cv$lambda.min
lambda_1se   <- lasso_cv$lambda.1se

# match n_features to cv lambda sequence
n_feat_cv <- n_features[match(round(lambda_seq, 10),
                               round(lasso_cv$glmnet.fit$lambda, 10))]

png("figures/lasso_cv_curve.png", width = 800, height = 550, res = 110)

par(mar = c(5, 4.5, 5, 2))

plot(log(lambda_seq), cvm,
     type  = "l", col = "steelblue", lwd = 2,
     xlab  = expression(log(lambda)),
     ylab  = "Cross-validated AUC (5-fold)",
     main  = "LASSO: CV AUC vs regularisation strength",
     ylim  = range(c(cvm - cvsd, cvm + cvsd)),
     xaxt  = "n")

# error ribbon
polygon(c(log(lambda_seq), rev(log(lambda_seq))),
        c(cvm - cvsd,      rev(cvm + cvsd)),
        col = adjustcolor("steelblue", alpha.f = 0.15), border = NA)

# custom x-axis with log(lambda) labels
axis(1)

# number of features on top axis
unique_nfeat <- n_feat_cv[!duplicated(n_feat_cv)]
unique_log_lambda <- sapply(unique(n_feat_cv), function(n) {
  mean(log(lambda_seq[n_feat_cv == n]))
})
axis(3, at = unique_log_lambda, labels = unique(n_feat_cv),
     cex.axis = 0.7, las = 2)
mtext("Number of features retained", side = 3, line = 3, cex = 0.85)

# vertical lines for lambda.min and lambda.1se
abline(v = log(lambda_min), lty = 2, col = "firebrick")
abline(v = log(lambda_1se), lty = 2, col = "darkgreen")
legend("bottomleft",
       legend = c(sprintf("lambda.min (%d features)", n_feat_cv[which.min(abs(lambda_seq - lambda_min))]),
                  sprintf("lambda.1se (%d features)", n_feat_cv[which.min(abs(lambda_seq - lambda_1se))])),
       col    = c("firebrick", "darkgreen"),
       lty    = 2, bty = "n", cex = 0.85)

dev.off()

print(Sys.time() - start_time)
