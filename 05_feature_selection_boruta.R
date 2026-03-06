# 05_feature_selection_boruta.R
# boruta (wrapper method) for feature selection
# uses random forest importance + statistical testing against shadow features

start_time <- Sys.time()
library(Boruta)
set.seed(42)

# ---- 1. load data ----

df_train <- readRDS("data/train.rds")
df_train$icu_death_flag <- as.factor(df_train$icu_death_flag)

# 2. fit boruta
# ranger handles factors natively so no need to create model matrix

bor <- Boruta(icu_death_flag ~ . , data = df_train, doTrace = 2)

# 3. get selected features
# TentativeRoughFix resolves any features that are still tentative

bor_fixed <- TentativeRoughFix(bor)

selected_features_boruta <- getSelectedAttributes(bor_fixed)

# ------- 4. save -------
dir.create("data", showWarnings = FALSE)
saveRDS(selected_features_boruta, "data/selected_features_boruta.rds")

print(Sys.time() - start_time)