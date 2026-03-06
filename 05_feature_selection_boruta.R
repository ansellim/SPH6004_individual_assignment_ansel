# 05_feature_selection_boruta.R
# DECLARATION: AI tools (Anthropic Claude Code) were used in the editing and development of this code.
# Boruta feature selection - wrapper method based on random forest importance.
# Compares each feature's importance against randomised shadow features to decide confirm/reject.

start_time <- Sys.time()
library(Boruta)
set.seed(42)


df_train <- readRDS("data/train.rds")
df_train$icu_death_flag <- as.factor(df_train$icu_death_flag)

# 2. fit boruta - factors handled natively, no need to create model matrix

bor <- Boruta(icu_death_flag ~ . , data = df_train, doTrace = 2)

# 3. resolve tentative features and extract confirmed ones

bor_fixed <- TentativeRoughFix(bor)

selected_features_boruta <- getSelectedAttributes(bor_fixed)

dir.create("data", showWarnings = FALSE)
saveRDS(selected_features_boruta, "data/selected_features_boruta.rds")

print(Sys.time() - start_time)