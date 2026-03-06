# =============================================================================
# 02_data_split.R
# data loading, preprocessing, train/test split, imputation, SMOTENC
# =============================================================================

library(dplyr)
library(ggplot2)      # missingness plot
library(caret)        # createDataPartition
library(themis)       # smotenc
set.seed(42)

# -----------------------------------------------------------------------------
# 0. Load and filter data
# Drop leakage, identifier columns and other columns that I do not want to consider as covariates. Only use ICU patients. Recode race to have fewer levels.
# -----------------------------------------------------------------------------
df_raw <- read.csv("data/Assignment1_mimic dataset.csv", stringsAsFactors = FALSE)

# List of units which I will consider as ICU units. The rest of the units are not considered as ICU units
icu_units <- c(
  "Medical Intensive Care Unit (MICU)",
  "Cardiac Vascular Intensive Care Unit (CVICU)",
  "Medical/Surgical Intensive Care Unit (MICU/SICU)",
  "Surgical Intensive Care Unit (SICU)",
  "Trauma SICU (TSICU)",
  "Coronary Care Unit (CCU)",
  "Neuro Surgical Intensive Care Unit (Neuro SICU)",
  "Intensive Care Unit (ICU)"
)

# Units that are not considered ICU units are: PACU, Med/Surg, Medicine, Neurology, Neuro Stepdown, Surgery/Trauma, Neuro Intermediate, and Surgery/Vascular/Intermediate.
all_units <- unique(c(df_raw$first_careunit,df_raw$last_careunit))
non_icu_units <- setdiff(all_units, icu_units)

# Exclude identifier columns, ethically excluded variables (I have chosen language and marital status, insurance type), as well as columns that would confer unfair information about the outcome variable (i.e. leakage columns)
exclude_cols <- c(
  "subject_id", "hadm_id", "stay_id",
  "first_careunit", "last_careunit", # Do not allow the type of ICU to affect mortality rate estimate ...
  "intime", "outtime", "los", "deathtime", # Do not allow length of stay information to influence predictions, because this information only available "after the fact"
  "insurance", "language", # I deliberately do not want a model that discriminates based on insurance or language. 
  "marital_status", # Based on prior knowledge I do not think marital status would influence how sick a patient is in the ICU.
  "hospital_expire_flag" # Leakage variable that specifies whether the patient died in hospital. ICU deaths are a subset of hospital deaths.
)

# Only include the data that has non-missing intime/outime, patients who have NEVER been in the ICU, and exclude the unwanted columns
df <- df_raw %>%
  filter(!is.na(outtime) & outtime != "") %>%
  filter(first_careunit %in% icu_units | last_careunit %in% icu_units) %>%
  select(-all_of(exclude_cols))

# Recode race
df$race <- dplyr::case_when(
  df$race %in% c("ASIAN", "ASIAN - CHINESE", "ASIAN - KOREAN",
                 "ASIAN - SOUTH EAST ASIAN", "ASIAN - ASIAN INDIAN")                    ~ "Asian",
  df$race %in% c("HISPANIC OR LATINO", "HISPANIC/LATINO - PUERTO RICAN",
                 "HISPANIC/LATINO - MEXICAN", "HISPANIC/LATINO - DOMINICAN",
                 "HISPANIC/LATINO - GUATEMALAN", "HISPANIC/LATINO - SALVADORAN",
                 "HISPANIC/LATINO - CUBAN", "HISPANIC/LATINO - COLUMBIAN",
                 "HISPANIC/LATINO - HONDURAN", "HISPANIC/LATINO - CENTRAL AMERICAN",
                 "SOUTH AMERICAN")                                                       ~ "Hispanic/Latino",
  df$race %in% c("WHITE", "WHITE - OTHER EUROPEAN", "WHITE - BRAZILIAN",
                 "WHITE - RUSSIAN", "WHITE - EASTERN EUROPEAN", "PORTUGUESE")           ~ "White",
  df$race %in% c("BLACK/AFRICAN AMERICAN", "BLACK/AFRICAN",
                 "BLACK/CAPE VERDEAN", "BLACK/CARIBBEAN ISLAND")                        ~ "Black",
  df$race %in% c("UNABLE TO OBTAIN", "UNKNOWN", "PATIENT DECLINED TO ANSWER")          ~  "Unknown", # deliberate choice to keep Unknown label instead of "NA"
  df$race %in% c("NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER",
                 "MULTIPLE RACE/ETHNICITY", "OTHER")                                    ~ "Other Race", # deliberate choice to keep Other label instead of lumping together with Unknown
  df$race %in% c("AMERICAN INDIAN/ALASKA NATIVE")                                       ~ "Native American",
  TRUE                                                                                   ~ df$race
)


# ------- 1. Define target + all candidate features (missingness filter done post-split)
target               <- "icu_death_flag"
categorical_features <- c("race", "gender")
numerical_features   <- setdiff(names(df)[sapply(df, is.numeric)], target)
all_features         <- c(categorical_features, numerical_features)

df <- df %>% select(all_of(c(all_features, target)))

# set factor levels before split for consistnecy
df[[target]] <- factor(df[[target]], levels = c(0, 1),
                             labels = c("Discharged", "DiedinICU"))
df$race   <- factor(df$race)
df$gender <- factor(df$gender)

# 1b. save dataset before train/test split

dir.create("data", showWarnings = FALSE)
saveRDS(df,  "data/df.rds")

# ---------- 2. Stratified 70/30 train/test split ----------
train_idx <- createDataPartition(df[[target]], p = 0.70, list = FALSE)
df_train  <- df[ train_idx, ]
df_test   <- df[-train_idx, ]


print(round(prop.table(table(df_train[[target]])), 4))
print(round(prop.table(table(df_test[[target]])),  4))

# 2b. filter features by missingness threshold - computed on training set only
MISSING_THRESHOLD <- 10  # percent

selected_num <- numerical_features[
  sapply(numerical_features, function(f) mean(is.na(df_train[[f]])) * 100) <= MISSING_THRESHOLD
]
selected_cat <- categorical_features[
  sapply(categorical_features, function(f) mean(is.na(df_train[[f]])) * 100) <= MISSING_THRESHOLD
]
selected_features <- c(selected_cat, selected_num)

num_features_chosen <- length(selected_features)
num_features_available <- ncol(df)-1
prop_features_chosen <- num_features_chosen / num_features_available

#  Both df_train and df_test are subsetted to the selected features  that were chosen based on missingness threshold analysis performed on the training set only.    
df_train <- df_train %>% select(all_of(c(selected_features, target)))
df_test  <- df_test  %>% select(all_of(c(selected_features, target)))


## Create histogram showing % missingness

df_train_x <- df_train %>% select(-icu_death_flag)
df_test_x  <- df_test  %>% select(-icu_death_flag)

miss_df <- rbind(
  data.frame(variable = names(df_train_x),
             pct_missing = colMeans(is.na(df_train_x)) * 100,
             split = "Train"),
  data.frame(variable = names(df_test_x),
             pct_missing = colMeans(is.na(df_test_x)) * 100,
             split = "Test")
)

var_order <- miss_df %>% filter(split == "Train") %>% arrange(desc(pct_missing)) %>% pull(variable)
miss_df$variable <- factor(miss_df$variable, levels = var_order)

missingness_plot <- ggplot(miss_df, aes(x = variable, y = pct_missing)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  facet_wrap(~ split) +
  labs(x = "Variable", y = "% Missing", title = "Missingness by Variable (Train vs Test)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

ggsave("figures/missingness_plot_train_test.png", plot = missingness_plot,
       width = 14, height = 6, dpi = 300)

# -----------------------------------------------------------------------------
# 3. Impute missing values with median (fit on train set only)
# apply same medians to test set too
# -----------------------------------------------------------------------------
train_medians <- sapply(selected_num, function(f) median(df_train[[f]], na.rm = TRUE))

impute_medians <- function(df, medians) {
  for (f in names(medians)) df[[f]][is.na(df[[f]])] <- medians[f]
  df
}

df_train <- impute_medians(df_train, train_medians)
df_test  <- impute_medians(df_test,  train_medians)


# -----------------------------------------------------------------------------
# 3b. z-score standardization - fit on train only, apply to both
#     only numeric features scaled; race & gender left as factors.
#     do this BEFORE smotenc so synthetic samples use standardized space
# -----------------------------------------------------------------------------
train_means <- sapply(selected_num, function(f) mean(df_train[[f]]))
train_sds   <- sapply(selected_num, function(f) sd(df_train[[f]]))
train_sds[train_sds == 0] <- 1   # avoid div by zero for constant cols

zscore_scale <- function(df, means, sds) {
  for (f in names(means)) df[[f]] <- (df[[f]] - means[f]) / sds[f]
  df
}

df_train <- zscore_scale(df_train, train_means, train_sds)
df_test  <- zscore_scale(df_test,  train_means, train_sds)


# -----------------------------------------------------------------------------
# 4. SMOTENC on training set only
#    handles numeric + categorical simultaneously
#    numeric: linearly interpolated btwn neighbours
#    categorical: majority vote among K neighbours
#    over_ratio=1 -> balance minority to match majority
# -----------------------------------------------------------------------------
print(table(df_train[[target]]))

df_train_pre_smote <- df_train   # keep a copy before SMOTE for partial-balance variant
df_train <- smotenc(df_train, var = target, k = 5, over_ratio = 1)

print(table(df_train[[target]]))

# 4b. Partially-balanced training set for SVM (minority = 50% of majority)
#     Smaller dataset speeds up SVM substantially.
df_train_small <- smotenc(df_train_pre_smote, var = target, k = 5, over_ratio = 0.5)
cat("df_train_small class balance:\n")
print(table(df_train_small[[target]]))

# 5. save
dir.create("data", showWarnings = FALSE)

saveRDS(df_train,       "data/train.rds")         # SMOTENC-balanced, scaled, factors intact
saveRDS(df_train_small, "data/train_small.rds")   # partial SMOTE (0.5), for SVM
saveRDS(df_test,        "data/test.rds")           # scaled, factors intact, no SMOTE

# Save all the features
saveRDS(selected_features, "data/candidate_features.rds")
