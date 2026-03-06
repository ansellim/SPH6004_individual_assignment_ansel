# =============================================================================
# 01_eda.R
# Exploratory Data Analysis - SPH6004 Individual Assignment
# not submitted; for personal understanding only
# =============================================================================

library(ggplot2)
library(dplyr)
library(tidyr)
library(corrplot)

# -----------------------------------------------------------------------------
# 1. Load data
# -----------------------------------------------------------------------------
df_raw <- read.csv("data/Assignment1_mimic dataset.csv", stringsAsFactors = FALSE)

# -----------------------------------------------------------------------------
# 2. Filter rows: Only use the patients who have ever been admitted to ICU
# -----------------------------------------------------------------------------
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

df <- df_raw %>%
  filter(!is.na(outtime) & outtime != "") %>%
  filter(first_careunit %in% icu_units | last_careunit %in% icu_units)


# -----------------------------------------------------------------------------
# 3. Select candidate features + target
# -----------------------------------------------------------------------------

# Exclude identifier columns, language and marital status, insurance type, as well as columns that would confer unfair information about the outcome variable (i.e. leakage columns)
exclude_cols <- c(
  "subject_id", "hadm_id", "stay_id",
  "first_careunit", "last_careunit",
  "intime", "outtime", "los", "deathtime",
  "insurance", "language", "marital_status",
  "hospital_expire_flag"
)

df <- df %>% select(-all_of(exclude_cols))

# Identify feature types
target       <- "icu_death_flag"
cat_features <- c("race", "gender")
num_features <- setdiff(names(df)[sapply(df, is.numeric)], target)

# sanity check - flag unexpected character/non-numeric cols
char_cols <- names(df)[sapply(df, is.character)]
unexpected <- setdiff(char_cols, c(target, cat_features))
if (length(unexpected) > 0) {
  warning("Unexpected character columns found: ", paste(unexpected, collapse = ", "))
}


# Convert target and categoricals
df[[target]] <- factor(df[[target]], levels = c(0, 1),
                       labels = c("Discharged", "Died in ICU"))
df$gender <- factor(df$gender)
df$race   <- factor(df$race)

# -----------------------------------------------------------------------------
# 4. Class distribution of target
# target is imbalanced, probably need SMOTE or similar
# -----------------------------------------------------------------------------
print(table(df[[target]]))
print(prop.table(table(df[[target]])))

ggplot(df, aes(x = icu_death_flag, fill = icu_death_flag)) +
  geom_bar() +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.5) +
  scale_fill_manual(values = c("steelblue", "tomato")) +
  labs(title = "ICU Discharge Status (Target Variable)",
       x = NULL, y = "Count", fill = NULL) +
  theme_bw() +
  theme(legend.position = "none")

# -----------------------------------------------------------------------------
# 5. Missing value analysis
# -----------------------------------------------------------------------------
candidate_features <- c(cat_features, num_features)

missing_summary <- data.frame(
  feature     = candidate_features,
  missing_pct = sapply(candidate_features, function(f) mean(is.na(df[[f]])) * 100)
) %>% arrange(desc(missing_pct))

print(missing_summary, row.names = FALSE)

# histogram of % missingness accross candidate features
ggplot(missing_summary, aes(x = missing_pct)) +
  geom_histogram(binwidth = 5, fill = "steelblue", color = "white") +
  scale_x_continuous(breaks = seq(0, 100, 10)) +
  labs(title = "Distribution of % Missingness Across Candidate Features",
       x = "% Missing", y = "Number of Features") +
  theme_bw()

# -----------------------------------------------------------------------------
# 6. Summary statistics for numeric features
# -----------------------------------------------------------------------------
summary(df %>% select(all_of(num_features)))

# -----------------------------------------------------------------------------
# 7. Categorical feature distributions
# -----------------------------------------------------------------------------
# Gender
ggplot(df, aes(x = gender, fill = icu_death_flag)) +
  geom_bar(position = "fill") +
  labs(title = "Gender vs ICU Outcome", x = "Gender",
       y = "Proportion", fill = "Outcome") +
  scale_fill_manual(values = c("steelblue", "tomato")) +
  theme_bw()

# Race (top 10)
top_races <- names(sort(table(df$race), decreasing = TRUE))[1:10]
df %>%
  filter(race %in% top_races) %>%
  ggplot(aes(x = reorder(race, race, function(x) -length(x)),
             fill = icu_death_flag)) +
  geom_bar(position = "fill") +
  coord_flip() +
  labs(title = "Race (top 10) vs ICU Outcome", x = "Race",
       y = "Proportion", fill = "Outcome") +
  scale_fill_manual(values = c("steelblue", "tomato")) +
  theme_bw()

# -----------------------------------------------------------------------------
# 7b. Recode race variable
# -----------------------------------------------------------------------------
# convert to character for recoding (race is currently a factor)
df$race <- as.character(df$race)

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
  df$race %in% c("UNABLE TO OBTAIN", "UNKNOWN", "PATIENT DECLINED TO ANSWER")          ~ "Unknown",
  df$race %in% c("NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER",
                 "MULTIPLE RACE/ETHNICITY", "OTHER")                                    ~ "Other",
  df$race %in% c("AMERICAN INDIAN/ALASKA NATIVE")                                       ~ "Native American",
  TRUE                                                                                   ~ df$race
)

df$race <- factor(df$race)

print(sort(table(df$race), decreasing = TRUE))

# -----------------------------------------------------------------------------
# 7c. Race vs ICU outcome (recoded)
# -----------------------------------------------------------------------------
ggplot(df, aes(x = reorder(race, race, function(x) -length(x)),
               fill = icu_death_flag)) +
  geom_bar(position = "fill") +
  coord_flip() +
  scale_fill_manual(values = c("steelblue", "tomato")) +
  labs(title = "Race vs ICU Outcome (after recoding)",
       x = "Race", y = "Proportion", fill = "Outcome") +
  theme_bw()

# -----------------------------------------------------------------------------
# 8. Select candidate features with < 10% missingness
# -----------------------------------------------------------------------------
MISSING_THRESHOLD <- 10  # percent

selected_num_features <- num_features[
  sapply(num_features, function(f) mean(is.na(df[[f]])) * 100) < MISSING_THRESHOLD
]

selected_cat_features <- cat_features[
  sapply(cat_features, function(f) mean(is.na(df[[f]])) * 100) < MISSING_THRESHOLD
]

selected_features <- c(selected_cat_features, selected_num_features)


# -----------------------------------------------------------------------------
# 9. Correlation heatmap (selected numeric feautres, complete cases)
# -----------------------------------------------------------------------------
cor_data <- df %>%
  select(all_of(selected_num_features)) %>%
  drop_na()


cor_mat <- cor(cor_data, method = "pearson")
corrplot(cor_mat, method = "color", type = "upper",
         tl.cex = 0.6, tl.col = "black",
         title = "Pearson Correlation — Key Features",
         mar = c(0, 0, 2, 0))

# -----------------------------------------------------------------------------
# 10. Age distribution by outcome
# -----------------------------------------------------------------------------
ggplot(df, aes(x = age, fill = icu_death_flag)) +
  geom_histogram(bins = 40, position = "identity", alpha = 0.6) +
  scale_fill_manual(values = c("steelblue", "tomato")) +
  labs(title = "Age Distribution by ICU Outcome",
       x = "Age", y = "Count", fill = "Outcome") +
  theme_bw()
