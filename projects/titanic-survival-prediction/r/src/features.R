# Feature engineering on top of the cleaned Titanic data.
#
# Unlike the Python version (which one-hot encodes categoricals by hand),
# this keeps Sex/Embarked/Title as factors: R's modelling functions
# (glm, randomForest, gbm) encode factor predictors natively, so a manual
# dummy-encoding step would just duplicate what the model formula already does.

CLEANED_PATH <- "data/processed/cleaned.csv"
FEATURES_PATH <- "data/processed/features.csv"

FEATURE_COLUMNS <- c(
  "Pclass", "Sex", "Age", "Fare", "Embarked",
  "FamilySize", "IsAlone", "Title", "HasCabin"
)
TARGET_COLUMN <- "Survived"

engineer <- function(df) {
  df$FamilySize <- df$SibSp + df$Parch + 1
  df$IsAlone <- as.integer(df$FamilySize == 1)
  df
}

build_model_frame <- function(df) {
  df <- engineer(df)
  cols <- FEATURE_COLUMNS
  if (TARGET_COLUMN %in% names(df)) cols <- c(cols, TARGET_COLUMN)
  model_df <- df[, cols]

  model_df$Pclass <- factor(model_df$Pclass, levels = c(1, 2, 3), ordered = TRUE)
  model_df$Sex <- factor(model_df$Sex)
  model_df$Embarked <- factor(model_df$Embarked)
  model_df$Title <- factor(model_df$Title)

  model_df
}

main <- function() {
  df <- read.csv(CLEANED_PATH, stringsAsFactors = FALSE)
  model_df <- build_model_frame(df)
  write.csv(model_df, FEATURES_PATH, row.names = FALSE)
  n_features <- ncol(model_df) - as.integer(TARGET_COLUMN %in% names(model_df))
  cat(sprintf("Wrote %d features for %d rows to %s\n", n_features, nrow(model_df), FEATURES_PATH))
}

main()
