# Cross-validate candidate models, select the best, and save it with its metrics.
suppressMessages({
  library(caret)
  library(randomForest)
  library(gbm)
  library(pROC)
  library(jsonlite)
})

FEATURES_PATH <- "data/processed/features.csv"
METRICS_PATH <- "results/metrics.json"
MODEL_PATH <- "results/model.rds"
RANDOM_STATE <- 42
GBM_NTREES <- 100

MODEL_NAMES <- c("logistic_regression", "random_forest", "gradient_boosting")

load_features <- function(path = FEATURES_PATH) {
  df <- read.csv(path, stringsAsFactors = FALSE)
  df$Pclass <- factor(df$Pclass, levels = c(1, 2, 3), ordered = TRUE)
  df$Sex <- factor(df$Sex)
  df$Embarked <- factor(df$Embarked)
  df$Title <- factor(df$Title)
  df$Survived <- factor(ifelse(df$Survived == 1, "Yes", "No"), levels = c("No", "Yes"))
  df
}

fit_model <- function(name, data) {
  if (name == "logistic_regression") {
    glm(Survived ~ ., data = data, family = binomial)
  } else if (name == "random_forest") {
    randomForest(Survived ~ ., data = data, ntree = 300)
  } else if (name == "gradient_boosting") {
    d <- data
    d$Survived <- as.numeric(d$Survived == "Yes")
    gbm(Survived ~ ., data = d, distribution = "bernoulli",
        n.trees = GBM_NTREES, interaction.depth = 3, shrinkage = 0.1, verbose = FALSE)
  } else {
    stop(sprintf("Unknown model: %s", name))
  }
}

predict_proba <- function(name, model, newdata) {
  if (name == "logistic_regression") {
    predict(model, newdata, type = "response")
  } else if (name == "random_forest") {
    predict(model, newdata, type = "prob")[, "Yes"]
  } else if (name == "gradient_boosting") {
    predict(model, newdata, n.trees = GBM_NTREES, type = "response")
  } else {
    stop(sprintf("Unknown model: %s", name))
  }
}

compute_metrics <- function(y_true, prob, threshold = 0.5) {
  pred_class <- factor(ifelse(prob > threshold, "Yes", "No"), levels = c("No", "Yes"))
  y_true <- factor(y_true, levels = c("No", "Yes"))
  cm <- confusionMatrix(pred_class, y_true, positive = "Yes")
  roc_obj <- roc(response = y_true, predictor = as.numeric(prob), levels = c("No", "Yes"),
                  direction = "<", quiet = TRUE)
  list(
    accuracy = unname(cm$overall["Accuracy"]),
    precision = unname(cm$byClass["Precision"]),
    recall = unname(cm$byClass["Recall"]),
    f1 = unname(cm$byClass["F1"]),
    roc_auc = as.numeric(auc(roc_obj))
  )
}

cross_validate_models <- function(data_full, folds) {
  results <- list()
  for (name in MODEL_NAMES) {
    fold_vals <- list(accuracy = c(), precision = c(), recall = c(), f1 = c(), roc_auc = c())
    for (test_idx in folds) {
      train_data <- data_full[-test_idx, ]
      test_data <- data_full[test_idx, ]
      model <- fit_model(name, train_data)
      prob <- predict_proba(name, model, test_data)
      m <- compute_metrics(test_data$Survived, prob)
      for (metric in names(fold_vals)) fold_vals[[metric]] <- c(fold_vals[[metric]], m[[metric]])
    }
    results[[name]] <- lapply(fold_vals, function(v) list(mean = mean(v), std = sd(v)))
  }
  results
}

main <- function() {
  data_full <- load_features()

  set.seed(RANDOM_STATE)
  folds <- createFolds(data_full$Survived, k = 5, list = TRUE, returnTrain = FALSE)

  cv_results <- cross_validate_models(data_full, folds)

  accuracies <- sapply(cv_results, function(r) r$accuracy$mean)
  best_model_name <- names(which.max(accuracies))
  cat(sprintf("Best model by mean CV accuracy: %s\n", best_model_name))

  set.seed(RANDOM_STATE)
  train_idx <- createDataPartition(data_full$Survived, p = 0.8, list = FALSE)
  train_data <- data_full[train_idx, ]
  test_data <- data_full[-train_idx, ]

  # Ship the holdout-trained model, not one refit on all data: evaluate.R
  # re-scores it against test_data, which would leak train data into the
  # score if the saved model had already seen it.
  holdout_model <- fit_model(best_model_name, train_data)
  prob <- predict_proba(best_model_name, holdout_model, test_data)
  holdout_metrics <- compute_metrics(test_data$Survived, prob)

  saveRDS(list(name = best_model_name, model = holdout_model), MODEL_PATH)

  output <- list(
    random_state = RANDOM_STATE,
    cross_validation = cv_results,
    best_model = best_model_name,
    holdout_metrics = holdout_metrics,
    feature_columns = setdiff(names(data_full), "Survived")
  )
  write_json(output, METRICS_PATH, auto_unbox = TRUE, pretty = TRUE, digits = 10)
  cat(sprintf("Wrote metrics to %s\n", METRICS_PATH))
  cat(sprintf("Saved final model to %s\n", MODEL_PATH))
}

main()
