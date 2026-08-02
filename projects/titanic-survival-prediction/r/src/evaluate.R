# Print a metrics summary and save diagnostic plots for the selected model.
suppressMessages({
  library(caret)
  library(randomForest)
  library(gbm)
  library(pROC)
  library(jsonlite)
  library(ggplot2)
})

FEATURES_PATH <- "data/processed/features.csv"
METRICS_PATH <- "results/metrics.json"
MODEL_PATH <- "results/model.rds"
FIGURES_DIR <- "results/figures"
GBM_NTREES <- 100

load_features <- function(path = FEATURES_PATH) {
  df <- read.csv(path, stringsAsFactors = FALSE)
  df$Pclass <- factor(df$Pclass, levels = c(1, 2, 3), ordered = TRUE)
  df$Sex <- factor(df$Sex)
  df$Embarked <- factor(df$Embarked)
  df$Title <- factor(df$Title)
  df$Survived <- factor(ifelse(df$Survived == 1, "Yes", "No"), levels = c("No", "Yes"))
  df
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

print_summary <- function(metrics) {
  cat("Cross-validated accuracy (5-fold, mean +/- std):\n")
  for (name in names(metrics$cross_validation)) {
    acc <- metrics$cross_validation[[name]]$accuracy
    cat(sprintf("  %-20s %.3f +/- %.3f\n", name, acc$mean, acc$std))
  }
  cat(sprintf("\nSelected model: %s\n", metrics$best_model))
  cat("Held-out evaluation (20% stratified split):\n")
  for (metric in names(metrics$holdout_metrics)) {
    cat(sprintf("  %-10s %.3f\n", metric, metrics$holdout_metrics[[metric]]))
  }
}

save_diagnostic_plots <- function(y_true, prob, threshold = 0.5) {
  pred_class <- factor(ifelse(prob > threshold, "Yes", "No"), levels = c("No", "Yes"))
  y_true <- factor(y_true, levels = c("No", "Yes"))
  cm <- confusionMatrix(pred_class, y_true, positive = "Yes")

  cm_df <- as.data.frame(cm$table)
  names(cm_df) <- c("Predicted", "Actual", "Freq")
  label_map <- c(No = "Did not survive", Yes = "Survived")
  cm_df$Predicted <- factor(label_map[as.character(cm_df$Predicted)], levels = label_map)
  cm_df$Actual <- factor(label_map[as.character(cm_df$Actual)], levels = rev(label_map))

  p_cm <- ggplot(cm_df, aes(x = Predicted, y = Actual, fill = Freq)) +
    geom_tile(color = "white", linewidth = 1) +
    geom_text(aes(label = Freq), size = 6, color = "#0b0b0b") +
    scale_fill_gradient(low = "#EAF2FC", high = "#1F3B57", guide = "none") +
    labs(title = "Confusion Matrix (held-out set)", x = "Predicted label", y = "True label") +
    theme_minimal(base_size = 12) +
    theme(panel.grid = element_blank(), plot.title = element_text(face = "bold"))
  ggsave(file.path(FIGURES_DIR, "confusion_matrix.svg"), p_cm, width = 5.2, height = 4.2, device = "svg")

  roc_obj <- roc(response = y_true, predictor = as.numeric(prob), levels = c("No", "Yes"),
                  direction = "<", quiet = TRUE)
  roc_df <- data.frame(
    fpr = 1 - roc_obj$specificities,
    tpr = roc_obj$sensitivities
  )
  roc_df <- roc_df[order(roc_df$fpr), ]
  auc_val <- as.numeric(auc(roc_obj))

  p_roc <- ggplot(roc_df, aes(x = fpr, y = tpr)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#cccccc") +
    geom_line(color = "#2A78D6", linewidth = 1.1) +
    labs(
      title = "ROC Curve (held-out set)",
      subtitle = sprintf("AUC = %.2f", auc_val),
      x = "False Positive Rate", y = "True Positive Rate"
    ) +
    theme_minimal(base_size = 12) +
    theme(plot.title = element_text(face = "bold"))
  ggsave(file.path(FIGURES_DIR, "roc_curve.svg"), p_roc, width = 5, height = 4.2, device = "svg")
}

main <- function() {
  metrics <- fromJSON(METRICS_PATH)
  print_summary(metrics)

  saved <- readRDS(MODEL_PATH)

  df <- load_features()
  set.seed(metrics$random_state)
  # Same split as train.R, recreated from the recorded random_state, so this
  # is the same held-out slice the saved model was scored against (and never trained on).
  train_idx <- createDataPartition(df$Survived, p = 0.8, list = FALSE)
  test_data <- df[-train_idx, ]

  prob <- predict_proba(saved$name, saved$model, test_data)
  save_diagnostic_plots(test_data$Survived, prob)
  cat(sprintf("\nSaved diagnostic plots to %s/\n", FIGURES_DIR))
}

main()
