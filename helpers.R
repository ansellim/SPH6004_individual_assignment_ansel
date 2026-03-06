# shared helpers
# DECLARATION: AI tools (Anthropic Claude Code) were used in the editing and development of this code.

compute_metrics <- function(actual, prob, threshold = 0.5) {
  roc_obj <- roc(actual, prob, quiet = TRUE)
  auc_val <- as.numeric(auc(roc_obj))
  pred <- ifelse(prob >= threshold, 1L, 0L)
  tp <- sum(pred == 1 & actual == 1); tn <- sum(pred == 0 & actual == 0)
  fp <- sum(pred == 1 & actual == 0); fn <- sum(pred == 0 & actual == 1)
  list(AUC         = round(auc_val,    4),
       Sensitivity = round(tp/(tp+fn), 4),
       Specificity = round(tn/(tn+fp), 4),
       PPV         = round(tp/(tp+fp), 4),
       NPV         = round(tn/(tn+fn), 4))
}
