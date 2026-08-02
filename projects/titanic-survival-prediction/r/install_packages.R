# Package versions this pipeline was developed and verified against
# (R 4.3.3). Exact pins would normally live in an renv.lock; this is the
# lightweight equivalent for a small, single-machine project.
#
# dplyr        1.1.4
# stringr      1.5.1
# caret        6.0.94
# randomForest 4.7.1.1
# gbm          2.1.8.1
# pROC         1.18.5
# jsonlite     1.8.8
# ggplot2      3.4.4

install.packages(c(
  "dplyr", "stringr", "caret", "randomForest",
  "gbm", "pROC", "jsonlite", "ggplot2"
))
