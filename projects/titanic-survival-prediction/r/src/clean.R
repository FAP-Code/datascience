# Load and clean the raw Titanic passenger data.
suppressMessages(library(dplyr))
suppressMessages(library(stringr))

RAW_PATH <- "../data/raw/titanic.csv"  # shared with the Python pipeline, one level up
PROCESSED_PATH <- "data/processed/cleaned.csv"

TITLE_MAP <- c(
  Mlle = "Miss", Ms = "Miss", Mme = "Mrs",
  Lady = "Rare", Countess = "Rare", Capt = "Rare", Col = "Rare",
  Don = "Rare", Dr = "Rare", Major = "Rare", Rev = "Rare",
  Sir = "Rare", Jonkheer = "Rare", Dona = "Rare",
  "the Countess" = "Rare"  # raw Name field spells this title "the Countess"
)

extract_title <- function(name) {
  title <- str_trim(str_split_fixed(str_split_fixed(name, ",", 2)[, 2], "\\.", 2)[, 1])
  ifelse(title %in% names(TITLE_MAP), unname(TITLE_MAP[title]), title)
}

load_raw <- function(path = RAW_PATH) {
  read.csv(path, na.strings = c("", "NA"), stringsAsFactors = FALSE)
}

clean <- function(df) {
  df$Title <- vapply(df$Name, extract_title, character(1))

  # Age is missing for ~20% of rows; impute from the Title/Pclass group median
  # rather than a single global median, since both are strong Age proxies.
  df <- df %>%
    group_by(Title, Pclass) %>%
    mutate(Age = ifelse(is.na(Age), median(Age, na.rm = TRUE), Age)) %>%
    ungroup() %>%
    as.data.frame()
  df$Age[is.na(df$Age)] <- median(df$Age, na.rm = TRUE)

  embarked_mode <- names(sort(table(df$Embarked), decreasing = TRUE))[1]
  df$Embarked[is.na(df$Embarked)] <- embarked_mode

  df <- df %>%
    group_by(Pclass) %>%
    mutate(Fare = ifelse(is.na(Fare), median(Fare, na.rm = TRUE), Fare)) %>%
    ungroup() %>%
    as.data.frame()

  # Cabin is ~77% missing, too sparse to use directly; keep only a presence flag.
  df$HasCabin <- as.integer(!is.na(df$Cabin))

  df
}

main <- function() {
  df <- clean(load_raw())
  write.csv(df, PROCESSED_PATH, row.names = FALSE)
  cat(sprintf("Wrote %d cleaned rows to %s\n", nrow(df), PROCESSED_PATH))
}

main()
