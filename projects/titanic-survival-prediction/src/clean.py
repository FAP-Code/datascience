"""Load and clean the raw Titanic passenger data."""
import pandas as pd

RAW_PATH = "data/raw/titanic.csv"
PROCESSED_PATH = "data/processed/cleaned.csv"

TITLE_MAP = {
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Lady": "Rare", "Countess": "Rare", "Capt": "Rare", "Col": "Rare",
    "Don": "Rare", "Dr": "Rare", "Major": "Rare", "Rev": "Rare",
    "Sir": "Rare", "Jonkheer": "Rare", "Dona": "Rare",
    "the Countess": "Rare",  # raw Name field spells this title "the Countess"
}


def extract_title(name: str) -> str:
    title = name.split(",")[1].split(".")[0].strip()
    return TITLE_MAP.get(title, title)


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values and derive the Title/HasCabin fields cleaning depends on."""
    df = df.copy()
    df["Title"] = df["Name"].apply(extract_title)

    # Age is missing for ~20% of rows; impute from the Title/Pclass group median
    # rather than a single global median, since both are strong Age proxies.
    df["Age"] = df.groupby(["Title", "Pclass"])["Age"].transform(lambda s: s.fillna(s.median()))
    df["Age"] = df["Age"].fillna(df["Age"].median())

    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Fare"] = df.groupby("Pclass")["Fare"].transform(lambda s: s.fillna(s.median()))

    # Cabin is ~77% missing, too sparse to use directly; keep only a presence flag.
    df["HasCabin"] = df["Cabin"].notna().astype(int)

    return df


def main():
    df = clean(load_raw())
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Wrote {len(df)} cleaned rows to {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
