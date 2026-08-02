"""Feature engineering on top of the cleaned Titanic data."""
import pandas as pd

CLEANED_PATH = "data/processed/cleaned.csv"
FEATURES_PATH = "data/processed/features.csv"

FEATURE_COLUMNS = [
    "Pclass", "Sex", "Age", "Fare", "Embarked",
    "FamilySize", "IsAlone", "Title", "HasCabin",
]
TARGET_COLUMN = "Survived"


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a model-ready, one-hot encoded frame with FEATURE_COLUMNS (+ target if present)."""
    df = engineer(df)
    columns = FEATURE_COLUMNS + ([TARGET_COLUMN] if TARGET_COLUMN in df else [])
    model_df = df[columns].copy()
    model_df = pd.get_dummies(model_df, columns=["Sex", "Embarked", "Title"], drop_first=True)
    return model_df


def main():
    df = pd.read_csv(CLEANED_PATH)
    model_df = build_model_frame(df)
    model_df.to_csv(FEATURES_PATH, index=False)
    n_features = model_df.shape[1] - (1 if TARGET_COLUMN in model_df else 0)
    print(f"Wrote {n_features} features for {len(model_df)} rows to {FEATURES_PATH}")


if __name__ == "__main__":
    main()
