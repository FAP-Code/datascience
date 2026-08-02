"""Build the two analysis-ready tables from the frozen raw subset.

1. income_group_series.csv — daily vaccination rollout by World Bank income
   group, for the equity comparison.
2. country_snapshot.csv — one row per country: vaccination coverage at the
   reference date, mean death rate over the Delta-wave outcome window, and
   the covariates used to control for confounding in the regression.
"""
import pandas as pd

RAW_PATH = "data/raw/owid_covid_subset.csv"
INCOME_SERIES_PATH = "data/processed/income_group_series.csv"
SNAPSHOT_PATH = "data/processed/country_snapshot.csv"

INCOME_GROUPS = [
    "Low-income countries", "Lower-middle-income countries",
    "Upper-middle-income countries", "High-income countries",
]

# Vaccination coverage as of this date is the predictor; chosen as the point
# many countries were entering the Delta wave, so it asks "did coverage going
# into the wave predict what happened during it."
VAX_SNAPSHOT_DATE = "2021-07-01"

# Delta-wave outcome window: mean daily death rate over these three months.
OUTCOME_START = "2021-08-01"
OUTCOME_END = "2021-10-31"


def build_income_group_series(df: pd.DataFrame) -> pd.DataFrame:
    series = df[df["location"].isin(INCOME_GROUPS)][
        ["location", "date", "people_fully_vaccinated_per_hundred"]
    ].dropna()
    return series.sort_values(["location", "date"])


def build_country_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    countries = df[df["continent"].notna()]

    vax_snapshot = (
        countries[countries["date"] == VAX_SNAPSHOT_DATE]
        [["location", "iso_code", "continent", "people_fully_vaccinated_per_hundred",
          "median_age", "gdp_per_capita", "population"]]
        .dropna(subset=["people_fully_vaccinated_per_hundred", "median_age", "gdp_per_capita"])
        .rename(columns={"people_fully_vaccinated_per_hundred": "vax_rate_2021_07_01"})
    )

    outcome_window = countries[
        (countries["date"] >= OUTCOME_START) & (countries["date"] <= OUTCOME_END)
    ]
    death_outcome = (
        outcome_window.groupby("location")["new_deaths_smoothed_per_million"]
        .mean()
        .rename("delta_wave_death_rate")
        .dropna()
    )

    snapshot = vax_snapshot.merge(death_outcome, on="location", how="inner")
    return snapshot.sort_values("location").reset_index(drop=True)


def main():
    df = pd.read_csv(RAW_PATH)

    income_series = build_income_group_series(df)
    income_series.to_csv(INCOME_SERIES_PATH, index=False)
    print(f"Wrote {len(income_series)} rows to {INCOME_SERIES_PATH}")

    snapshot = build_country_snapshot(df)
    snapshot.to_csv(SNAPSHOT_PATH, index=False)
    print(f"Wrote {len(snapshot)} countries to {SNAPSHOT_PATH}")


if __name__ == "__main__":
    main()
