"""Download the OWID COVID-19 dataset and freeze a trimmed, documented subset.

The upstream file is a live, ~95MB, 429k-row CSV covering every country and
day since 2020-01-01, updated daily. For reproducibility we don't re-fetch it
on every run: this script produces a frozen snapshot (data/raw/owid_covid_subset.csv)
that the rest of the pipeline reads, restricted to the columns, locations, and
date range this project actually uses.
"""
import pandas as pd

SOURCE_URL = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
OUTPUT_PATH = "data/raw/owid_covid_subset.csv"

COLUMNS = [
    "iso_code", "continent", "location", "date",
    "people_vaccinated_per_hundred", "people_fully_vaccinated_per_hundred",
    "new_cases_smoothed_per_million", "new_deaths_smoothed_per_million",
    "median_age", "gdp_per_capita", "population",
]

# OWID includes World Bank income-group aggregates as pseudo-location rows
# (continent is null for these, same as "World" or "Europe").
INCOME_GROUPS = [
    "Low-income countries", "Lower-middle-income countries",
    "Upper-middle-income countries", "High-income countries",
]

MIN_POPULATION = 1_000_000
DATE_START = "2020-12-01"
DATE_END = "2021-12-31"


def main():
    df = pd.read_csv(SOURCE_URL, usecols=COLUMNS)
    df["date"] = pd.to_datetime(df["date"])

    is_real_country = df["continent"].notna() & (df["population"] > MIN_POPULATION)
    is_income_group = df["location"].isin(INCOME_GROUPS)
    in_date_range = (df["date"] >= DATE_START) & (df["date"] <= DATE_END)

    subset = df[(is_real_country | is_income_group) & in_date_range].copy()
    subset["date"] = subset["date"].dt.strftime("%Y-%m-%d")
    subset.to_csv(OUTPUT_PATH, index=False)

    n_countries = subset.loc[subset["continent"].notna(), "location"].nunique()
    print(f"Wrote {len(subset)} rows ({n_countries} countries + {len(INCOME_GROUPS)} income groups) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
