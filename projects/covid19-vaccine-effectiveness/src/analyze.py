"""Correlation and regression analysis of vaccination coverage vs. Delta-wave mortality.

Model: log1p(delta_wave_death_rate) ~ vax_rate_2021_07_01 + median_age + log(gdp_per_capita)

The outcome and GDP per capita are both right-skewed (skew 1.56 and 1.89
respectively), so both are log-transformed — standard practice for this kind
of economic/epidemiological data. median_age and log(gdp_per_capita) are
included as covariates because both plausibly influence a country's mortality
independently of vaccination coverage (older populations face higher
COVID mortality risk; richer countries generally have stronger healthcare
systems), and both are also correlated with how early a country accessed
vaccines — so leaving them out would risk attributing their effect to
vaccination coverage.
"""
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

SNAPSHOT_PATH = "data/processed/country_snapshot.csv"
RESULTS_PATH = "results/regression_results.json"


def main():
    df = pd.read_csv(SNAPSHOT_PATH)

    df["log_death_rate"] = np.log1p(df["delta_wave_death_rate"])
    df["log_gdp_per_capita"] = np.log(df["gdp_per_capita"])

    pearson_r, pearson_p = stats.pearsonr(df["vax_rate_2021_07_01"], df["log_death_rate"])
    spearman_r, spearman_p = stats.spearmanr(df["vax_rate_2021_07_01"], df["delta_wave_death_rate"])

    X = sm.add_constant(df[["vax_rate_2021_07_01", "median_age", "log_gdp_per_capita"]])
    y = df["log_death_rate"]
    model = sm.OLS(y, X).fit()

    coefs = {}
    for name in X.columns:
        coefs[name] = {
            "coef": float(model.params[name]),
            "std_err": float(model.bse[name]),
            "p_value": float(model.pvalues[name]),
            "ci_low": float(model.conf_int().loc[name, 0]),
            "ci_high": float(model.conf_int().loc[name, 1]),
        }

    output = {
        "n_countries": len(df),
        "vax_snapshot_date": "2021-07-01",
        "outcome_window": "2021-08-01 to 2021-10-31",
        "correlation": {
            "pearson_r_vax_vs_log_death": pearson_r,
            "pearson_p_value": pearson_p,
            "spearman_r_vax_vs_death": spearman_r,
            "spearman_p_value": spearman_p,
        },
        "regression": {
            "formula": "log1p(delta_wave_death_rate) ~ vax_rate_2021_07_01 + median_age + log(gdp_per_capita)",
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
            "f_statistic": float(model.fvalue),
            "f_p_value": float(model.f_pvalue),
            "coefficients": coefs,
        },
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Pearson r (vax rate vs. log death rate): {pearson_r:.3f} (p={pearson_p:.4f})")
    print(f"Spearman rho (vax rate vs. death rate):   {spearman_r:.3f} (p={spearman_p:.4f})")
    print()
    print(model.summary())
    print(f"\nWrote regression results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
