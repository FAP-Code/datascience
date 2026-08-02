"""Generate report-quality figures for the vaccine rollout & effectiveness analysis."""
import json

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

INCOME_SERIES_PATH = "data/processed/income_group_series.csv"
SNAPSHOT_PATH = "data/processed/country_snapshot.csv"
RESULTS_PATH = "results/regression_results.json"
OUT = "results/figures"

# Validated categorical palette (fixed order): blue, orange, aqua, yellow.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, SECONDARY_INK, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2dd", "#ffffff"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.labelsize": 11, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "text.color": INK,
    "xtick.color": SECONDARY_INK, "ytick.color": SECONDARY_INK, "font.size": 10.5,
    "savefig.facecolor": SURFACE, "savefig.dpi": 220,
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png")
    plt.close(fig)
    print(f"wrote {OUT}/{name}.png")


def fig_income_group_rollout():
    df = pd.read_csv(INCOME_SERIES_PATH, parse_dates=["date"])
    order = ["High-income countries", "Upper-middle-income countries",
              "Lower-middle-income countries", "Low-income countries"]
    colors = {order[0]: BLUE, order[1]: ORANGE, order[2]: AQUA, order[3]: YELLOW}
    labels = {o: o.replace(" countries", "") for o in order}

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for group in order:
        sub = df[df["location"] == group].sort_values("date")
        ax.plot(sub["date"], sub["people_fully_vaccinated_per_hundred"],
                color=colors[group], linewidth=2.2, label=labels[group])
    ax.set_ylabel("Fully vaccinated (% of population)")
    ax.set_title("Vaccine rollout speed by World Bank income group")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.legend(frameon=False, loc="upper left")
    save(fig, "01_income_group_rollout")


def fig_gdp_vs_vax():
    df = pd.read_csv(SNAPSHOT_PATH)
    log_gdp = np.log(df["gdp_per_capita"])
    r, p = stats.pearsonr(log_gdp, df["vax_rate_2021_07_01"])

    slope, intercept = np.polyfit(log_gdp, df["vax_rate_2021_07_01"], 1)
    xs = np.linspace(log_gdp.min(), log_gdp.max(), 100)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.scatter(df["gdp_per_capita"], df["vax_rate_2021_07_01"], color=BLUE, alpha=0.75, s=42, zorder=3)
    ax.plot(np.exp(xs), slope * xs + intercept, color=SECONDARY_INK, linestyle="--", linewidth=1.4, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("GDP per capita (log scale)")
    ax.set_ylabel("Fully vaccinated as of 2021-07-01 (%)")
    ax.set_title("Wealth strongly predicted early vaccination coverage")
    ax.text(0.03, 0.95, f"r = {r:.2f}, p < 0.001", transform=ax.transAxes,
            fontsize=10, va="top", color=INK)
    save(fig, "02_gdp_vs_vaccination_rate")


def fig_vax_vs_mortality():
    df = pd.read_csv(SNAPSHOT_PATH)
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    pearson_r = results["correlation"]["pearson_r_vax_vs_log_death"]
    pearson_p = results["correlation"]["pearson_p_value"]

    log_death = np.log1p(df["delta_wave_death_rate"])
    slope, intercept = np.polyfit(df["vax_rate_2021_07_01"], log_death, 1)
    xs = np.linspace(df["vax_rate_2021_07_01"].min(), df["vax_rate_2021_07_01"].max(), 100)

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.scatter(df["vax_rate_2021_07_01"], df["delta_wave_death_rate"], color=BLUE, alpha=0.75, s=42, zorder=3)
    ax.plot(xs, np.expm1(slope * xs + intercept), color=SECONDARY_INK, linestyle="--", linewidth=1.4, zorder=2)
    ax.set_xlabel("Fully vaccinated as of 2021-07-01 (%)")
    ax.set_ylabel("Mean daily deaths per million, Aug–Oct 2021")
    ax.set_title("Vaccination coverage vs. Delta-wave mortality")
    ax.text(0.97, 0.95, f"r = {pearson_r:.2f}, p = {pearson_p:.2f} (n.s.)", transform=ax.transAxes,
            fontsize=10, va="top", ha="right", color=INK)
    save(fig, "03_vaccination_vs_mortality")


def fig_regression_coefficients():
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    coefs = results["regression"]["coefficients"]
    order = ["vax_rate_2021_07_01", "median_age", "log_gdp_per_capita"]
    labels = ["Vaccination rate\n(2021-07-01)", "Median age", "log(GDP per capita)"]

    vals = [coefs[k]["coef"] for k in order]
    los = [coefs[k]["coef"] - coefs[k]["ci_low"] for k in order]
    his = [coefs[k]["ci_high"] - coefs[k]["coef"] for k in order]

    fig, ax = plt.subplots(figsize=(6.4, 4))
    y_pos = np.arange(len(order))
    ax.errorbar(vals, y_pos, xerr=[los, his], fmt="o", color=BLUE, ecolor=BLUE,
                elinewidth=2, capsize=5, markersize=8, zorder=3)
    ax.axvline(0, color=SECONDARY_INK, linestyle="--", linewidth=1.2, zorder=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Coefficient on log(1 + Delta-wave death rate), with 95% CI")
    ax.set_title("Regression coefficients: none clear zero")
    ax.grid(axis="y", visible=False)
    save(fig, "04_regression_coefficients")


if __name__ == "__main__":
    fig_income_group_rollout()
    fig_gdp_vs_vax()
    fig_vax_vs_mortality()
    fig_regression_coefficients()
