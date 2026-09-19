"""Task 02 — clean and explore the Titanic dataset.

Source: https://github.com/datasciencedojo/datasets/blob/master/titanic.csv
"""
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "titanic.csv"
OUT = ROOT / "outputs"
SOURCE = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"


def load_data() -> pd.DataFrame:
    DATA.parent.mkdir(exist_ok=True)
    if not DATA.exists():
        print("Downloading Titanic data…")
        urlretrieve(SOURCE, DATA)
    return pd.read_csv(DATA)


def main() -> None:
    sns.set_theme(style="whitegrid")
    OUT.mkdir(exist_ok=True)
    raw = load_data()
    missing_before = raw.isna().sum()
    df = raw.copy()
    df["Age"] = df["Age"].fillna(df.groupby(["Pclass", "Sex"])["Age"].transform("median"))
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode().iat[0])
    df = df.drop(columns=["Cabin"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    sns.barplot(data=df, x="Sex", y="Survived", hue="Sex", legend=False, errorbar=None, ax=axes[0], palette="Set2")
    axes[0].set(title="Survival Rate by Gender", ylabel="Survival rate", xlabel="Gender")
    sns.barplot(data=df, x="Pclass", y="Survived", hue="Pclass", legend=False, errorbar=None, ax=axes[1], palette="Set1")
    axes[1].set(title="Survival Rate by Passenger Class", ylabel="Survival rate", xlabel="Passenger class")
    fig.tight_layout()
    fig.savefig(OUT / "survival_patterns.png", dpi=180)
    plt.close(fig)

    numeric = df.select_dtypes(include="number")
    plt.figure(figsize=(9, 7))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Titanic Numeric-Variable Correlations")
    plt.tight_layout()
    plt.savefig(OUT / "correlation_heatmap.png", dpi=180)
    plt.close()

    survival_by_sex = df.groupby("Sex")["Survived"].mean()
    survival_by_class = df.groupby("Pclass")["Survived"].mean()
    (OUT / "findings.txt").write_text(
        f"Rows: {len(df)}\n"
        f"Missing values before cleaning: {int(missing_before.sum())}; after cleaning: {int(df.isna().sum().sum())}.\n"
        f"Survival rate — female: {survival_by_sex['female']:.1%}; male: {survival_by_sex['male']:.1%}.\n"
        f"Survival rate — class 1: {survival_by_class.loc[1]:.1%}; class 3: {survival_by_class.loc[3]:.1%}.\n",
        encoding="utf-8",
    )
    print(f"Task 02 complete — figures saved in {OUT}")


if __name__ == "__main__":
    main()
