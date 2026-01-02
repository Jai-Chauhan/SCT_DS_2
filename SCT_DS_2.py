import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

ZIP_PATH = "C:/Users/cjani/Downloads/titanic (1).zip"


def load_all_csv_from_zip(zip_path):
    """
    Loads ALL CSV files inside a ZIP archive into a dictionary of DataFrames.
    Key = file name
    Value = pandas DataFrame
    """
    dataframes = {}

    with zipfile.ZipFile(zip_path, "r") as z:
        for file in z.namelist():
            if file.lower().endswith(".csv"):
                with z.open(file) as f:
                    df = pd.read_csv(f)
                    dataframes[os.path.basename(file)] = df

    return dataframes


# -------- LOAD DATA --------
datasets = load_all_csv_from_zip(ZIP_PATH)

# Example: use the FIRST dataset (change if needed)
df = list(datasets.values())[0]

print("\n===== DATA PREVIEW =====\n")
print(df.head())

print("\n===== DATA INFO =====\n")
print(df.info())

print("\n===== MISSING VALUES =====\n")
print(df.isnull().sum())


# -------- BASIC DATA CLEANING --------

# Handle missing numeric values
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].median())

# Handle missing categorical values
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Remove duplicates
df = df.drop_duplicates()

print("\n===== AFTER CLEANING — MISSING VALUES =====\n")
print(df.isnull().sum())


# -------- BASIC EDA --------

print("\n===== NUMERIC SUMMARY =====\n")
print(df.describe())

print("\n===== CATEGORICAL SUMMARY =====\n")
for col in df.select_dtypes(include=['object']).columns:
    print(f"\nValue counts for {col}:\n", df[col].value_counts())


# -------- DISTRIBUTIONS --------
numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    plt.figure(figsize=(7,4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()


# -------- CORRELATION ANALYSIS --------
if len(numeric_cols) > 1:
    plt.figure(figsize=(8,6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()


# -------- RELATIONSHIP ANALYSIS --------
# Example relationships (auto-detects possible useful ones)
if "Survived" in df.columns:
    target = "Survived"
elif "Outcome" in df.columns:
    target = "Outcome"
else:
    target = None


if target:
    print(f"\n===== Relationship with {target} =====\n")

    # Categorical vs Target
    for col in df.select_dtypes(include=['object']).columns:
        plt.figure(figsize=(7,4))
        sns.countplot(data=df, x=col, hue=target)
        plt.title(f"{target} by {col}")
        plt.xticks(rotation=45)
        plt.show()

    # Numeric vs Target
    for col in numeric_cols:
        plt.figure(figsize=(7,4))
        sns.boxplot(data=df, x=target, y=col)
        plt.title(f"{col} vs {target}")
        plt.show()


print("\n===== TREND & PATTERN INSIGHTS =====\n")
print("• Checked missing values & handled them")
print("• Generated numeric & categorical summaries")
print("• Visualized distributions")
print("• Correlation matrix shows strongest relationships")
print("• Relationship plots reveal how features vary across target")
