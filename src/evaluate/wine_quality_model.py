# ----------------------
# 1. Imports
# ----------------------
import argparse

import os
import mlflow
import joblib
import matplotlib.pyplot as plt
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utility.wine_quality_lib import (quality_to_class,clean_data)

# ----------------------
# 2. Load dataset
# ----------------------


parser = argparse.ArgumentParser()
parser.add_argument("--wine_data_red", type=str, required=True, help='Red Wine Dataset for training')
parser.add_argument("--wine_data_white", type=str, required=True, help='White Wine Dataset for training')
parser.add_argument("--out_dir", type=str, default="artifacts", help="Directory to save plots & outputs")
args = parser.parse_args()
mlflow.autolog()
# %% [markdown]
# ## First load the data
# The first thing we need to do is load the data we're going to work with and have a quick look at a summary of it.
# Pandas gives us a function to read CSV files.
# **You might need update the location of the dataset to point to the correct place you saved it to!**
# "../" means "back one directory from where we are now"
# "./" means "from where we are now"

# %%

red_wine = pd.read_csv(args.wine_data_red, sep=',')
white_wine = pd.read_csv(args.wine_data_white, sep=',')
df = pd.concat([red_wine, white_wine], axis=0).reset_index(drop=True)


# red_wine = pd.read_csv("winequality_red_b01048312.csv", sep=";")
# white_wine = pd.read_csv("winequality_white_b01048312.csv", sep=";")
# df = pd.concat([red_wine, white_wine], axis=0).reset_index(drop=True)


# df = pd.read_csv(args.wine_data, sep=',')

# df = pd.read_csv(url_red, sep=',')
print("Initial data shape:", df.shape)
print(df.head())
df.info()

# ----------------------
# 3. Data Cleaning
# ----------------------

df = clean_data(df)

aftterclean = df.copy()

# 3.3 Remove outliers using z-score
print("Data shape after cleaning:", df.shape)
os.makedirs(args.out_dir, exist_ok=True)
# ----------------------
# 4. Feature Analysis / Correlation
# ----------------------
# Check correlation matrix
corr_matrix = df.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Identify highly correlated features (>0.85)
high_corr = [col for col in upper_tri.columns if any(upper_tri[col] > 0.85)]
print("Highly correlated features to drop:", high_corr)
df.drop(columns=high_corr, inplace=True)  # optional



df['quality_class'] = df['quality'].apply(quality_to_class)
X = df.drop(['quality', 'quality_class'], axis=1)
y = df['quality_class']

print("Class distribution:\n", y.value_counts())

# ----------------------
# 6. Train/Test Split
# ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------
# 7. Feature Scaling
# ----------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------
# 8. Train Random Forest Classifier
# ----------------------
clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
clf.fit(X_train_scaled, y_train)

# ----------------------
# 9. Evaluate Model
# ----------------------
y_pred = clf.predict(X_test_scaled)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion matrix visualization
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ----------------------
# 10. Save model and scaler
# ----------------------
joblib.dump(clf, "wine_quality_rf_classifier.pkl")
joblib.dump(scaler, "wine_quality_scaler.pkl")
print("Model and scaler saved successfully!")

# Calculate metrics
total_records = len(aftterclean)
null_records = aftterclean.isnull().any(axis=1).sum()
duplicate_records = aftterclean.duplicated().sum()
non_null_records = total_records - null_records

# Prepare data for visualization
data_summary = pd.DataFrame({
    "Category": ["Total Records", "Non-Null Records", "Null Records", "Duplicate Records"],
    "Count": [total_records, non_null_records, null_records, duplicate_records]
})

# ---------------------------
# Visualization
# ---------------------------
plt.figure(figsize=(8, 6))
ax = sns.barplot(x="Category", y="Count", data=data_summary, palette="coolwarm")

# Add count labels on each bar
for i, row in enumerate(data_summary.itertuples()):
    ax.text(i, row.Count + total_records * 0.01, f"{row.Count:,}",
            ha='center', va='bottom', fontsize=11, weight='bold', color='black')

# Beautify the chart
plt.title("Data Quality Overview: Null, Duplicate, and Non-Null Records", fontsize=14, weight="bold")
plt.ylabel("Number of Records")
plt.xlabel("")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# Optional: Print detailed summary
print("Data Quality Summary:")
print(data_summary)


# ----------------------
# 10. Plots -> PNG files
# ----------------------
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
cm_png = os.path.join(args.out_dir, "confusion_matrix.png")
plt.tight_layout()
plt.savefig(cm_png)
plt.close()
mlflow.log_artifact(cm_png)



# Feature Importance
fi = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8, 5))
sns.barplot(x=fi.values[:12], y=fi.index[:12], palette="viridis")
plt.title("Top 12 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
fi_png = os.path.join(args.out_dir, "feature_importance.png")
plt.tight_layout()
plt.savefig(fi_png)
plt.close()
mlflow.log_artifact(fi_png)


# ---------------------------
# Visualization
# ---------------------------
plt.figure(figsize=(8, 6))
ax = sns.barplot(x="Category", y="Count", data=data_summary, palette="coolwarm")

# Add count labels on each bar
for i, row in enumerate(data_summary.itertuples()):
    ax.text(i, row.Count + total_records * 0.01, f"{row.Count:,}",
            ha='center', va='bottom', fontsize=11, weight='bold', color='black')

# Beautify the chart
plt.title("Data Quality Overview: Null, Duplicate, and Non-Null Records", fontsize=14, weight="bold")
plt.ylabel("Number of Records")
plt.xlabel("")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# Optional: Print detailed summary
print("Data Quality Summary:")
print(data_summary)
