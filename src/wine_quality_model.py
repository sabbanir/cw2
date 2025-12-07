# ----------------------
# 1. Imports
# ----------------------
import argparse
import os
from typing import Dict

import joblib
import matplotlib.pyplot as plt
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utility.wine_quality_lib import clean_data, quality_to_class, split_and_scale,make_features,train_rf,evaluate

# ----------------------
# 2. Load dataset
# ----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--wine_data_red", type=str, required=True, help='Red Wine Dataset for training')
parser.add_argument("--wine_data_white", type=str, required=True, help='White Wine Dataset for training')
parser.add_argument("--out_dir", type=str, default="artifacts", help="Directory to save plots & outputs")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)
mlflow.autolog()

red_wine = pd.read_csv(args.wine_data_red, sep=',')
white_wine = pd.read_csv(args.wine_data_white, sep=',')
df = pd.concat([red_wine, white_wine], axis=0).reset_index(drop=True)


# red_wine = pd.read_csv("data/winequality_red_b01048312.csv", sep=";")
# white_wine = pd.read_csv("data/winequality_white_b01048312.csv", sep=";")
# df = pd.concat([red_wine, white_wine], axis=0).reset_index(drop=True)



print("Initial data shape:", df.shape)
print(df.head())
df.info()

# ----------------------
# 3. Data Cleaning & Remove outliers using z-score
# ----------------------
df = clean_data(df)
aftterclean = df.copy()
print("Data shape after cleaning:", df.shape)



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



X, y = make_features(df)




# 3) split & scale
split = split_and_scale(X, y)

# ----------------------
# 8. Train Random Forest Classifier
# ----------------------
clf = train_rf(split, n_estimators=50)
# ----------------------
# 9. Evaluate Model
# ----------------------
metrics:Dict[str, object] = evaluate(clf, split)
assert 'report' in metrics and 'confusion_matrix' in metrics

print("Classification Report:\n")

# Confusion matrix visualization
sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ----------------------
# 10. Save model and scaler
# ----------------------
joblib.dump(clf, "wine_quality_rf_classifier.pkl")
joblib.dump(StandardScaler(), "wine_quality_scaler.pkl")
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
cm = metrics['confusion_matrix']
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
