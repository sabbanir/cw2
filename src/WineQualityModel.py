# ----------------------
# 1. Imports
# ----------------------
import argparse

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import os

from utility.wine_quality_lib import (
    DEFAULT_FEATURES,
    generate_synthetic_wine_data,
    quality_to_class,
    clean_data,
    make_features,
    split_and_scale,
    train_rf,
    evaluate,
    save_artifacts,
)

def plot_confusion_matrix(cm, out_dir):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Classification Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(os.path.join(out_dir, "classification_matrix.png"))
    plt.close()


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

red_wine = pd.read_csv(args.wine_data_red, sep=';')
white_wine = pd.read_csv(args.wine_data_white, sep=';')
df = pd.concat([red_wine, white_wine], axis=0).reset_index(drop=True)


print("Initial data shape:", df.shape)
print(df.head())
df.info()

df_clean = clean_data(df, z_threshold=3.0)

df_clean.info()


# 2. Clean
df_clean = clean_data(df, z_threshold=3.0)

# 3. Features/target
X, y = make_features(df_clean)

# 4. Split & scale
split = split_and_scale(X, y, seed=123)

# 5. Train
clf = train_rf(split, n_estimators=120, seed=123)

# 6. Evaluate
results = evaluate(clf, split)
cm = results["confusion_matrix"]

# 7. Save artifacts
model_path, scaler_path = save_artifacts(clf, split.scaler, out_dir=args.out_dir)

loaded_clf = joblib.load(model_path)
loaded_scaler = joblib.load(scaler_path)

# Recreate the scaled test set using loaded scaler to ensure parity
# (NOTE: split already contains scaled arrays; we ensure logic matches after reload)
# We need the original X_test to re-transform; reconstruct via indices proportionally
# In integration paths, we only validate equal lengths and non-error predictions.
preds_original = clf.predict(split.X_test_scaled)
preds_loaded = loaded_clf.predict(split.X_test_scaled)


# Confusion matrix visualization
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
cm_png = os.path.join(args.out_dir, "confusion_matrix.png")
plt.tight_layout()
plt.savefig(cm_png)
plt.close()
mlflow.log_artifact(cm_png)
