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

# red_wine = pd.read_csv("data/winequality_red_b01048312.csv", sep=";")
# white_wine = pd.read_csv("data/winequality_white_b01048312.csv", sep=";")
# df = pd.concat([red_wine, white_wine], axis=0).reset_index(drop=True)


print("Initial data shape:", df.shape)
print(df.head())
df.info()

df_clean = clean_data(df, z_threshold=3.0)

df_clean.info()
