
'''Reusable wine-quality ML utilities (no side effects, no plotting).
This is extracted to enable unit/integration/smoke testing.
'''

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# --- Data generation for tests -------------------------------------------------
DEFAULT_FEATURES = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]

def generate_synthetic_wine_data(n_rows: int = 50, seed: int = 42) -> pd.DataFrame:
    """Create a tiny synthetic dataset with the same schema as the UCI wine-quality data.
    The values are random but shaped to look plausible; good enough for smoke/integration tests.
    """
    rng = np.random.default_rng(seed)
    data = {col: rng.normal(loc=0.0, scale=1.0, size=n_rows) for col in DEFAULT_FEATURES}
    # Make alcohol positive and pH in ~[2.5,4.0]
    data['alcohol'] = np.clip(rng.normal(10.5, 1.2, n_rows), 0.0, None)
    data['pH'] = np.clip(rng.normal(3.2, 0.15, n_rows), 2.5, 4.5)
    # Density around 0.996
    data['density'] = np.clip(rng.normal(0.996, 0.001, n_rows), 0.990, 1.010)
    # Target: quality (3-8)
    quality = np.clip(rng.integers(3, 9, size=n_rows), 3, 8)
    df = pd.DataFrame(data)
    df['quality'] = quality
    return df

# --- Preprocessing & modeling --------------------------------------------------

def quality_to_class(q: int) -> int:
    """Map numeric wine quality to three classes as in the user script.
    Low (<=5) -> 0, Medium (6) -> 1, High (>=7) -> 2
    """
    if q <= 5:
        return 0
    elif q == 6:
        return 1
    else:
        return 2

def clean_data(df: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
    """Fill missing values, drop duplicates, remove numeric outliers via z-score.
    Keeps the 'quality' column intact.
    """
    df = df.copy()
    # Fill NaNs with column medians (numeric only)
    df_numeric = df.select_dtypes(include=[np.number])
    df[df_numeric.columns] = df_numeric.fillna(df_numeric.median())
    # Drop duplicates
    df = df.drop_duplicates()
    # Remove outliers using z-score on numeric cols
    df_numeric = df.select_dtypes(include=[np.number])
    if not df_numeric.empty:
        z_scores = np.abs((df_numeric - df_numeric.mean()) / df_numeric.std(ddof=0))
        mask = (z_scores < z_threshold).all(axis=1)
        df = df.loc[mask]
    return df

def make_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Create target column `quality_class` and return (X, y)."""
    df = df.copy()
    df['quality_class'] = df['quality'].apply(quality_to_class)
    X = df.drop(['quality', 'quality_class'], axis=1)
    y = df['quality_class']
    return X, y

@dataclass
class Split:
    X_train_scaled: np.ndarray
    X_test_scaled: np.ndarray
    y_train: pd.Series
    y_test: pd.Series
    scaler: StandardScaler

def split_and_scale(X: pd.DataFrame, y: pd.Series, seed: int = 42) -> Split:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return Split(X_train_scaled, X_test_scaled, y_train, y_test, scaler)

def train_rf(split: Split, n_estimators: int = 100, seed: int = 42) -> RandomForestClassifier:
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, class_weight="balanced")
    clf.fit(split.X_train_scaled, split.y_train)
    return clf

def evaluate(clf: RandomForestClassifier, split: Split) -> Dict[str, object]:
    y_pred = clf.predict(split.X_test_scaled)
    report = classification_report(split.y_test, y_pred, output_dict=True)
    cm = confusion_matrix(split.y_test, y_pred)
    return {"report": report, "confusion_matrix": cm}

def save_artifacts(clf: RandomForestClassifier, scaler: StandardScaler, out_dir: str = "artifacts") -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    import joblib
    model_path = os.path.join(out_dir, "wine_quality_rf_classifier.pkl")
    scaler_path = os.path.join(out_dir, "wine_quality_scaler.pkl")
    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)
    return model_path, scaler_path