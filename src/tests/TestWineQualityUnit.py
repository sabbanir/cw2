
import os
import shutil
import unittest
import numpy as np
import pandas as pd

# Import the library under test (adjust if your package path differs)
from src.utility.wine_quality_lib import (
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

class TestWineQualityUnitpy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Small synthetic dataset for consistent tests
        cls.df_small = generate_synthetic_wine_data(n_rows=120, seed=123)
        # Ensure expected schema exists (from lib DEFAULT_FEATURES)
        for col in DEFAULT_FEATURES + ["quality"]:
            assert col in cls.df_small.columns, f"Missing expected column: {col}"

    # ---------------------------
    # generate_synthetic_wine_data
    # ---------------------------
    def test_generate_synthetic_wine_data_shapes_and_ranges(self):
        df = generate_synthetic_wine_data(n_rows=25, seed=42)
        self.assertEqual(len(df), 25)
        # Basic value sanity checks derived from the lib’s generation logic
        self.assertTrue((df["alcohol"] >= 0).all())  # clipped non-negative
        self.assertTrue(((df["pH"] >= 2.5) & (df["pH"] <= 4.5)).all())
        self.assertTrue(((df["density"] >= 0.990) & (df["density"] <= 1.010)).all())
        # quality in [3..8]
        self.assertTrue(((df["quality"] >= 3) & (df["quality"] <= 8)).all())

    # ---------------------------
    # quality_to_class
    # ---------------------------
    def test_quality_to_class_buckets(self):
        self.assertEqual(quality_to_class(3), 0)  # <=5 -> 0
        self.assertEqual(quality_to_class(5), 0)
        self.assertEqual(quality_to_class(6), 1)  # ==6 -> 1
        self.assertEqual(quality_to_class(7), 2)  # >=7 -> 2
        self.assertEqual(quality_to_class(8), 2)

    # ---------------------------
    # clean_data
    # ---------------------------
    def test_clean_data_fills_nulls_and_drops_duplicates(self):
        df = self.df_small.copy()
        # Introduce NaNs and duplicates
        df.loc[0, "alcohol"] = np.nan
        df.loc[1, "pH"] = np.nan
        df = pd.concat([df, df.iloc[[5]]], ignore_index=True)  # one duplicate

        before_rows = len(df)
        df_clean = clean_data(df, z_threshold=3.0)

        # Numeric NaNs should be filled with medians per lib behavior
        self.assertFalse(df_clean.select_dtypes(include=[np.number]).isnull().any().any())
        # Duplicate removed
        self.assertLess(len(df_clean), before_rows)
        # quality column preserved
        self.assertIn("quality", df_clean.columns)

    def test_clean_data_outlier_mask_does_not_drop_all(self):
        df_clean = clean_data(self.df_small, z_threshold=3.0)
        # Should retain a reasonable fraction of rows; not drop everything
        self.assertGreater(len(df_clean), len(self.df_small) * 0.5)

    # ---------------------------
    # make_features
    # ---------------------------
    def test_make_features_returns_X_y(self):
        df_clean = clean_data(self.df_small)
        X, y = make_features(df_clean)
        # X should exclude 'quality' and 'quality_class'
        self.assertNotIn("quality", X.columns)
        self.assertNotIn("quality_class", X.columns)
        self.assertEqual(len(X), len(y))
        # y values in {0,1,2}
        self.assertTrue(set(y.unique()).issubset({0, 1, 2}))

    # ---------------------------
    # split_and_scale
    # ---------------------------
    def test_split_and_scale_shapes_and_standardization(self):
        df_clean = clean_data(self.df_small)
        X, y = make_features(df_clean)
        split = split_and_scale(X, y, seed=42)

        # Shape consistency
        self.assertEqual(split.X_train_scaled.shape[0], split.y_train.shape[0])
        self.assertEqual(split.X_test_scaled.shape[0], split.y_test.shape[0])
        self.assertEqual(split.X_train_scaled.shape[1], split.X_test_scaled.shape[1])

        # Standardization: train mean ~ 0, std ~ 1 (tolerances for finite sample)
        train_mean = split.X_train_scaled.mean(axis=0)
        train_std = split.X_train_scaled.std(axis=0)
        self.assertTrue(np.all(np.abs(train_mean) < 1e-6))
        self.assertTrue(np.all(np.abs(train_std - 1.0) < 1e-3))

        # Stratification sanity: class distribution roughly preserved
        def proportions(arr):
            vals, counts = np.unique(arr, return_counts=True)
            total = counts.sum()
            return {int(v): c / total for v, c in zip(vals, counts)}
        p_all = proportions(y.values)
        p_train = proportions(split.y_train.values)
        for k in p_all:
            self.assertAlmostEqual(p_all[k], p_train.get(k, 0.0), delta=0.12)

    # ---------------------------
    # train_rf
    # ---------------------------
    def test_train_rf_fits_and_predicts(self):
        df_clean = clean_data(self.df_small)
        X, y = make_features(df_clean)
        split = split_and_scale(X, y, seed=7)
        clf = train_rf(split, n_estimators=60, seed=7)

        # Has predict and returns correct length
        preds = clf.predict(split.X_test_scaled)
        self.assertEqual(len(preds), len(split.y_test))

    # ---------------------------
    # evaluate
    # ---------------------------
    def test_evaluate_returns_report_and_confusion_matrix(self):
        df_clean = clean_data(self.df_small)
        X, y = make_features(df_clean)
        split = split_and_scale(X, y, seed=13)
        clf = train_rf(split, n_estimators=80, seed=13)

        results = evaluate(clf, split)
        self.assertIn("report", results)
        self.assertIn("confusion_matrix", results)

        # Report is a dict from sklearn classification_report(output_dict=True)
        self.assertIsInstance(results["report"], dict)
        # Confusion matrix is 3x3 ndarray (three classes), non-negative
        cm = results["confusion_matrix"]
        self.assertIsInstance(cm, np.ndarray)
        self.assertEqual(cm.shape, (3, 3))
        self.assertTrue((cm >= 0).all())

    # ---------------------------
    # save_artifacts
    # ---------------------------
    def test_save_artifacts_writes_model_and_scaler(self):
        out_dir = "tmp_artifacts_test"
        # Clean slate
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)

        df_clean = clean_data(self.df_small)
        X, y = make_features(df_clean)
        split = split_and_scale(X, y, seed=21)
        clf = train_rf(split, n_estimators=40, seed=21)

        model_path, scaler_path = save_artifacts(clf, split.scaler, out_dir=out_dir)
        self.assertTrue(os.path.exists(model_path))
        self.assertTrue(os.path.exists(scaler_path))

        # Cleanup
        shutil.rmtree(out_dir)

if __name__ == "__main__":
    unittest.main()
