from src.utility.wine_quality_lib import generate_synthetic_wine_data, clean_data, make_features, split_and_scale, train_rf

def test_smoke_tiny_run():
    df = generate_synthetic_wine_data(25)
    df = clean_data(df)
    X, y = make_features(df)
    split = split_and_scale(X, y)
    clf = train_rf(split, n_estimators=10)
    # Minimal assertion that model trained and predicts without error
    preds = clf.predict(split.X_test_scaled)
    assert len(preds) == len(split.y_test)