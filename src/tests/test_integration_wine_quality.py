import os
from src.utility.wine_quality_lib import (
    generate_synthetic_wine_data, clean_data, make_features,
    split_and_scale, train_rf, evaluate, save_artifacts
)

def test_end_to_end_pipeline(tmp_path):
    # 1) data
    df = generate_synthetic_wine_data(60)
    df = clean_data(df)
    # 2) features
    X, y = make_features(df)
    # 3) split & scale
    split = split_and_scale(X, y)
    # 4) train
    clf = train_rf(split, n_estimators=50)
    # 5) eval
    metrics = evaluate(clf, split)
    assert 'report' in metrics and 'confusion_matrix' in metrics
    # 6) save
    out_dir = tmp_path / 'artifacts'
    m_path, s_path = save_artifacts(clf, split.scaler, out_dir=str(out_dir))
    assert os.path.exists(m_path)
    assert os.path.exists(s_path)