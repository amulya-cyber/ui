import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pandas as pd
import pytest
import app


@pytest.fixture(scope="module")
def scored():
    raw = app.make_synthetic(300)
    clean, _ = app.clean_data(raw, app.map_columns(raw))
    return app.add_anomaly_scores(clean)


def test_duplicates_removed_and_result_normalised():
    raw = app.make_synthetic(100)
    clean, rep = app.clean_data(raw, app.map_columns(raw))
    assert rep["duplicates_removed"] > 0
    assert set(clean["Result"]) <= {"PASS", "FAIL"}
    assert not clean.duplicated(["Device_ID", "Test_ID"]).any()


def test_column_mapping_handles_renames():
    raw = app.make_synthetic(50).rename(columns={"Temperature_C": "Temp", "Measured_Value": "Value"})
    assert app.map_columns(raw)["Temp"] == "Temperature_C"


def test_missing_required_column_gives_clear_error():
    raw = app.make_synthetic(50).drop(columns=["Upper_Limit"])
    with pytest.raises(ValueError, match="Required columns"):
        app.map_columns(raw)


def test_yield_math(scored):
    t = app.yield_by(scored, "Test_ID")
    assert (t["yield_%"] + t["fail_rate_%"]).round(1).eq(100).all()


def test_outlier_flag_per_test(scored):
    assert scored["is_outlier"].sum() > 0


def test_no_leakage_in_features(scored):
    num, cat = app.get_feature_lists(scored)
    assert not set(num + cat) & set(app.POST_TEST_COLUMNS)


def test_prediction_output_shape(scored):
    m = app.train_models(scored)
    rec = {f: scored[f].iloc[0] for f in m["features"]}
    out = app.predict_record(m, rec)
    assert 0 <= out["fail_probability"] <= 1 and out["label"] in ("PASS", "FAIL")


def test_anomaly_score_range(scored):
    s = scored["anomaly_score"].dropna()
    assert s.between(0, 1).all()


def test_explainer_insufficient_on_weak_evidence(scored):
    weak = scored[(scored.Result == "PASS") & (scored.robust_z.abs() < 1) & (scored.margin_norm > 0.3)].iloc[0]
    assert app.explain_record(weak, scored)["status"] == "insufficient"