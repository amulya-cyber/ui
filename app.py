"""
ATE Test Intelligence - single-file Streamlit app.

Run:  streamlit run app.py

Pipeline: upload CSV -> map columns (schema-driven) -> clean -> yield analysis
-> anomaly detection -> failure investigation + evidence-based explanation
-> ML failure prediction (trained before-the-test features only).

Nothing in here depends on specific test names, lot names or row counts.
"""
import glob
import io
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, StratifiedKFold,
                                     cross_val_predict, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# 1. SCHEMA LAYER: map whatever column names arrive onto canonical names
# ---------------------------------------------------------------------------
SYNONYMS = {
    "Device_ID": ["deviceid", "device", "dutid", "dut", "unitid", "partid"],
    "Test_ID": ["testid", "testnumber", "testno", "testcode"],
    "Test_Name": ["testname", "test", "testdescription"],
    "Lot_ID": ["lotid", "lot", "lotnumber"],
    "Wafer_ID": ["waferid", "wafer", "wafernumber"],
    "VDD_V": ["vddv", "vdd", "supplyvoltage", "voltage", "vcc", "vddvolts"],
    "Temperature_C": ["temperaturec", "temperature", "temp", "tempc"],
    "Measured_Value": ["measuredvalue", "measurement", "measured", "value", "reading"],
    "Lower_Limit": ["lowerlimit", "lowlimit", "lsl", "lolimit", "minlimit", "lower"],
    "Upper_Limit": ["upperlimit", "highlimit", "usl", "hilimit", "maxlimit", "upper"],
    "Result": ["result", "passfail", "status", "outcome"],
    "Failure_Mode": ["failuremode", "failmode", "failurecategory"],
    "Retest_Count": ["retestcount", "retests", "retest"],
    "Yield_Impact": ["yieldimpact", "yieldloss"],
}
REQUIRED = ["Test_ID", "Measured_Value", "Lower_Limit", "Upper_Limit", "Result"]
NUMERIC = ["VDD_V", "Temperature_C", "Measured_Value", "Lower_Limit", "Upper_Limit",
           "Retest_Count", "Yield_Impact"]
# Columns only known AFTER the test runs -> must never be used to predict the result.
POST_TEST_COLUMNS = ["Measured_Value", "Failure_Mode", "Retest_Count", "Yield_Impact"]


def _norm(name):
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def map_columns(df):
    """Return {original_column: canonical_name}. Raises ValueError if required ones are missing."""
    mapping, used = {}, set()
    normalized = {c: _norm(c) for c in df.columns}
    for canon, syns in SYNONYMS.items():
        candidates = [_norm(canon)] + syns
        for cand in candidates:
            hit = next((c for c, n in normalized.items() if n == cand and c not in used), None)
            if hit is not None:
                mapping[hit] = canon
                used.add(hit)
                break
    found = set(mapping.values())
    # If Test_ID is absent but Test_Name exists, use the name as the test identity.
    if "Test_ID" not in found and "Test_Name" in found:
        src = next(c for c, v in mapping.items() if v == "Test_Name")
        mapping[src] = "Test_ID"
        found = set(mapping.values())
    missing = [c for c in REQUIRED if c not in found]
    if missing:
        raise ValueError(
            f"Required columns not found: {missing}. Columns in file: {list(df.columns)}")
    return mapping


# ---------------------------------------------------------------------------
# 2. CLEANING + DATA QUALITY (Task 1)
# ---------------------------------------------------------------------------
def robust_z(s):
    """Median/MAD z-score; falls back to std if MAD is 0."""
    med = s.median()
    scale = 1.4826 * (s - med).abs().median()
    if not scale or np.isnan(scale):
        scale = s.std()
    if not scale or np.isnan(scale):
        scale = 1.0
    return (s - med) / scale


def clean_data(raw, mapping):
    """Returns (clean_df, report). Every decision is recorded in report['decisions']."""
    log = []
    df = raw.rename(columns=mapping)[list(mapping.values())].copy()
    report = {"rows_raw": len(raw), "cols_raw": raw.shape[1]}

    # Profile (before cleaning)
    report["missing_before"] = df.isna().sum().to_dict()
    report["exact_duplicates"] = int(df.duplicated().sum())

    # Text normalisation
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype("string").str.strip()
    for c in NUMERIC:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Device_ID" not in df:
        df["Device_ID"] = "ROW_" + pd.Series(range(len(df))).astype(str)
        log.append("No Device_ID column: used row number as device id.")

    # Result: normalise, never impute
    r = df["Result"].astype("string").str.upper().str.strip()
    r = r.replace({"P": "PASS", "OK": "PASS", "GOOD": "PASS",
                   "F": "FAIL", "NG": "FAIL", "BAD": "FAIL"})
    df["Result"] = r.where(r.isin(["PASS", "FAIL"]))
    bad_result = int(df["Result"].isna().sum())
    df = df[df["Result"].notna()]
    report["rows_dropped_bad_result"] = bad_result
    log.append(f"Dropped {bad_result} rows with missing/unrecognised Result "
               "(the label is never imputed).")

    # Duplicates
    n0 = len(df)
    df = df.drop_duplicates()
    log.append(f"Removed {n0 - len(df)} exact duplicate rows.")
    keys = [k for k in ["Device_ID", "Test_ID"] if k in df]
    n1 = len(df)
    df = (df.assign(_filled=df.notna().sum(axis=1))
            .sort_values(["_filled"] + (["Retest_Count"] if "Retest_Count" in df else []))
            .drop_duplicates(subset=keys, keep="last").drop(columns="_filled")
            .sort_index())
    log.append(f"Removed {n1 - len(df)} repeated Device_ID+Test_ID records "
               "(kept the most complete / highest retest record).")
    report["duplicates_removed"] = n0 - len(df)

    # Missing values
    df["Measured_Value_missing"] = df["Measured_Value"].isna()
    for c in ["Lower_Limit", "Upper_Limit", "VDD_V", "Temperature_C"]:
        if c in df and df[c].isna().any():
            n = int(df[c].isna().sum())
            df[c] = df[c].fillna(df.groupby("Test_ID")[c].transform("median"))
            df[c] = df[c].fillna(df[c].median())
            log.append(f"{c}: filled {n} missing values with the per-test median (global median fallback).")
    for c in ["Retest_Count", "Yield_Impact"]:
        if c in df and df[c].isna().any():
            n = int(df[c].isna().sum())
            df[c] = df[c].fillna(0)
            log.append(f"{c}: filled {n} missing values with 0.")
    for c in ["Lot_ID", "Wafer_ID", "Test_Name"]:
        if c in df and df[c].isna().any():
            n = int(df[c].isna().sum())
            df[c] = df[c].fillna("UNKNOWN")
            log.append(f"{c}: filled {n} missing values with 'UNKNOWN'.")
    if "Failure_Mode" in df:
        n = int(df["Failure_Mode"].isna().sum())
        df["Failure_Mode"] = df["Failure_Mode"].fillna(
            pd.Series(np.where(df["Result"] == "PASS", "NONE", "UNKNOWN"), index=df.index))
        log.append(f"Failure_Mode: filled {n} missing (NONE for PASS, UNKNOWN for FAIL).")
    n_mv = int(df["Measured_Value_missing"].sum())
    log.append(f"Measured_Value: {n_mv} missing values left as NaN (not imputed: "
               "inventing a measurement would corrupt the analysis). They are excluded from anomaly scoring.")

    # Outliers: per-test robust z-score
    df["robust_z"] = df.groupby("Test_ID")["Measured_Value"].transform(robust_z)
    df["is_outlier"] = df["robust_z"].abs() > 3.5
    report["outliers"] = int(df["is_outlier"].sum())
    log.append("Outliers: flagged where |robust z| > 3.5 computed per Test_ID (median/MAD, so "
               "outliers do not distort their own threshold). Flagged, not deleted.")

    df["Out_Of_Spec"] = (df["Measured_Value"] < df["Lower_Limit"]) | (df["Measured_Value"] > df["Upper_Limit"])
    report["decisions"] = log
    report["rows_clean"] = len(df)
    return df.reset_index(drop=True), report


def yield_by(df, col):
    g = df.groupby(col)["Result"].agg(total="count", fails=lambda s: int((s == "FAIL").sum()))
    g["yield_%"] = (100 * (1 - g["fails"] / g["total"])).round(2)
    g["fail_rate_%"] = (100 - g["yield_%"]).round(2)
    return g.sort_values("fail_rate_%", ascending=False)


# ---------------------------------------------------------------------------
# 3. ANOMALY DETECTION (Task 3)
# ---------------------------------------------------------------------------
def add_anomaly_scores(df, contamination=0.03, seed=42):
    """Combine: IsolationForest rank, per-test robust z, and closeness to spec limits."""
    d = df.copy()
    span = (d["Upper_Limit"] - d["Lower_Limit"]).replace(0, np.nan)
    d["margin_norm"] = (np.minimum(d["Measured_Value"] - d["Lower_Limit"],
                                   d["Upper_Limit"] - d["Measured_Value"]) / span)
    feats = pd.DataFrame({"rz": d["robust_z"].clip(-20, 20),
                          "pos": ((d["Measured_Value"] - d["Lower_Limit"]) / span).clip(-3, 4)})
    for c in ["VDD_V", "Temperature_C"]:
        if c in d:
            feats[c] = d[c]
    feats = feats.fillna(feats.median()).fillna(0)
    X = StandardScaler().fit_transform(feats)
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed)
    iso.fit(X)
    raw = -iso.score_samples(X)
    d["iso_score"] = pd.Series(raw, index=d.index).rank(pct=True)
    zc = (d["robust_z"].abs() / 6).clip(0, 1).fillna(0)
    closeness = (1 - (d["margin_norm"] / 0.25).clip(0, 1)).fillna(0)  # 1 = at/over a limit
    d["anomaly_score"] = (0.40 * d["iso_score"] + 0.35 * zc + 0.25 * closeness).round(4)
    d.loc[d["Measured_Value"].isna(), "anomaly_score"] = np.nan
    return d


def explain_record(row, df, min_group=20):
    """Evidence-based explanation. Uses only numbers present in the data."""
    overall_fail = (df["Result"] == "FAIL").mean()
    evidence, signals, causes = [], 0, []
    x, lo, hi = row["Measured_Value"], row["Lower_Limit"], row["Upper_Limit"]
    if pd.isna(x):
        return {"status": "insufficient", "evidence": ["Measured_Value is missing for this record."],
                "causes": [], "recommendation": "Re-run the measurement; nothing can be concluded.",
                "confidence": "None"}
    span = hi - lo
    dist = min(x - lo, hi - x) / span if span else np.nan
    out_spec = x < lo or x > hi
    evidence.append(f"Measured {x:.4g} against limits [{lo:.4g}, {hi:.4g}]; recorded Result = {row['Result']}.")
    if out_spec:
        side = "below the lower" if x < lo else "above the upper"
        evidence.append(f"Value is {side} limit by {abs(x - (lo if x < lo else hi)):.4g}.")
        signals += 1
    elif not np.isnan(dist):
        evidence.append(f"Value is inside spec with {100 * dist:.1f}% of the spec window to the nearest limit.")
        if dist < 0.10:
            signals += 1
    tdf = df[df["Test_ID"] == row["Test_ID"]]
    rz = row["robust_z"]
    evidence.append(f"Versus {len(tdf)} records of test {row['Test_ID']}: robust z = {rz:.2f} "
                    f"(test median {tdf['Measured_Value'].median():.4g}).")
    z_hit = abs(rz) > 3
    signals += int(z_hit)
    for col, label in [("Temperature_C", "temperature"), ("VDD_V", "supply voltage")]:
        if col in df and not pd.isna(row.get(col)) and len(tdf) > 5:
            z = robust_z(tdf[col]).loc[row.name] if row.name in tdf.index else 0
            if abs(z) > 2:
                evidence.append(f"{label.capitalize()} {row[col]:.4g} is unusual for this test (z = {z:.2f}).")
                signals += 1
                if out_spec or z_hit:
                    causes.append(f"Possible {label} sensitivity (unverified): the abnormal reading coincides "
                                  f"with an atypical {label}.")
    for col, label in [("Lot_ID", "lot"), ("Wafer_ID", "wafer")]:
        if col in df and row[col] != "UNKNOWN":
            g = df[df[col] == row[col]]
            if len(g) >= min_group:
                fr = (g["Result"] == "FAIL").mean()
                if fr >= 1.5 * overall_fail and fr - overall_fail >= 0.03:
                    evidence.append(f"{label.capitalize()} {row[col]} fail rate is {100 * fr:.1f}% vs "
                                    f"{100 * overall_fail:.1f}% overall (n={len(g)}).")
                    signals += 1
                    causes.append(f"Possible {label}-level process issue (unverified): elevated fail rate in "
                                  f"{label} {row[col]}.")
    if "Retest_Count" in df and row.get("Retest_Count", 0) > 0:
        evidence.append(f"Device was retested {int(row['Retest_Count'])} time(s).")
        signals += 1
    if "Failure_Mode" in df and row["Result"] == "FAIL":
        evidence.append(f"Recorded failure mode: {row['Failure_Mode']}.")
    if row["Result"] == "PASS" and (z_hit or (not np.isnan(dist) and dist < 0.10)):
        causes.append("Record passes but is statistically unusual or close to a limit: possible marginal "
                      "device / guard-band risk (unverified).")

    if signals == 0 or (signals == 1 and not out_spec and not z_hit):
        return {"status": "insufficient", "evidence": evidence, "causes": [],
                "recommendation": "No action indicated from this data.", "confidence": "Insufficient evidence"}
    if not causes:
        causes.append("The data shows a deviation but not enough independent evidence to attribute a cause.")
    conf = "High" if signals >= 4 else "Medium" if signals >= 2 else "Low"
    rec = ("Review lot/wafer and test conditions listed above; re-test the device at nominal conditions."
           if out_spec else "Monitor this device/test; consider tighter guard-bands if the pattern repeats.")
    return {"status": "ok", "evidence": evidence, "causes": causes, "recommendation": rec, "confidence": conf}


# ---------------------------------------------------------------------------
# 4. ML FAILURE PREDICTION (Task 2)
# ---------------------------------------------------------------------------
def get_feature_lists(df):
    """Pre-test features only. Post-test columns are excluded to avoid leakage."""
    num = [c for c in ["VDD_V", "Temperature_C", "Lower_Limit", "Upper_Limit"]
           if c in df and df[c].notna().any()]
    cat = [c for c in ["Test_ID", "Lot_ID", "Wafer_ID"] if c in df]
    return num, cat


def build_preprocessor(num, cat):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), cat),
    ])


def train_models(df, seed=42):
    num, cat = get_feature_lists(df)
    feats = num + cat
    if not feats:
        raise ValueError("No usable pre-test features found.")
    X, y = df[feats].copy(), (df["Result"] == "FAIL").astype(int)
    n_fail = int(y.sum())
    if n_fail < 20 or n_fail > len(y) - 20:
        raise ValueError(f"Too few examples of one class (FAIL={n_fail}, PASS={len(y) - n_fail}); "
                         "cannot train a reliable classifier.")
    groups = df["Device_ID"] if "Device_ID" in df and df["Device_ID"].nunique() > 10 else None
    split_note = "stratified random split"
    tr, te = None, None
    if groups is not None:
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(X, y, groups))
        if y.iloc[tr].nunique() == 2 and y.iloc[te].nunique() == 2:
            split_note = "group split by Device_ID (a device never appears in both train and test)"
        else:
            tr = None
    if tr is None:
        groups = None
        tr, te = train_test_split(np.arange(len(X)), test_size=0.25, stratify=y, random_state=seed)
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=3, n_jobs=-1,
                                                class_weight="balanced_subsample", random_state=seed),
    }
    cv = GroupKFold(n_splits=3) if groups is not None else StratifiedKFold(3, shuffle=True, random_state=seed)
    rows, fitted = [], {}
    for name, clf in models.items():
        pipe = Pipeline([("pre", build_preprocessor(num, cat)), ("clf", clf)])
        # Threshold chosen on out-of-fold TRAIN predictions only (no test-set leakage).
        oof = cross_val_predict(pipe, Xtr, ytr, cv=cv, method="predict_proba",
                                groups=groups.iloc[tr] if groups is not None else None)[:, 1]
        p, r, t = precision_recall_curve(ytr, oof)
        f1s = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-9)
        thr = float(t[int(np.argmax(f1s))]) if len(t) else 0.5
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        pred = (proba >= thr).astype(int)
        rows.append({"Model": name, "Precision": precision_score(yte, pred, zero_division=0),
                     "Recall": recall_score(yte, pred, zero_division=0),
                     "F1": f1_score(yte, pred, zero_division=0),
                     "ROC-AUC": roc_auc_score(yte, proba), "PR-AUC": average_precision_score(yte, proba),
                     "Threshold": thr, "Accuracy": float((pred == yte).mean())})
        fitted[name] = (pipe, thr, confusion_matrix(yte, pred))
    res = pd.DataFrame(rows).set_index("Model")
    best = res["PR-AUC"].idxmax()
    pipe, thr, cm = fitted[best]
    sample = Xte.sample(min(2000, len(Xte)), random_state=seed)
    imp = permutation_importance(pipe, sample, yte.loc[sample.index], scoring="average_precision",
                                 n_repeats=3, random_state=seed, n_jobs=-1)
    importance = pd.Series(imp.importances_mean, index=feats).sort_values(ascending=False)
    return {"results": res, "best": best, "pipeline": pipe, "threshold": thr, "confusion": cm,
            "importance": importance, "features": feats, "num": num, "cat": cat,
            "split": split_note, "fail_ratio": float(y.mean()),
            "baseline_accuracy": float(1 - y.mean()), "X_test": Xte, "y_test": yte}


def predict_record(model, record):
    """Reusable prediction function. record: dict of pre-test features."""
    missing = [f for f in model["features"] if f not in record]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    row = pd.DataFrame([{f: record[f] for f in model["features"]}])
    p = float(model["pipeline"].predict_proba(row)[0, 1])
    return {"fail_probability": p, "label": "FAIL" if p >= model["threshold"] else "PASS",
            "threshold": model["threshold"]}


# ---------------------------------------------------------------------------
# 5. PRACTICE DATA GENERATOR (so you can test before the real CSV arrives)
# ---------------------------------------------------------------------------
def make_synthetic(n_devices=1500, seed=7):
    rng = np.random.default_rng(seed)
    tests = [("P301", "SCAN_CHAIN", 800, 1600, 1200, 110, "TIMING_VIOLATION"),
             ("P302", "IDDQ_LEAK", 0.5, 5.0, 2.2, 0.9, "LEAKAGE_HIGH"),
             ("P303", "VOUT_REG", 1.05, 1.15, 1.10, 0.020, "VOLTAGE_OUT_OF_RANGE"),
             ("P304", "FREQ_MAX", 450, 550, 500, 20, "FREQ_LOW"),
             ("P305", "RISE_TIME", 1.0, 3.0, 2.0, 0.40, "SLEW_FAIL"),
             ("P306", "IO_DRIVE", 8, 16, 12, 1.6, "DRIVE_WEAK"),
             ("P307", "POWER_MW", 20, 60, 40, 8, "POWER_HIGH"),
             ("P308", "OFFSET_MV", -5, 5, 0, 2.0, "OFFSET_FAIL")]
    rows = []
    for d in range(n_devices):
        lot = f"LOT{rng.integers(1, 6)}"
        wafer = f"{lot}_W{rng.integers(1, 6):02d}"
        vdd = round(float(rng.choice([0.95, 1.0, 1.1, 1.2])), 2)
        temp = round(float(rng.normal(60, 25)), 1)
        for tid, name, lo, hi, mu, sd, fm in tests:
            shift = sd * 1.6 * (temp - 25) / 60 + (sd * 1.4 if (lot == "LOT3" and tid == "P302") else 0)
            val = rng.normal(mu + shift, sd)
            if rng.random() < 0.01:
                val = mu + sd * rng.choice([-1, 1]) * rng.uniform(8, 15)
            fail = val < lo or val > hi
            rows.append({"Device_ID": f"ATE{300001 + d}", "Test_ID": tid, "Test_Name": name,
                         "Lot_ID": lot, "Wafer_ID": wafer, "VDD_V": vdd, "Temperature_C": temp,
                         "Measured_Value": round(float(val), 4), "Lower_Limit": lo, "Upper_Limit": hi,
                         "Result": "FAIL" if fail else "PASS", "Failure_Mode": fm if fail else "NONE",
                         "Retest_Count": int(rng.integers(0, 3)) if fail else 0,
                         "Yield_Impact": round(float(rng.uniform(0.1, 1.0)), 2) if fail else 0.0})
    df = pd.DataFrame(rows)
    for c in ["VDD_V", "Temperature_C", "Measured_Value", "Lot_ID"]:
        df.loc[df.sample(frac=0.03, random_state=1).index, c] = np.nan
    df = pd.concat([df, df.sample(frac=0.02, random_state=2)], ignore_index=True)
    df["Result"] = df["Result"].where(rng.random(len(df)) > 0.1, df["Result"].str.lower())
    return df.sample(frac=1, random_state=3).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. STREAMLIT UI (Task 4)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Cleaning data...")
def run_pipeline(raw):
    mapping = map_columns(raw)
    clean, report = clean_data(raw, mapping)
    report["mapping"] = {k: v for k, v in mapping.items() if k != v}
    scored = add_anomaly_scores(clean)
    return scored, report


def fig_to_st(fig):
    st.pyplot(fig)
    plt.close(fig)


def tab_quality(df, report):
    st.subheader("Data quality profile")
    c = st.columns(4)
    c[0].metric("Raw rows", report["rows_raw"])
    c[1].metric("Clean rows", report["rows_clean"])
    c[2].metric("Duplicates removed", report["duplicates_removed"])
    c[3].metric("Outliers flagged", report["outliers"])
    if report["mapping"]:
        st.info(f"Columns renamed automatically: {report['mapping']}")
    miss = pd.Series(report["missing_before"])
    st.write("**Missing values before cleaning**")
    st.dataframe(miss[miss > 0].rename("missing").to_frame() if (miss > 0).any()
                 else pd.DataFrame({"missing": ["none"]}))
    st.write("**Cleaning decisions**")
    for line in report["decisions"]:
        st.markdown(f"- {line}")
    st.write("**Basic statistics**")
    st.dataframe(df[[c for c in NUMERIC if c in df]].describe().T)


def tab_dashboard(df):
    fail_rate = (df["Result"] == "FAIL").mean()
    c = st.columns(4)
    c[0].metric("Overall yield", f"{100 * (1 - fail_rate):.2f}%")
    c[1].metric("Fail rate", f"{100 * fail_rate:.2f}%")
    c[2].metric("Devices", df["Device_ID"].nunique())
    c[3].metric("Records", len(df))
    by_test = yield_by(df, "Test_ID")
    left, right = st.columns(2)
    with left:
        st.write("**Fail rate by test (%)**")
        st.bar_chart(by_test["fail_rate_%"])
    with right:
        if "Lot_ID" in df:
            st.write("**Fail rate by lot (%)**")
            st.bar_chart(yield_by(df, "Lot_ID")["fail_rate_%"])
    left, right = st.columns(2)
    with left:
        if "Failure_Mode" in df:
            fm = df.loc[df["Result"] == "FAIL", "Failure_Mode"].value_counts()
            st.write("**Failure modes**")
            st.bar_chart(fm)
    with right:
        for col in ["Temperature_C", "VDD_V"]:
            if col in df and df[col].notna().sum() > 50:
                n_bins = min(8, df[col].nunique())
                b = pd.qcut(df[col], q=n_bins, duplicates="drop")
                rate = df.groupby(b, observed=True)["Result"].apply(lambda s: 100 * (s == "FAIL").mean())
                rate.index = rate.index.astype(str)
                st.write(f"**Fail rate vs {col} (%)**")
                st.line_chart(rate)
    if "Wafer_ID" in df:
        with st.expander("Yield by wafer"):
            st.dataframe(yield_by(df, "Wafer_ID"))
    st.write("**Yield by test**")
    if "Test_Name" in df:
        names = df.drop_duplicates("Test_ID").set_index("Test_ID")["Test_Name"]
        by_test.insert(0, "Test_Name", names.reindex(by_test.index))
    st.dataframe(by_test)


def tab_anomalies(df):
    st.subheader("Anomaly ranking")
    st.caption("Score = 0.40 x IsolationForest rank + 0.35 x |per-test robust z|/6 + 0.25 x closeness to a spec "
               "limit. IsolationForest: 200 trees, contamination 3%, features standardised.")
    top_n = st.slider("Show top N", 10, 200, 50)
    cols = [c for c in ["Device_ID", "Test_ID", "Lot_ID", "Measured_Value", "Lower_Limit", "Upper_Limit",
                        "Result", "Failure_Mode", "robust_z", "anomaly_score"] if c in df]
    ranked = df.dropna(subset=["anomaly_score"]).sort_values("anomaly_score", ascending=False)
    st.dataframe(ranked[cols].head(top_n))
    left, right = st.columns(2)
    with left:
        st.write("**Most suspicious tests (mean anomaly score)**")
        st.bar_chart(ranked.groupby("Test_ID")["anomaly_score"].mean().sort_values(ascending=False))
    with right:
        st.write("**Most suspicious devices (max anomaly score)**")
        st.dataframe(ranked.groupby("Device_ID")["anomaly_score"].max().sort_values(ascending=False).head(10))
    flagged = ranked.head(max(1, int(0.03 * len(ranked))))
    st.write("**Top 3% anomalies vs known Result** (PASS here = hidden / marginal risk)")
    st.dataframe(flagged["Result"].value_counts().rename("count").to_frame())
    if "Failure_Mode" in df:
        st.write("**Top 3% anomalies by Failure_Mode**")
        st.dataframe(flagged["Failure_Mode"].value_counts().rename("count").to_frame())
    st.write("**Share of anomalies that are out of spec:** "
             f"{100 * flagged['Out_Of_Spec'].mean():.1f}%")


def tab_investigation(df):
    st.subheader("Failure investigation")
    c = st.columns(3)
    dev = c[0].selectbox("Device", sorted(df["Device_ID"].unique()))
    sub = df[df["Device_ID"] == dev]
    test = c[1].selectbox("Test", sorted(sub["Test_ID"].unique()))
    rows = sub[sub["Test_ID"] == test]
    row = rows.iloc[0]
    c[2].metric("Result", row["Result"])
    st.dataframe(rows.drop(columns=["Measured_Value_missing"], errors="ignore"))
    tdf = df[df["Test_ID"] == test]["Measured_Value"].dropna()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(tdf, bins=60, color="#9bb7d4")
    ax.axvline(row["Lower_Limit"], color="red", linestyle="--", label="limits")
    ax.axvline(row["Upper_Limit"], color="red", linestyle="--")
    if not pd.isna(row["Measured_Value"]):
        ax.axvline(row["Measured_Value"], color="black", linewidth=2, label="this record")
    ax.set_xlabel(f"Measured value ({test})")
    ax.legend()
    fig_to_st(fig)
    st.write("### AI analysis")
    if st.button("Analyse this record"):
        out = explain_record(row, df)
        if out["status"] == "insufficient":
            st.warning("Insufficient evidence to determine a root cause.")
        st.markdown("**Observed evidence**")
        for e in out["evidence"]:
            st.markdown(f"- {e}")
        if out["causes"]:
            st.markdown("**Possible causes (inferred, unverified)**")
            for e in out["causes"]:
                st.markdown(f"- {e}")
        st.markdown(f"**Recommendation:** {out['recommendation']}")
        st.markdown(f"**Confidence:** {out['confidence']}")


def tab_prediction(df):
    st.subheader("Failure prediction (before the test runs)")
    st.caption("Features used: " + ", ".join(sum(get_feature_lists(df), [])) +
               ". Excluded as post-test (leakage): " + ", ".join(c for c in POST_TEST_COLUMNS if c in df))
    if st.button("Train models"):
        try:
            with st.spinner("Training..."):
                st.session_state["model"] = train_models(df)
        except ValueError as e:
            st.error(str(e))
    m = st.session_state.get("model")
    if not m:
        return
    st.write(f"Split: {m['split']}. FAIL ratio: {100 * m['fail_ratio']:.1f}%.")
    st.dataframe(m["results"].style.format("{:.3f}"))
    st.info(f"A model that always predicts PASS would score {100 * m['baseline_accuracy']:.1f}% accuracy "
            "while catching 0% of failures: that is why precision, recall, F1 and PR-AUC matter. "
            f"Best model by PR-AUC: **{m['best']}** (threshold {m['threshold']:.3f}, tuned on out-of-fold "
            "training predictions).")
    st.write("**Confusion matrix** (rows = actual PASS/FAIL, cols = predicted PASS/FAIL)")
    st.dataframe(pd.DataFrame(m["confusion"], index=["actual PASS", "actual FAIL"],
                              columns=["pred PASS", "pred FAIL"]))
    st.write("**Feature influence (permutation importance, drop in PR-AUC)**")
    st.bar_chart(m["importance"])
    if st.button("Save model to models/ate_model.joblib"):
        os.makedirs("models", exist_ok=True)
        joblib.dump({k: m[k] for k in ["pipeline", "threshold", "features", "num", "cat"]},
                    "models/ate_model.joblib")
        st.success("Saved.")
    st.write("### Predict on new input")
    if st.button("Load a random unseen test-set record"):
        st.session_state["example"] = m["X_test"].sample(1).iloc[0].to_dict()
    ex = st.session_state.get("example", {})
    record = {}
    cols = st.columns(3)
    for i, f in enumerate(m["features"]):
        with cols[i % 3]:
            if f in m["cat"]:
                opts = sorted(df[f].dropna().unique())
                default = opts.index(ex[f]) if f in ex and ex[f] in opts else 0
                record[f] = st.selectbox(f, opts, index=default, key=f"in_{f}_{ex.get(f)}")
            else:
                default = float(ex[f]) if f in ex else float(df[f].median())
                record[f] = st.number_input(f, value=default, key=f"in_{f}_{ex.get(f)}")
    if st.button("Predict"):
        out = predict_record(m, record)
        st.metric("Fail probability", f"{100 * out['fail_probability']:.1f}%")
        st.write(f"Predicted: **{out['label']}** (decision threshold {out['threshold']:.3f})")


def main():
    st.set_page_config(page_title="ATE Test Intelligence", layout="wide")
    st.title("ATE Test Intelligence")
    with st.sidebar:
        st.header("Data")
        up = st.file_uploader("Upload CSV", type="csv")
        local = sorted(glob.glob("data/*.csv"))
        pick = st.selectbox("...or a file in data/", ["(none)"] + local)
        if st.button("Generate practice dataset"):
            st.session_state["practice"] = make_synthetic()
        if "practice" in st.session_state:
            st.download_button("Download practice CSV", st.session_state["practice"].to_csv(index=False),
                               "ATE_practice.csv")
    raw = None
    if up is not None:
        try:
            raw = pd.read_csv(io.BytesIO(up.getvalue()), sep=None, engine="python")
        except Exception as e:
            st.error(f"Could not read the CSV: {e}")
            st.stop()
        raw = raw.loc[:, ~raw.columns.astype(str).str.startswith("Unnamed")]
    elif pick != "(none)":
        raw = pd.read_csv(pick)
    elif "practice" in st.session_state:
        raw = st.session_state["practice"]
    if raw is None or raw.empty:
        st.info("Upload a CSV, pick one from data/, or generate a practice dataset from the sidebar.")
        st.stop()
    try:
        df, report = run_pipeline(raw)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    t = st.tabs(["Data quality", "Dashboard", "Anomalies", "Investigation", "Prediction"])
    with t[0]:
        tab_quality(df, report)
    with t[1]:
        tab_dashboard(df)
    with t[2]:
        tab_anomalies(df)
    with t[3]:
        tab_investigation(df)
    with t[4]:
        tab_prediction(df)


if __name__ == "__main__":
    main()