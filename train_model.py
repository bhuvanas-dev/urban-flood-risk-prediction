import os, json, re, warnings
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.path.join(ROOT, "train_2400.csv")
TEST_FILE = os.path.join(ROOT, "test_500.csv")
MODEL_DIR = os.path.join(ROOT, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# These are the five recurring descriptive tags in the supplied dataset.
# event_YYYY-MM-DD values are retained as event metadata, not learned as
# prediction classes because most are one-off date identifiers.
TARGET_TAGS = ["monitor", "low_lying", "extreme_rain_history",
               "sparse_drainage", "ponding_hotspot"]

TEXT_COLUMNS = ["city_name", "admin_ward", "dem_source", "land_use",
                "soil_group", "storm_drain_type", "rainfall_source"]
NUMERIC_COLUMNS = ["latitude", "longitude", "elevation_m",
                   "drainage_density_km_per_km2", "storm_drain_proximity_m",
                   "historical_rainfall_intensity_mm_hr", "return_period_years"]

def split_tags(value):
    return [x.strip() for x in str(value).split("|") if x.strip()]

def make_text(frame):
    # TF-IDF is applied only to descriptive text/categorical fields.
    return frame[TEXT_COLUMNS].fillna("unknown").astype(str).agg(" ".join, axis=1)

def make_numeric(frame):
    return frame[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0)

def make_targets(frame):
    rows = []
    for value in frame["risk_labels"].fillna(""):
        tags = set(split_tags(value))
        rows.append([int(tag in tags) for tag in TARGET_TAGS])
    return np.asarray(rows, dtype=np.int32)

def main():
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    missing = [c for c in TEXT_COLUMNS + NUMERIC_COLUMNS + ["risk_labels"] if c not in train.columns]
    if missing:
        raise ValueError("Missing required dataset columns: " + ", ".join(missing))

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1200,
                                 token_pattern=r"(?u)\b\w+\b")
    X_text_train = vectorizer.fit_transform(make_text(train))
    X_text_test = vectorizer.transform(make_text(test))

    scaler = StandardScaler()
    X_num_train = scaler.fit_transform(make_numeric(train))
    X_num_test = scaler.transform(make_numeric(test))
    X_train = hstack([X_text_train, csr_matrix(X_num_train)]).tocsr()
    X_test = hstack([X_text_test, csr_matrix(X_num_test)]).tocsr()
    y_train, y_test = make_targets(train), make_targets(test)

    models, report = {}, {}
    predictions = np.zeros_like(y_test)
    for i, tag in enumerate(TARGET_TAGS):
        positives = int(y_train[:, i].sum())
        negatives = int(len(y_train) - positives)
        if positives == 0 or negatives == 0:
            raise ValueError(f"Training split needs both positive and negative examples for '{tag}'.")
        model = XGBClassifier(
            n_estimators=180, max_depth=4, learning_rate=0.06,
            subsample=0.9, colsample_bytree=0.9,
            objective="binary:logistic", eval_metric="logloss",
            scale_pos_weight=negatives / positives,
            random_state=42, n_jobs=4
        )
        model.fit(X_train, y_train[:, i])
        predictions[:, i] = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
        models[tag] = model
        p, r, f, _ = precision_recall_fscore_support(
            y_test[:, i], predictions[:, i], average="binary", zero_division=0
        )
        report[tag] = {
            "test_support": int(y_test[:, i].sum()),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1_score": round(float(f), 4)
        }

    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "numeric_scaler.joblib"))
    joblib.dump(models, os.path.join(MODEL_DIR, "xgboost_models.joblib"))
    metadata = {
        "target_tags": TARGET_TAGS, "text_columns": TEXT_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "train_records": len(train), "test_records": len(test),
        "tfidf_features": int(X_text_train.shape[1]),
        "numeric_features": len(NUMERIC_COLUMNS),
        "test_exact_match_accuracy": round(float(np.mean(np.all(y_test == predictions, axis=1))), 4),
        "per_tag_metrics": report,
        "note": "Date-specific event_* tags remain in the original dataset but are not predicted as generalizable classes."
    }
    with open(os.path.join(MODEL_DIR, "evaluation_report.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("\nMODEL TRAINING COMPLETED")
    print(f"Training records: {len(train)} | Testing records: {len(test)}")
    print(f"TF-IDF features: {X_text_train.shape[1]} | Numeric features: {len(NUMERIC_COLUMNS)}")
    print("\nPer-label test metrics:")
    for tag, values in report.items():
        print(f"{tag:24s} Precision={values['precision']:.3f} Recall={values['recall']:.3f} F1={values['f1_score']:.3f} Support={values['test_support']}")
    print("\nExact-match accuracy:", metadata["test_exact_match_accuracy"])
    print("\nSaved trained models and report in the 'model' folder.")

if __name__ == "__main__":
    main()
