"""
demo_stream.py — Watchdog real-data demo (never runs in production).

Trains a LogisticRegression on the UCI Adult Income dataset, registers it
with a real baseline, streams predictions through the Watchdog API, then
injects a drift phase and delayed labels so the monitoring engine has
real data to work with.

Usage
-----
    cd backend
    python scripts/demo_stream.py [--base-url http://localhost:8000] [--drift]

Requirements: scikit-learn, pandas, numpy (already in requirements.txt via scipy/pandas)
"""

from __future__ import annotations

import argparse
import sys
import time
import io
import json
import urllib.request
import urllib.error
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on path so watchdog_client is importable
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "clients" / "python"))

from watchdog_client import WatchdogClient  # noqa: E402


def _fetch_adult() -> tuple:
    """Download UCI Adult Income dataset and return (X_train, X_test, y_train, y_test, feature_names)."""
    try:
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        print("ERROR: scikit-learn and pandas are required. pip install scikit-learn pandas")
        sys.exit(1)

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    cols = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
    ]
    print("Downloading Adult Income dataset...")
    try:
        df = pd.read_csv(url, names=cols, na_values=" ?", skipinitialspace=True)
    except Exception as e:
        print(f"Download failed ({e}). Using synthetic fallback data.")
        rng = np.random.default_rng(42)
        n = 5000
        df = pd.DataFrame({
            "age": rng.integers(18, 70, n).astype(float),
            "education_num": rng.integers(1, 16, n).astype(float),
            "hours_per_week": rng.integers(20, 60, n).astype(float),
            "capital_gain": rng.exponential(500, n),
            "capital_loss": rng.exponential(50, n),
            "income": rng.choice([0, 1], n, p=[0.75, 0.25]),
        })
        feature_cols = ["age", "education_num", "hours_per_week", "capital_gain", "capital_loss"]
        X = df[feature_cols].values
        y = df["income"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test, feature_cols

    # Drop rows with nulls in key columns
    numeric_cols = ["age", "education_num", "hours_per_week", "capital_gain", "capital_loss"]
    df = df.dropna(subset=numeric_cols)
    df["label"] = (df["income"].str.strip() == ">50K").astype(int)

    X = df[numeric_cols].values.astype(float)
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test, numeric_cols


def _train_model(X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def _register_model(base_url: str) -> str:
    """Create a model via the API and return its ID."""
    payload = json.dumps({
        "name": "Adult Income Classifier",
        "version": "v1.0-demo",
        "algorithm": "Logistic Regression",
        "environment": "staging",
        "taskType": "binary_classification",
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/api/models",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())["id"]


def _upload_baseline(base_url: str, model_id: str, X_train, feature_names: list[str]) -> None:
    """Upload training data as a baseline CSV."""
    import io as _io
    import csv

    buf = _io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(feature_names)
    for row in X_train:
        writer.writerow([round(float(v), 4) for v in row])
    csv_bytes = buf.getvalue().encode()

    boundary = "----WatchdogBoundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="baseline.csv"\r\n'
        f"Content-Type: text/csv\r\n\r\n"
    ).encode() + csv_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{base_url}/api/models/{model_id}/baseline",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    print(f"  Baseline uploaded: {result.get('featureCount')} features, {result.get('sampleSize')} rows")
    if result.get("warnings"):
        for w in result["warnings"]:
            print(f"  Warning: {w}")


def _stream_predictions(
    client: WatchdogClient,
    model,
    X_test,
    y_test,
    feature_names: list[str],
    batch_size: int = 50,
    inject_drift: bool = False,
) -> list[str]:
    """Stream test predictions in batches. Returns list of prediction_ids."""
    import numpy as np

    prediction_ids = []
    n = len(X_test)
    n_batches = (n + batch_size - 1) // batch_size

    print(f"\nStreaming {n} predictions in {n_batches} batches of {batch_size}...")

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, n)
        X_batch = X_test[start:end]

        # Inject drift in the second half: shift age and hours_per_week
        if inject_drift and i >= n_batches // 2:
            X_batch = X_batch.copy()
            X_batch[:, 0] += 15   # age shift +15 years
            X_batch[:, 2] += 10   # hours_per_week shift +10

        probs = model.predict_proba(X_batch)[:, 1]
        preds = (probs >= 0.5).astype(int)

        batch = []
        for j, (x_row, pred, prob) in enumerate(zip(X_batch, preds, probs)):
            pid = f"demo-{start + j:05d}"
            prediction_ids.append(pid)
            batch.append({
                "predictionId": pid,
                "features": {name: round(float(v), 4) for name, v in zip(feature_names, x_row)},
                "prediction": float(pred),
                "score": round(float(prob), 6),
            })

        result = client.log_batch(batch)
        print(f"  Batch {i+1}/{n_batches}: ingested={result.get('ingested')}, "
              f"duplicates={result.get('duplicates')}"
              + (" [DRIFT INJECTED]" if inject_drift and i >= n_batches // 2 else ""))
        time.sleep(0.1)

    return prediction_ids


def _send_labels(client: WatchdogClient, prediction_ids: list[str], y_test) -> None:
    """Send ground-truth labels for all predictions."""
    print(f"\nSending {len(prediction_ids)} ground-truth labels...")
    labels = [
        {"predictionId": pid, "actualValue": float(y_test[i])}
        for i, pid in enumerate(prediction_ids)
        if i < len(y_test)
    ]
    # Send in chunks of 500
    for i in range(0, len(labels), 500):
        chunk = labels[i:i+500]
        result = client.log_ground_truth(chunk)
        print(f"  Labels chunk {i//500 + 1}: matched={result.get('matched')}, orphaned={result.get('orphaned')}")


def _trigger_evaluation(base_url: str, model_id: str) -> None:
    req = urllib.request.Request(
        f"{base_url}/api/models/{model_id}/reevaluate",
        data=b"",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    print(f"\nEvaluation result:")
    print(f"  Status:       {result.get('status')}")
    print(f"  Health score: {result.get('healthScore')}")
    print(f"  Predictions:  {result.get('nPredictions')}")
    print(f"  Labeled:      {result.get('nLabeled')}")
    drifts = result.get("drifts", [])
    if drifts:
        print(f"  Feature drift ({len(drifts)} features):")
        for d in sorted(drifts, key=lambda x: x.get("driftScore") or 0, reverse=True)[:5]:
            print(f"    {d['featureName']:20s}  PSI={d.get('driftScore') or 'n/a':<8}  {d['status']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Watchdog demo stream")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--drift", action="store_true", help="Inject drift in second half of predictions")
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    print("=" * 60)
    print("Watchdog AI — Real-data demo stream")
    print(f"Target: {args.base_url}")
    print(f"Drift injection: {'ON' if args.drift else 'OFF'}")
    print("=" * 60)

    # 1. Load data and train model
    X_train, X_test, y_train, y_test, feature_names = _fetch_adult()
    print(f"\nDataset: {len(X_train)} train, {len(X_test)} test rows, features: {feature_names}")

    print("Training Logistic Regression...")
    model = _train_model(X_train, y_train)
    train_acc = (model.predict(X_train) == y_train).mean()
    test_acc = (model.predict(X_test) == y_test).mean()
    print(f"  Train accuracy: {train_acc:.3f}  Test accuracy: {test_acc:.3f}")

    # 2. Register model
    print("\nRegistering model...")
    model_id = _register_model(args.base_url)
    print(f"  Model ID: {model_id}")

    # 3. Upload baseline
    print("\nUploading training baseline...")
    _upload_baseline(args.base_url, model_id, X_train, feature_names)

    # 4. Stream predictions
    client = WatchdogClient(base_url=args.base_url, model_id=model_id)
    prediction_ids = _stream_predictions(
        client, model, X_test, y_test, feature_names,
        batch_size=args.batch_size,
        inject_drift=args.drift,
    )

    # 5. Send labels
    _send_labels(client, prediction_ids, y_test)

    # 6. Trigger evaluation
    print("\nTriggering drift + performance evaluation...")
    _trigger_evaluation(args.base_url, model_id)

    print("\nDemo complete. Open the Watchdog UI to see real drift and performance metrics.")
    print(f"Model detail: {args.base_url.replace('8000', '5173')}/models/{model_id}")


if __name__ == "__main__":
    main()
