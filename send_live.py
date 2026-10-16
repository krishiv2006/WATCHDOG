import uuid
import pandas as pd
import requests

BASE = "http://localhost:8001"
MODEL_ID = "71899ab1-46f6-4958-a51f-83834195827a"
DRIFT = False          # change to True to simulate drift

df = pd.read_csv("diabetes.csv").sample(400, random_state=1)
features = [c for c in df.columns if c != "Outcome"]

if DRIFT:
    df["Glucose"] += 40
    df["BMI"] += 6
    df["Age"] += 10

rows = [
    {
        "predictionId": str(uuid.uuid4()),
        "features": {c: float(r[c]) for c in features},
        "prediction": float(r["Outcome"]),
    }
    for _, r in df.iterrows()
]

for i in range(0, len(rows), 100):
    res = requests.post(
        f"{BASE}/api/models/{MODEL_ID}/predictions",
        json={"predictions": rows[i:i + 100]},
    )
    print(res.status_code, res.text[:300])