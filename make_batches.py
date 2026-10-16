import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("diabetes.csv")
train = df.sample(frac=0.6, random_state=42)
test = df.drop(train.index)
features = [c for c in df.columns if c != "Outcome"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(train[features], train["Outcome"])

train.to_csv("baseline_train.csv", index=False)

shifts = [0, 10, 20, 35, 50]
start = pd.Timestamp.now().normalize() - pd.Timedelta(days=len(shifts) - 1)
for i, s in enumerate(shifts, 1):
    b = test.sample(n=min(300, len(test)), replace=True, random_state=i).copy()
    b["Glucose"] += s
    b["BMI"] += s * 0.15
    b["prediction"] = model.predict(b[features])
    b["actual"] = b["Outcome"]
    b["timestamp"] = (start + pd.Timedelta(days=7 * (i - 1))).isoformat()
    b = b.drop(columns=["Outcome"])
    b.to_csv(f"batch_{i}_shift_{s}.csv", index=False)
    acc = (b["prediction"] == b["actual"]).mean()
    print(f"batch_{i}_shift_{s}.csv  rows={len(b)}  accuracy={acc:.3f}")