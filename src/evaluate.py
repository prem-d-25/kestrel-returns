# This file will run 80 20 data and save the result 

import json
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from prep import CATS, NUMS, load, dedupe, build_features

CUTOFFS = [0.5, 0.3, 0.2, 0.15]

df = dedupe(load("train"))
X, y = build_features(df), df.returned.values

# oldest 80% to study, newest 20% as the hidden exam
cut = int(len(df) * 0.8)
Xtr, Xte, ytr, yte = X.iloc[:cut], X.iloc[cut:], y[:cut], y[cut:]

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATS),
    ("num", StandardScaler(), NUMS)])
model = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=2000))])
model.fit(Xtr, ytr)
p = model.predict_proba(Xte)[:, 1]

rows = []
for t in CUTOFFS:
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    rows.append({
        "cutoff": t,
        "flagged": int(tp + fp),
        "caught": int(tp), "missed": int(fn),
        "false_alarms": int(fp), "correct_ok": int(tn),
        "accuracy": float(accuracy_score(yte, pred)),
        "precision": float(precision_score(yte, pred, zero_division=0)),
        "recall": float(recall_score(yte, pred)),
    })

result = {
    "train_rows": int(len(Xtr)), "test_rows": int(len(Xte)),
    "train_from": str(df.order_placed_at.iloc[0].date()),
    "train_to": str(df.order_placed_at.iloc[cut - 1].date()),
    "test_from": str(df.order_placed_at.iloc[cut].date()),
    "test_to": str(df.order_placed_at.iloc[-1].date()),
    "real_returns_in_test": int(yte.sum()),
    "baseline_accuracy": float(1 - yte.mean()),
    "roc_auc": float(roc_auc_score(yte, p)),
    "pr_auc": float(average_precision_score(yte, p)),
    "cutoffs": rows,
}
with open("models/evidence.json", "w") as f:
    json.dump(result, f, indent=2)
print("saved models/evidence.json")
print("ROC-AUC:", round(result["roc_auc"], 4))