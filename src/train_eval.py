import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from prep import CATS, NUMS, load, dedupe, build_features

df = dedupe(load("train"))
X, y = build_features(df), df.returned.values
print("rows after dedupe:", len(df))

# time-based 80/20 split: oldest 80% train, newest 20% test
cut = int(len(df) * 0.8)
Xtr, Xte, ytr, yte = X.iloc[:cut], X.iloc[cut:], y[:cut], y[cut:]
print("train:", df.order_placed_at.iloc[0], "->", df.order_placed_at.iloc[cut - 1])
print("test :", df.order_placed_at.iloc[cut], "->", df.order_placed_at.iloc[-1])
print("return rate train/test:", ytr.mean().round(3), yte.mean().round(3))
print("BASELINE 'nobody returns' accuracy:", round(1 - yte.mean(), 4))


def make(model, scale=False):
    num = StandardScaler() if scale else "passthrough"
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATS),
        ("num", num, NUMS)])
    return Pipeline([("pre", pre), ("model", model)])


models = {
    "LogisticRegression": make(LogisticRegression(max_iter=2000), scale=True),
    "HistGradientBoosting": make(HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.05, max_iter=200, random_state=0)),
}

for name, m in models.items():
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    print(f"\n=== {name} ===")
    print("ROC-AUC:", round(roc_auc_score(yte, p), 4),
          "| PR-AUC:", round(average_precision_score(yte, p), 4))
    for t in [0.5, 0.3, 0.2, 0.15]:
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
        print(f"thr {t}: acc={accuracy_score(yte, pred):.3f} "
              f"prec={precision_score(yte, pred, zero_division=0):.3f} "
              f"rec={recall_score(yte, pred):.3f}  TP={tp} FP={fp} FN={fn} TN={tn}")



# It takes the older 80% of orders and lets the model study them.
# It hides the newest 20% (2,101 orders), asks the model to predict them, and compares its guesses with the real answers.
# That's where the numbers you pasted came from: how many returns it caught, how many false alarms, and so on.