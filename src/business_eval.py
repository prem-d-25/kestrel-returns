# This replays the same 20% newest orders as before (the model never saw them). It compares four strategies using the real outcomes.

import json
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import rules as R
from prep import CATS, NUMS, load, dedupe, build_features

df = dedupe(load("train"))
X, y = build_features(df), df.returned.values
cut = int(len(df) * 0.8)
Xtr, Xte, ytr, yte = X.iloc[:cut], X.iloc[cut:], y[:cut], y[cut:]

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATS),
    ("num", StandardScaler(), NUMS)])
model = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=2000))])
model.fit(Xtr, ytr)
p = model.predict_proba(Xte)[:, 1]
V = Xte.order_value_inr.values
shield = Xte.shield.values == 1
n = len(yte)


def actual(action, margin=R.MARGIN):
    if action == "ship":
        return yte * R.RETURN_COST
    if action == "call":
        return R.CALL_COST + yte * (1 - R.CALL_PREVENTS) * R.RETURN_COST
    return R.HOLD_CANCEL * margin * V + (1 - R.HOLD_CANCEL) * yte * R.RETURN_COST


def total(actions, margin=R.MARGIN):
    cost = np.zeros(n)
    for a in ("ship", "call", "hold"):
        m = actions == a
        cost[m] = actual(a, margin)[m]
    return cost.sum()


ship_all = np.full(n, "ship")
ritu = np.where(p >= 0.2, "hold", "ship")
c = R.costs(p, V)
ours = np.where((c["call"] < c["ship"]) & ((c["call"] <= c["hold"]) | shield), "call", "ship")
ours = np.where((c["hold"] < np.minimum(c["ship"], c["call"])) & ~shield, "hold", ours)

base = total(ship_all)
rows = []
for name, a in [("Ship everything (do nothing)", ship_all),
                ("Ritu's plan: hold orders scoring 20%+", ritu),
                ("Our plan: call orders at about 11%+ risk", ours)]:
    t = total(a)
    rows.append({"strategy": name, "total_cost": float(t), "saving": float(base - t),
                 "calls": int((a == "call").sum()), "holds": int((a == "hold").sum()),
                 "saving_per_700": float((base - t) / n * 700)})
    print(f"{name:42s} saving Rs {base - t:9,.0f} | calls {(a=='call').sum()} | holds {(a=='hold').sum()}")

called = ours == "call"
breakeven = called.sum() * R.CALL_COST / (yte[called].sum() * R.RETURN_COST)

result = {
    "test_orders": int(n), "returns_in_test": int(yte.sum()),
    "rows": rows,
    "ritu_margin_check": {f"{m:.0%}": float(base - total(ritu, m)) for m in (0.10, 0.20, 0.30)},
    "call_breakeven_effect": float(breakeven),
    "call_effect_assumed": R.CALL_PREVENTS,
}
with open("models/business.json", "w") as f:
    json.dump(result, f, indent=2)
print("call break-even effect:", round(breakeven, 3))
print("saved models/business.json")