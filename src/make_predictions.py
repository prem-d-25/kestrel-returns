import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from prep import CATS, NUMS, load, dedupe, build_features

# train on ALL old orders
df = dedupe(load("train"))
X, y = build_features(df), df.returned.values

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATS),
    ("num", StandardScaler(), NUMS)])
model = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=2000))])
model.fit(X, y)
joblib.dump(model, "models/return_model.pkl")

# score the new orders
te = dedupe(load("test"))
te["score"] = model.predict_proba(build_features(te))[:, 1]

# same order and shape as sample_submission.csv
sub = pd.read_csv("data/sample_submission.csv")[["order_id"]]
out = sub.merge(te[["order_id", "score"]], on="order_id", how="left")
assert len(out) == 2096 and out.score.notna().all(), "row mismatch!"
out.to_csv("predictions.csv", index=False)

print(out.head())
print("rows:", len(out), "| average score:", round(out.score.mean(), 3))



# The model now studies all the old orders, not just 80%. More examples make it slightly better.
# It saves the trained model to models/return_model.pkl. This is the file the Streamlit app will load later, so nobody has to retrain it.
# It looks at the 2,096 new orders from test_unlabelled.csv, which have no answers, and gives each a score between 0 and 1.
# It writes those scores to predictions.csv in the same format as sample_submission.csv. This is the file Kestrel will grade