import sys
sys.path.insert(0, "src")   # so we can reuse the cleaning code from src/prep.py

import joblib
import pandas as pd
import streamlit as st

from prep import dedupe, build_features

REQUIRED = ["order_id", "order_placed_at", "customer_id", "sku", "sales_channel",
            "payment_mode", "discount_pct", "qty", "order_value_inr",
            "promised_delivery_days", "delivery_pincode", "is_gift",
            "customer_prior_orders", "customer_prior_returns", "delivery_note", "source"]
NUMBERS = ["discount_pct", "qty", "order_value_inr", "promised_delivery_days",
           "customer_prior_orders", "customer_prior_returns"]
KNOWN = {"sales_channel": ["app", "web", "marketplace", "partner_outlet"],
         "payment_mode": ["prepaid_upi", "prepaid_card", "cod", "emi"],
         "is_gift": ["Y", "N"]}


@st.cache_resource
def get_model():
    return joblib.load("models/return_model.pkl")


def read_and_check(file):
    """Returns (clean dataframe or None, list of errors, list of warnings)."""
    errors, warnings = [], []
    try:
        df = pd.read_csv(file, dtype={"delivery_pincode": str})
    except Exception:
        return None, ["This file could not be read as a CSV."], []
    if df.empty:
        return None, ["The file has no rows."], []

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return None, [f"Missing columns: {', '.join(missing)}"], []

    # blocking problems (the model would break or give nonsense)
    df["order_placed_at"] = pd.to_datetime(df["order_placed_at"], errors="coerce")
    n = df["order_placed_at"].isna().sum()
    if n:
        errors.append(f"{n} rows have a missing or invalid order_placed_at date.")
    for c in NUMBERS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        n = df[c].isna().sum()
        if n:
            errors.append(f"{n} rows have a blank or non-numeric value in '{c}'.")

    products = pd.read_csv("data/products.csv")
    customers = pd.read_csv("data/customers.csv")
    bad_sku = sorted(set(df["sku"]) - set(products["sku"]))
    if bad_sku:
        errors.append(f"Unknown product code(s): {', '.join(map(str, bad_sku[:5]))}")
    bad_cust = set(df["customer_id"]) - set(customers["customer_id"])
    if bad_cust:
        errors.append(f"{len(bad_cust)} customer_id(s) are not in customers.csv "
                      f"(e.g. {sorted(map(str, bad_cust))[:3]}).")

    # non-blocking problems
    for col, allowed in KNOWN.items():
        n = (~df[col].isin(allowed)).sum()
        if n:
            warnings.append(f"{n} rows have a new/unexpected '{col}' value. "
                            "The model has not seen it, so those scores are less reliable.")
    n = df["order_id"].duplicated().sum()
    if n:
        warnings.append(f"{n} duplicate order_id rows were merged into one each.")
    if (df["customer_prior_returns"] > df["customer_prior_orders"]).any():
        warnings.append("Some rows have more prior returns than prior orders. Please check them.")

    return df, errors, warnings


def show():
    st.header("Batch Predictions")
    st.write("Upload the orders waiting to be dispatched (for example `test_unlabelled.csv`). "
             "You get back one score per order: **higher = more likely to be returned**.")

    file = st.file_uploader("Upload orders CSV", type="csv")
    if file is None:
        return

    df, errors, warnings = read_and_check(file)
    for e in errors:
        st.error(e)
    if errors:
        st.stop()
    for w in warnings:
        st.warning(w)

    order = df["order_id"].drop_duplicates().tolist()   # keep the uploaded order
    clean = dedupe(df)
    clean["score"] = get_model().predict_proba(build_features(clean))[:, 1]
    out = clean.set_index("order_id").loc[order, ["score"]].reset_index()

    c1, c2, c3 = st.columns(3)
    c1.metric("Orders scored", f"{len(out):,}")
    c2.metric("Average score", f"{out.score.mean():.1%}")
    c3.metric("Orders scoring above 30%", f"{(out.score >= 0.3).sum():,}")

    st.dataframe(out.head(20), hide_index=True, use_container_width=True)
    st.download_button("Download predictions.csv",
                       out.to_csv(index=False), "predictions.csv", "text/csv")
    st.caption("Showing the first 20 rows. The download contains all of them.")