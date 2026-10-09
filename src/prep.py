import numpy as np
import pandas as pd

CATS = ["sales_channel", "payment_mode", "family", "tier", "state", "note"]
NUMS = ["shield", "is_gift", "no_address", "discount_pct", "qty", "order_value_inr",
        "promised_delivery_days", "list_price_inr", "warranty_months",
        "customer_prior_orders", "customer_prior_returns", "prior_return_rate",
        "hour", "dow"]


def load(split):
    path = {"train": "data/train.csv", "test": "data/test_unlabelled.csv"}[split]
    return pd.read_csv(path, parse_dates=["order_placed_at"],
                       dtype={"delivery_pincode": str})


def dedupe(df):
    # 'crm' sorts before 'partner_feed', so keep="first" keeps the crm copy
    df = df.sort_values(["order_id", "source"])
    df = df.drop_duplicates("order_id", keep="first")
    return df.sort_values("order_placed_at").reset_index(drop=True)


def build_features(df):
    cu = pd.read_csv("data/customers.csv")
    pr = pd.read_csv("data/products.csv")
    d = df.merge(cu, on="customer_id", how="left").merge(pr, on="sku", how="left")

    # October 2025 gateway fix: values stored x100
    expected = d.list_price_inr * d.qty * (1 - d.discount_pct / 100)
    ratio = d.order_value_inr / expected
    d["order_value_inr"] = np.where(ratio > 50, d.order_value_inr / 100, d.order_value_inr)

    X = pd.DataFrame(index=d.index)
    X["sales_channel"] = d.sales_channel
    X["payment_mode"] = d.payment_mode
    X["family"] = d.family
    X["tier"] = d.sku.str[-2:]
    X["state"] = d.state
    X["note"] = (d.delivery_note.fillna("none")
                 .str.replace(r"^Landmark.*", "Landmark", regex=True)
                 .str.replace(r"\d+", "#", regex=True)
                 .str.replace(r"flat #[A-D]", "flat #", regex=True))
    X["shield"] = (d.shield_member == "Y").astype(int)
    X["is_gift"] = (d.is_gift == "Y").astype(int)
    X["no_address"] = (d.delivery_pincode == "000000").astype(int)
    for c in ["discount_pct", "qty", "order_value_inr", "promised_delivery_days",
              "list_price_inr", "warranty_months",
              "customer_prior_orders", "customer_prior_returns"]:
        X[c] = d[c]
    X["prior_return_rate"] = d.customer_prior_returns / d.customer_prior_orders.clip(lower=1)
    X["hour"] = d.order_placed_at.dt.hour
    X["dow"] = d.order_placed_at.dt.dayofweek
    return X[CATS + NUMS]


# Removes duplicate orders and the two columns that contain the answer.
# Fixes the October prices that are 100 times too big.
# Joins in customer and product info and builds the clues the model will look at (payment mode, Shield, gift, discount, previous return rate, and so on).