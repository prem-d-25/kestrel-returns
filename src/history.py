import pandas as pd


def build_history():
    """One row per customer: how many orders and returns they have had so far."""
    cols = ["customer_id", "order_placed_at", "customer_prior_orders",
            "customer_prior_returns"]
    tr = pd.read_csv("data/train.csv", usecols=cols + ["returned"],
                     parse_dates=["order_placed_at"])
    te = pd.read_csv("data/test_unlabelled.csv", usecols=cols,
                     parse_dates=["order_placed_at"])
    te["returned"] = 0   # outcome not known yet
    d = pd.concat([tr, te]).sort_values("order_placed_at")
    last = d.groupby("customer_id").tail(1)   # most recent order per customer
    return pd.DataFrame({
        "orders": (last.customer_prior_orders + 1).values,
        "returns": (last.customer_prior_returns + last.returned).values,
    }, index=last.customer_id.values)