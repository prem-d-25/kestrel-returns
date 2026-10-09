import numpy as np

# Clues that are real but not useful to an employee, so we do not show them
HIDE = ("tier_", "note_", "state_", "warranty_months", "list_price_inr",
        "hour", "dow", "qty")

# Yes/no clues: (text when yes, text when no)
FLAGS = {
    "shield": ("Is a Shield member", "Is not a Shield member"),
    "is_gift": ("Order is a gift", "Order is not a gift"),
    "no_address": ("No delivery address captured", "Delivery address captured"),
}

CATEGORY = {
    "payment_mode_cod": "Pays cash on delivery",
    "payment_mode_prepaid_upi": "Paid by UPI",
    "payment_mode_prepaid_card": "Paid by card",
    "payment_mode_emi": "Pays by EMI",
    "sales_channel_marketplace": "Bought on a marketplace",
    "sales_channel_partner_outlet": "Bought at a partner outlet",
    "sales_channel_app": "Bought on the app",
    "sales_channel_web": "Bought on the website",
}


def phrase(name, row):
    """Turn one clue into a sentence an employee can read. None = do not show."""
    if name.startswith(HIDE):
        return None
    if name in CATEGORY:
        return CATEGORY[name]
    if name.startswith("family_"):
        return "Product type: " + name[len("family_"):]
    v = row[name].iloc[0] if name in row.columns else None
    if name in FLAGS:
        return FLAGS[name][0] if v == 1 else FLAGS[name][1]
    if name == "customer_prior_returns":
        return f"Customer has {int(v)} past return(s)"
    if name == "customer_prior_orders":
        return f"Customer has {int(v)} past order(s)"
    if name == "prior_return_rate":
        return f"{v:.0%} of this customer's past orders were returned"
    if name == "discount_pct":
        return f"Discount of {v:.0f}%"
    if name == "promised_delivery_days":
        return f"Delivery promised in {v:.0f} days"
    if name == "order_value_inr":
        return f"Order value Rs {v:,.0f}"
    return None


def explain(model, X_row, top=3):
    """X_row: one-row DataFrame from build_features.
    Returns (reasons that raise risk, reasons that lower risk) as plain sentences."""
    pre, clf = model.named_steps["pre"], model.named_steps["model"]
    contrib = pre.transform(X_row)[0] * clf.coef_[0]
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]

    up, down = [], []
    for i in np.argsort(contrib)[::-1]:            # biggest risk-raisers first
        text = phrase(names[i], X_row)
        if text and contrib[i] > 0.05 and len(up) < top:
            up.append(text)
    for i in np.argsort(contrib):                  # biggest risk-lowerers first
        text = phrase(names[i], X_row)
        if text and contrib[i] < -0.05 and len(down) < top:
            down.append(text)
    return up, down