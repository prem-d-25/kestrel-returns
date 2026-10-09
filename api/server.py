import sys
sys.path.insert(0, "src")   # reuse the code from src/ (run from the kestrel-returns folder)

import joblib
import pandas as pd
from typing import Literal, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import rules as R
from prep import build_features
from reasons import explain
from history import build_history

app = FastAPI(title="Kestrel Returns API")
model = joblib.load("models/return_model.pkl")
KNOWN_SKUS = set(pd.read_csv("data/products.csv")["sku"])
KNOWN_CUSTOMERS = set(pd.read_csv("data/customers.csv")["customer_id"])
HISTORY = build_history()
PRICES = pd.read_csv("data/products.csv").set_index("sku")["list_price_inr"].to_dict()

ACTION_TEXT = {
    "ship": "Ship normally",
    "call": "Make a confirmation call before dispatch",
    "hold": "Hold the order",
}


class Order(BaseModel):
    order_placed_at: str = Field(examples=["2026-07-01 09:44"])
    customer_id: str = Field(examples=["KC101598"])
    sku: str = Field(examples=["KH-MG-01"])
    sales_channel: Literal["app", "web", "marketplace", "partner_outlet"]
    payment_mode: Literal["prepaid_upi", "prepaid_card", "cod", "emi"]
    discount_pct: float = Field(ge=0, le=90)
    qty: int = Field(ge=1, le=50)
    order_value_inr: float = Field(gt=0)
    promised_delivery_days: int = Field(ge=0, le=60)
    delivery_pincode: str = Field(examples=["440075"])
    is_gift: Literal["Y", "N"]
    customer_prior_orders: Optional[int] = Field(default=None, ge=0)
    customer_prior_returns: Optional[int] = Field(default=None, ge=0)
    delivery_note: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(order: Order):
    # 1. check the product and customer exist
    if order.sku not in KNOWN_SKUS:
        raise HTTPException(422, f"Unknown product code: {order.sku}")
    if order.customer_id not in KNOWN_CUSTOMERS:
        raise HTTPException(422, f"Unknown customer: {order.customer_id}")

    # 1b. the order value must match list price x quantity x discount
    expected = PRICES[order.sku] * order.qty * (1 - order.discount_pct / 100)
    if abs(order.order_value_inr - expected) > 0.01 * expected:
        raise HTTPException(
            422, f"Order value Rs {order.order_value_inr:,.2f} does not match list price x "
                 f"quantity x discount (Rs {expected:,.2f}). Please check it "
                 "(values 100 times too big usually mean the amount is in paise).")
    
    
    # 2. customer history: use what was sent, otherwise look it up
    po, pr = order.customer_prior_orders, order.customer_prior_returns
    if po is None and pr is None:
        source = "looked up"
        if order.customer_id in HISTORY.index:
            po = int(HISTORY.loc[order.customer_id, "orders"])
            pr = int(HISTORY.loc[order.customer_id, "returns"])
        else:
            po, pr = 0, 0
    elif po is None or pr is None:
        raise HTTPException(422, "Give both past orders and past returns, or neither.")
    else:
        source = "entered"
    if pr > po:
        raise HTTPException(422, "Past returns cannot be more than past orders.")

    # 3. build the one-row table the model expects
    try:
        df = pd.DataFrame([order.model_dump()])
        df["customer_prior_orders"], df["customer_prior_returns"] = po, pr
        df["order_placed_at"] = pd.to_datetime(df["order_placed_at"])
    except Exception:
        raise HTTPException(422, "order_placed_at is not a valid date/time.")

    # 4. model answers, then business rules decide the action
    X = build_features(df)
    p = float(model.predict_proba(X)[0, 1])
    rec = R.recommend(p, float(X.order_value_inr.iloc[0]), bool(X.shield.iloc[0]))
    up, down = explain(model, X)
    return {
        "return_probability": round(p, 4),
        "risk": rec["risk"],
        "action": rec["action"],
        "action_text": ACTION_TEXT[rec["action"]],
        "reasons_raising_risk": up,
        "reasons_lowering_risk": down,
        "customer_history": {"past_orders": po, "past_returns": pr, "source": source},
        "expected_cost_rs": rec["expected_cost"],
        "saving_vs_ship_rs": rec["saving_vs_ship"],
    }