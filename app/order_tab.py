import requests
import pandas as pd
import streamlit as st
from datetime import datetime

API_URL = "http://localhost:8000/predict"


@st.cache_data
def get_products():
    return pd.read_csv("data/products.csv")


def show_result(res):
    p = res["return_probability"]
    risk = res["risk"]

    h = res["customer_history"]
    st.info(f"Customer history found automatically: {h['past_orders']} past order(s), "
            f"{h['past_returns']} past return(s).")

    c1, c2 = st.columns(2)
    c1.metric("Return probability", f"{p:.0%}")
    c2.metric("Risk level", risk)

    if res["action"] == "ship":
        st.success(f"Recommended action: {res['action_text']}")
    elif res["action"] == "call":
        st.warning(f"Recommended action: {res['action_text']}")
    else:
        st.error(f"Recommended action: {res['action_text']}")

    left, right = st.columns(2)
    with left:
        st.write("**Reasons the risk is higher**")
        for r in res["reasons_raising_risk"] or ["None of note"]:
            st.markdown(f"- {r}")
    with right:
        st.write("**Reasons the risk is lower**")
        for r in res["reasons_lowering_risk"] or ["None of note"]:
            st.markdown(f"- {r}")

    cost = res["expected_cost_rs"]
    st.caption(
        f"Average cost of each choice for this order: ship Rs {cost['ship']:,.0f}, "
        f"call Rs {cost['call']:,.0f}, hold Rs {cost['hold']:,.0f}. "
        f"The recommended action saves about Rs {res['saving_vs_ship_rs']:,.0f} "
        "compared with shipping. These use Kestrel's policy figures "
        "(Rs 1,150 per return, Rs 45 per call)."
    )


def show():
    st.header("Check an Order")
    st.write("Enter the order details and click **Predict**. "
             "The answer comes from the Kestrel returns service.")

    products = get_products()
    names = dict(zip(products["sku"], products["model_name"]))
    prices = dict(zip(products["sku"], products["list_price_inr"]))

    with st.form("order_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            customer_id = st.text_input("Customer ID", "KC101598")
            sku = st.selectbox("Product", list(names), format_func=lambda s: f"{names[s]} ({s})")
            qty = st.number_input("Quantity", 1, 50, 1)
            discount = st.number_input("Discount %", 0.0, 90.0, 10.0)
        with c2:
            channel = st.selectbox("Sales channel", ["app", "web", "marketplace", "partner_outlet"])
            payment = st.selectbox("Payment mode", ["prepaid_upi", "prepaid_card", "cod", "emi"])
            gift = st.selectbox("Is it a gift?", ["N", "Y"])
            days = st.number_input("Promised delivery (days)", 0, 60, 5)
        with c3:
            pincode = st.text_input("Delivery pincode (000000 if none)", "440075")
            note = st.text_input("Delivery note (optional)", "")

        value = round(prices[sku] * qty * (1 - discount / 100), 2)
        st.caption(f"Order value (calculated from list price, quantity and discount): Rs {value:,.2f}")
        go = st.form_submit_button("Predict")

    if not go:
        return

    if not (pincode.isdigit() and len(pincode) == 6):
        st.error("Pincode must be exactly 6 digits.")
        return

    order = {
        "order_placed_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "customer_id": customer_id.strip(),
        "sku": sku, "sales_channel": channel, "payment_mode": payment,
        "discount_pct": discount, "qty": int(qty), "order_value_inr": value,
        "promised_delivery_days": int(days), "delivery_pincode": pincode,
        "is_gift": gift,
        "delivery_note": note or None,
    }

    try:
        resp = requests.post(API_URL, json=order, timeout=10)
    except requests.exceptions.RequestException:
        st.error("The returns service is not running, so no prediction can be made. "
                 "Start it with:  uvicorn server:app --app-dir api --port 8000  "
                 "and then click Predict again.")
        return

    if resp.status_code == 200:
        show_result(resp.json())
        return

    try:
        detail = resp.json().get("detail", "Unknown problem")
    except ValueError:   # the service crashed and sent plain text
        st.error("The returns service hit an internal error for this order. "
                 "Please try again, or tell the data team. (Technical detail is in the service window.)")
        return
    if isinstance(detail, list):   # validation messages from the endpoint
        detail = "; ".join(f"{d['loc'][-1]}: {d['msg']}" for d in detail)
    st.error(f"The order could not be checked: {detail}")