# Evidence Page 

import json
import pandas as pd
import streamlit as st


def show():
    st.header("Model Evidence")

    try:
        with open("models/evidence.json") as f:
            r = json.load(f)
        needed = ["train_rows", "test_rows", "train_from", "train_to", "test_from",
                  "test_to", "real_returns_in_test", "baseline_accuracy",
                  "roc_auc", "cutoffs"]
        if any(k not in r for k in needed):
            raise KeyError("incomplete results file")
    except FileNotFoundError:
        st.error("Results file not found. Run:  python src/evaluate.py")
        return
    except (json.JSONDecodeError, KeyError):
        st.error("The results file is damaged or out of date. "
                 "Run:  python src/evaluate.py  to create it again.")
        return

    st.subheader("How we tested it")
    st.write(
        f"The model studied **{r['train_rows']:,} older orders** "
        f"({r['train_from']} to {r['train_to']}). "
        f"Then we tested it on the **{r['test_rows']:,} newest orders** "
        f"({r['test_from']} to {r['test_to']}) that it had never seen. "
        f"{r['real_returns_in_test']} of those were really returned."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC (0.5 = coin flip, 1.0 = perfect)", f"{r['roc_auc']:.2f}")
    c2.metric("Accuracy of 'nobody returns'", f"{r['baseline_accuracy']:.1%}")
    c3.metric("Business target (accuracy)", "95%")

    st.warning(
        "The 95% target is not reached. Only about 1 in 9 orders is returned, so "
        "even a lazy 'nobody returns' guess is right "
        f"{r['baseline_accuracy']:.1%} of the time. Accuracy alone is not a good "
        "measure here. What matters is how many real returns we catch."
    )

    st.subheader("The trade-off: pick how sensitive the flag is")
    st.caption("A lower cutoff catches more returns but wrongly flags more good orders.")

    t = pd.DataFrame(r["cutoffs"])
    table = pd.DataFrame({
        "Cutoff": t["cutoff"],
        "Orders flagged": t["flagged"],
        "Real returns caught": t["caught"].astype(str) + " of " + str(r["real_returns_in_test"]),
        "Real returns missed": t["missed"],
        "Good orders wrongly flagged": t["false_alarms"],
        "Accuracy": (t["accuracy"] * 100).round(1).astype(str) + "%",
        "Share of returns caught": (t["recall"] * 100).round(0).astype(int).astype(str) + "%",
        "Flags that were right": (t["precision"] * 100).round(0).astype(int).astype(str) + "%",
    })
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.subheader("Why not 95%, and can it get better?")
    st.write(
        "**Why it is lower:** at dispatch time we only know simple facts about the "
        "order, such as payment mode, Shield status, gift, discount and the "
        "customer's past returns. These are only weak hints about who will return "
        "something. The strongest clues (a return pickup being booked) happen "
        "*after* dispatch, so using them would be cheating."
    )
    st.write(
        "**What may improve it later:**\n"
        "- More months of order history, so the model sees more examples of returns.\n"
        "- New facts captured at order time, such as the reason a customer gives "
        "for a past return, or delivery problems.\n"
        "- Retraining regularly as new orders with known outcomes come in.\n\n"
        "We cannot promise this reaches 95%. Returns also depend on things nobody "
        "can know at dispatch, like a customer changing their mind."
    )

    with st.expander("Data problems we found and fixed"):
        st.markdown(
            "- **651 orders were listed twice** (partner-outlet re-import). We kept one copy.\n"
            "- **Two columns contained the answer** (`last_service_event_type` and "
            "`pickup_scheduled_at`). Every REVERSE_PICKUP order was a return, but this "
            "is only known after dispatch. We removed them so the results are honest.\n"
            "- **October 2025 order values were 100 times too big** (new payment gateway). We corrected them.\n"
            "- **Customer sign-up dates** were later than the order for 19% of customers, "
            "so we did not use them."
        )
    
        st.subheader("What it means in rupees")
    try:
        with open("models/business.json") as f:
            b = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        st.info("Rupee comparison not available. Run:  python src/business_eval.py")
        return

    st.write(f"We replayed the same {b['test_orders']:,} newest orders, using what "
             "really happened to each one, and counted the cost of each plan.")
    money = pd.DataFrame({
        "Plan": [x["strategy"] for x in b["rows"]],
        "Calls made": [x["calls"] for x in b["rows"]],
        "Orders held": [x["holds"] for x in b["rows"]],
        "Saving vs doing nothing (Rs)": [f"{x['saving']:,.0f}" for x in b["rows"]],
        "Saving per 700 orders (Rs)": [f"{x['saving_per_700']:,.0f}" for x in b["rows"]],
    })
    st.dataframe(money, hide_index=True, use_container_width=True)
    st.write(
        "A negative number means that plan **loses** money compared with shipping everything. "
        "Holding does not stop returns in Kestrel's policy, and about 12% of held customers "
        "cancel, so holding costs more than it saves."
    )
    st.caption(
        f"The saving from calls depends on Kestrel's pilot figure that a call prevents "
        f"{b['call_effect_assumed']:.0%} of returns. We could not measure it. "
        f"Calls still pay for themselves if they prevent at least "
        f"{b['call_breakeven_effect']:.0%}. The 20% profit lost on a cancelled sale is our "
        "assumption (Finance to confirm); at 10% or 30% Ritu's plan still loses money."
    )