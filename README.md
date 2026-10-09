# Kestrel Home: Returns Risk

A small, free, local system that scores each order for return risk before dispatch
and recommends **Ship**, **Call** or **Hold**. It uses no paid API, no GPU and no internet.

## What is in it

| Folder | Purpose |
|---|---|
| `src/` | The thinking: data cleaning (`prep.py`), business rules (`rules.py`), reasons (`reasons.py`), customer history lookup (`history.py`), training and evaluation scripts |
| `api/` | The endpoint (FastAPI): `POST /predict` takes one order as JSON |
| `app/` | The screen (Streamlit): Batch Predictions, Check an Order, Model Evidence |
| `models/` | Trained model (`return_model.pkl`) and test results (`evidence.json`) |
| `predictions.csv` | Scores for every order in `test_unlabelled.csv` |

## Requirements

- Python 3.10 or newer
- The Kestrel data files placed in a `data/` folder (not included, because the data is
  private and must not be published): `train.csv`, `test_unlabelled.csv`,
  `customers.csv`, `products.csv`, `sample_submission.csv`

## Setup (Windows)

Run every command from the `kestrel-returns` folder.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Mac/Linux, activate with `source .venv/bin/activate`.

## Run it

The trained model is already in `models/`, so you can start straight away.
Use **two terminals**, both with the environment activated.

**Terminal 1: the endpoint**
```
uvicorn server:app --app-dir api --port 8000
```
Check it at http://localhost:8000/docs (try `POST /predict`) or http://localhost:8000/health.

**Terminal 2: the screen**
```
streamlit run app/main.py
```

If the endpoint is not running, the "Check an Order" tab shows a polite message
instead of crashing. The other two tabs do not need the endpoint.

## Using the screen

1. **Batch Predictions**: upload `test_unlabelled.csv`, check the summary, download `predictions.csv`.
   The file is checked first: missing columns, unknown products or customers and bad numbers are reported.
2. **Check an Order**: choose the customer, product, payment mode and so on, then click **Predict**.
   You get the return probability, risk level, reasons and the recommended action.
   The customer's past orders and returns are looked up automatically from the customer ID.
3. **Model Evidence**: how the model was tested, how often it is wrong, and why 95% is not reached.

## Using the endpoint

`POST /predict` with one order as JSON. Past orders and past returns are optional and
are looked up from the customer ID if left out. The reply contains:
`return_probability`, `risk`, `action`, `action_text`, `reasons_raising_risk`,
`reasons_lowering_risk`, `customer_history`, `expected_cost_rs`, `saving_vs_ship_rs`.

## Re-creating the model (optional)

```
python src/evaluate.py          # 80/20 time-based test, writes models/evidence.json
python src/make_predictions.py  # trains on all old orders, writes models/return_model.pkl and predictions.csv
python src/business_eval.py     # rupee comparison of Ship / Ritu's hold / Call
```

## How it works

1. **Model:** Logistic Regression (scikit-learn). It outputs the probability that an order is returned.
2. **Reasons:** read from the model's own weights and written as plain sentences.
3. **Business rules** (`src/rules.py`): compare the average cost of Ship, Call and Hold and pick the cheapest.
   Shield members are never held. The cost figures sit at the top of that file and can be changed
   without retraining.

## Data decisions

- **Duplicates:** 651 partner-feed re-imports removed (the CRM copy is kept).
- **Leakage:** `last_service_event_type` and `pickup_scheduled_at` were dropped. They are
  recorded after a return starts, and test orders have no pickups at all.
- **October 2025 order values** were 100 times too big (new payment gateway). They are corrected.
- **`signup_date`** is not used. It is later than the order date for about 19% of orders.
- **Pincode 000000** is treated as "no address captured".

## Results (honest)

- Tested on the newest 20% of old orders (2026-04-02 to 2026-06-30), which the model never saw.
- ROC-AUC about 0.78. Accuracy 80% to 90% depending on the cutoff.
- "Nobody returns" already scores 88.5% accuracy, so the **95% target is not reached**.
- On that test, Ritu's "hold what is flagged" rule would have cost about Rs 75,000 more than
  shipping. Calling orders at 11% risk or higher would have saved about Rs 36,000 over three
  months (about Rs 12,000 per 700 orders).

## Limitations

- The 35% call effect and the 12% hold cancellation rate come from Kestrel's policy and were not measured here.
- The 20% margin lost on a cancelled sale is our assumption, not from the pack.
- Customer history is looked up from the exported files, not a live order database.
- The endpoint trusts the `order_value_inr` it is given.
- Only the main paths were tested, not every bad-input case.

## Cost and privacy

No paid calls, so one prediction costs Rs 0 and a month of about 700 orders costs Rs 0.
Never upload the `data/` folder to a public place. It is excluded by `.gitignore`.