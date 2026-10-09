
# Submission form: Kestrel Home Returns Risk (Variant A)

**Name:** Prem Dave  **Date:** 9 October 2026

## GitHub repo URL

https://github.com/prem-d-25/kestrel-returns

(The repo contains code, the saved model, `predictions.csv`, README and memo. The `data/` folder is excluded because Kestrel's policy says customer data must not be published.)

## What did you build, and what business decision does it support? State the number and the rupees.

I built a returns-risk system that runs free and locally: a Logistic Regression model gives each order a return probability, a separate rules layer (`src/rules.py`) turns it into Ship, Call or Hold using Kestrel's own costs, a FastAPI endpoint (`POST /predict`) returns the answer plus plain-language reasons, and a Streamlit screen has three tabs (Batch Predictions, Check an Order, Model Evidence).

**Decision it supports:** do not hold flagged orders; make a Rs 45 confirmation call on orders with about 11% return risk or more (about 3 in 10 orders) and ship the rest. Never hold Shield customers.

**The number:** 95% accuracy is not honestly reachable. "Nobody returns" already scores 88.5%. Our model has ROC-AUC 0.78 and about 89% accuracy at a strict cutoff.

**The rupees** (replay of the newest 2,101 orders using real outcomes, return = Rs 1,150, call = Rs 45):

- Ritu's plan (hold orders scoring 20%+): **Rs 75,234 worse** than shipping everything.
- Our plan (call at about 11%+ risk, 657 calls, 0 holds): **Rs 36,445 better**, about **Rs 12,000 per 700 orders** (one month).

## What score do you expect predictions.csv to get on the hidden outcomes, on which metric, and why that metric? How did you estimate it?

**Expected: ROC-AUC about 0.78 (plausible range 0.74 to 0.81).**

**Why this metric:** the file is a ranking (a score, higher = more likely returned), not a yes/no. ROC-AUC measures how well the scores put real returns above non-returns, and does not depend on a cutoff. Accuracy would be misleading because only 11.4% of orders are returned. If Kestrel scores with accuracy at a cutoff, I expect 84% to 90%, below the 95% promise.

**How I estimated it:** a time-based 80/20 split. The model studied the oldest 8,403 orders (2025-04-01 to 2026-04-01) and was tested on the newest 2,101 (2026-04-02 to 2026-06-30), which it never saw. That gave ROC-AUC 0.783 and PR-AUC 0.41. A time split matches the real job (predict newer orders from older ones). The final model is trained on all 10,504 de-duplicated orders.

**Why the range is not tighter:** the hidden orders are July to September 2026, further from the training data than my test window, so some drift is possible. Also, the hidden set has about 2,000 orders and about 230 returns, so the score will move by a few points from chance alone.

## How do you know it works? How you validated, on what split, error rate, and the kind of case it gets wrong.

Validation was the time-based split above. Results on the 2,101 unseen orders (242 really returned):

| Cutoff | Accuracy | Returns caught   | Good orders wrongly flagged    |
| ------ | -------- | ---------------- | ------------------------------ |
| 0.5    | 89.6%    | 35 of 242 (14%)  | 12                             |
| 0.3    | 87.8%    | 79 of 242 (33%)  | 93                             |
| 0.2    | 83.9%    | 108 of 242 (45%) | 204 (about 11% of good orders) |
| 0.15   | 80.3%    | 138 of 242 (57%) | 310                            |

It is a useful ranking tool, not a precise predictor: at the 0.2 cutoff only 35% of flagged orders really come back, and it misses more than half of all returns. All of this is on the Model Evidence tab.

**Cases it gets wrong:** it relies on payment mode (COD returns 18.8% vs UPI 7.6%), past returns, Shield status and gift orders. So it will miss returns from customers who look ordinary on those clues (for example prepaid, no past returns), and it will over-flag COD or repeat-returner customers who keep the order. I did not run a separate error analysis by segment; this comes from the model's weights and rates by category.

I also tested the endpoint with valid orders and several bad inputs (unknown product, unknown customer, wrong order value, impossible history). I did not run every bad-input case.

## Did you change, narrow, or push back on the client's ask? What, when, and why.

Yes, four things:

1. **95% accuracy:** I did not promise it. It cannot be reached honestly from information known at dispatch, and "nobody returns" already scores 88.5%. The Evidence tab says so openly.
2. **"Flag and hold":** I replaced holding with confirmation calls. Holding loses Rs 75,234 on the replay: about 12% of held customers cancel and the policy says nothing about holding preventing returns. Meenal's email and the pilot pointed to calls.
3. **Rs 600 per return:** I used Finance's Rs 1,150, which Farhan said is the reported figure.
4. **"We'll deal with Shield later":** I made Shield customers a protected group now (never held), because holding them risks the highest-value segment and they are 22% of orders.

## What is wrong with what you are handing us, or with the data you handed us?

**Data:**

- 651 orders appeared twice (every `partner_feed` row duplicated a `crm` row). I kept the `crm` copy.
- `last_service_event_type` and `pickup_scheduled_at` leak the answer: every REVERSE_PICKUP order is a return, and test has no pickups at all. I dropped both. This is why accuracy is not 95%.
- October 2025 order values were exactly 100 times too big (paise, new payment gateway). I corrected them.
- `signup_date` is after the order date for 19% of orders, so I did not use it. Shield status also comes from the export-day customer file, so I cannot rule out that some customers joined Shield after the order.
- Pincode `000000` appears at about 8% in every channel, not just walk-ins. I treat it as "no address captured".
- Meenal's "most returns are Shield" is not supported: Shield is about 36% of returns in the training file.
- I did not check when `customer_prior_orders` and `customer_prior_returns` were calculated. I only checked that returns never exceed orders. If they were computed at export day, they could leak.
- The UTC/IST timestamp issue did not affect the model because the columns that carried it were dropped.

**What I'm handing you:**

- The Rs 45, Rs 1,150, 35% call effect and 12% cancellation figures come from the policy. I could not measure them.
- The 20% profit lost on a cancelled sale is my assumption (not in the pack). Ritu's plan loses money at 10%, 20% and 30%.
- Customer history in the endpoint is looked up from the exported files, not a live database, and the newest orders count as "not returned" because their outcome is unknown.
- The "HIGH" risk label at 0.30 is a label choice, not tuned.
- The rupee replay uses the same 2,101 orders as the accuracy test. The 11% call cutoff comes from policy arithmetic, not tuning on those orders, but the saving is still one sample.
- Not every bad input was tested; the README says so.
- Some editor warnings show in `server.py` and `batch_tab.py` because of the `sys.path` import line. The app runs.

## What does one prediction cost, and what would a month cost at about 700 orders?

**Rs 0 per prediction.** No paid API is used. The model is a small file on the laptop running on CPU, with no internet needed.

Arithmetic: 700 orders x Rs 0 = **Rs 0 per month**. No usage-based bill exists, which addresses Farhan's concern. The only running cost is the Rs 45 confirmation calls the business chooses to make: about 31% of 700 = about 217 calls x Rs 45 = about Rs 9,800 a month, against about Rs 22,000 of returns avoided on the replay (net saving about Rs 12,000). That is a business cost, not a model cost.

## What did you deliberately leave out, and why that rather than something else?

- **SHAP and other models:** the Logistic Regression weights already give readable reasons, and it scored higher than HistGradientBoosting (0.78 vs 0.75). More complexity would not have helped.
- **A hold strategy:** it loses money on the evidence, so building it out would be wasted effort.
- **Confusion matrix page:** the cutoff table shows the same information more simply.
- **Delivery-note text analysis, time-zone repair, pincode/geography features:** the notes looked like random templates, and time-zone repair only matters for the columns I dropped.
- **Live database, login, automatic retraining:** out of scope for a working demo in 48 hours. The README says how to retrain.
- **A full automated test suite:** I tested the main paths by hand. I chose to spend the time on finding leakage and on the rupee analysis instead.

## Anything you built or found that nobody asked for?

- Ritu's plan loses about Rs 75,000 on the replay, while a call plan saves about Rs 36,000.
- Calls stay profitable if they prevent at least 16% of returns (the pilot says 35%).
- The October paise error, the leaky columns and the duplicates, plus the claim that most returns are Shield being incorrect.
- The endpoint rejects an order value that does not match price x quantity x discount, so the October-type error cannot silently reach the model.
- The Check an Order screen looks up customer history from the customer ID, so employees do not need to know it.
- File checks on the Batch tab (missing columns, unknown products, bad numbers) and a polite message when the endpoint is not running.

## What did you use AI for? Tools, models, where they helped, where they misled you, what you threw away. Link your three-minute screen recording.

**Tools:** ChatGPT (understanding the brief and the first plan), Claude (data analysis, code, README, memo, this form), and Antigravity IDE (editing files). **Cost: Rs 0**, all on free plans or my student subscription.

**Where they helped:** turning the emails and policy into a plan, finding the leakage, duplicates and October error by running an inspection script, writing the code, and explaining each step in simple language.

**Where they misled me or got it wrong:**

- Claude's edit instructions for `api/server.py` put code in the wrong place twice, causing two crashes. I fixed them by pasting the whole file and re-running.
- Claude first gave a wrong call break-even figure (14%), and corrected it to 16%.
- The first version of the reasons showed meaningless items ("warranty months", "model tier 02"). I hid them.
- ChatGPT's plan assumed things (use SHAP, try several models, whether to use `source`) before looking at the data. I checked the data first.

**What I threw away:** HistGradientBoosting model, the leaky and unreliable columns (`last_service_event_type`, `pickup_scheduled_at`, `signup_date`), the hold-flagged-orders plan, SHAP, and the confusion-matrix page.

**Screen recording link:** https://drive.google.com/drive/folders/1xmatgk__IgthV5M-O95cS8aOS_7vEDvZ?usp=sharing (the recording is in this Drive folder)

## Your Public Google Drive Link

https://drive.google.com/drive/folders/1xmatgk__IgthV5M-O95cS8aOS_7vEDvZ?usp=sharing (folder with the recording and the memo)

## Someone picks this up on Monday and you are unreachable. The three things they need to know.

1. **How to run it:** follow `README.md`. Put the data files in `data/` (not in the repo), activate the environment, start `uvicorn server:app --app-dir api --port 8000`, then `streamlit run app/main.py`. The trained model is already in `models/`. To retrain, run `python src/evaluate.py`, `python src/make_predictions.py` and `python src/business_eval.py`.
2. **The money numbers are Kestrel's claims, not measured.** The whole Call recommendation rests on "a call prevents 35% of returns" and the 12% hold cancellation. The two-week call trial with an uncalled control group is the first thing to do. The cost figures live at the top of `src/rules.py`; changing them needs no retraining.
3. **Do not promise 95% accuracy, and do not add the dropped columns back.** `last_service_event_type` and `pickup_scheduled_at` would make the numbers look great but are only known after a return starts. The customer-history lookup in the endpoint reads exported files, so it must be pointed at the live order database before real use.
