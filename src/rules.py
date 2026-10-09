# All the money numbers live here, so they are easy to change.
RETURN_COST = 1150      # Rs per return (policy section 4, Finance)
CALL_COST = 45          # Rs per completed call
CALL_PREVENTS = 0.35    # share of returns a call stops (spring pilot)
HOLD_CANCEL = 0.12      # share of held orders the customer cancels
MARGIN = 0.20           # ASSUMPTION: profit lost when a sale is cancelled (not in the pack)

CALL_BREAKEVEN = CALL_COST / (CALL_PREVENTS * RETURN_COST)   # about 0.112
HIGH_RISK = 0.30        # label only: at 0.30 about 46% of flagged orders really came back


def costs(p, order_value):
    """Average cost of each choice. Works for one order or whole arrays."""
    return {
        "ship": p * RETURN_COST,
        "call": CALL_COST + p * (1 - CALL_PREVENTS) * RETURN_COST,
        "hold": HOLD_CANCEL * MARGIN * order_value
                + (1 - HOLD_CANCEL) * p * RETURN_COST,
    }


def risk_level(p):
    if p < CALL_BREAKEVEN:
        return "LOW"
    return "MEDIUM" if p < HIGH_RISK else "HIGH"


def recommend(p, order_value, shield):
    c = costs(p, order_value)
    options = ["ship", "call"] if shield else ["ship", "call", "hold"]
    action = min(options, key=lambda a: c[a])
    return {
        "risk": risk_level(p),
        "action": action,
        "expected_cost": {k: round(v, 2) for k, v in c.items()},
        "saving_vs_ship": round(c["ship"] - c[action], 2),
    }