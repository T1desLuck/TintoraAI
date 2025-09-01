import math
from src.utils.dlb import DynamicLossBalancer


def test_entropy_aware_modulation():
    base = {"l1": 1.0, "perc": 0.1, "adv": 0.01}
    hist = {"l1": 0.5, "perc": 0.7, "adv": 0.3}

    dlb = DynamicLossBalancer(strategy="entropy_aware")
    dlb.update(hist)

    w_low = dlb.compute_weights(base, context={"entropy": 0.0})
    w_high = dlb.compute_weights(base, context={"entropy": 1.0})

    # L1 weight should not decrease (softly increases with entropy)
    assert w_high["l1"] >= w_low["l1"]

    # perc and adv should not increase (softly decrease with entropy)
    assert w_high["perc"] <= w_low["perc"]
    assert w_high["adv"] <= w_low["adv"]

    # weights remain finite and non-negative
    for k in base.keys():
        for w in (w_low[k], w_high[k]):
            assert math.isfinite(float(w)) and float(w) >= 0.0
