from src.utils import DynamicLossBalancer


def test_dlb_weights_balance():
    dlb = DynamicLossBalancer(decay=0.0)
    base = {"a": 1.0, "b": 1.0}
    # потери различаются на порядок
    dlb.update({"a": 1.0, "b": 10.0})
    w = dlb.compute_weights(base)
    # ожидание: вес меньшего лосса повысится, большего — понизится
    assert w["a"] > w["b"]
    # базовые положительны
    assert w["a"] > 0 and w["b"] > 0
