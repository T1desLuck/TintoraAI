from __future__ import annotations
from typing import Dict


class DynamicLossBalancer:
    """
    Простая динамическая балансировка лоссов на основе EMA их величин.
    Итоговый вес для i-го лосса: w_i = base_lambda_i * (m_ref / (m_i + eps))
    где m_i — EMA по модулю лосса, m_ref — среднее по всем m_i.
    Это выравнивает масштаб вкладов без изменения нулевых лямбд.

    Дополнительно поддерживается стратегия 'entropy_aware':
    при повышенной неопределённости (контекстный ключ 'entropy') слегка
    увеличивать вклад стабильных терминов (например, L1) и снижать агрессивные (adv/perc).
    Масштабирование мягкое и ограниченное, чтобы не ломать обучение.
    """

    def __init__(
        self, decay: float = 0.9, eps: float = 1e-8, strategy: str | None = None
    ):
        self.decay = float(decay)
        self.eps = float(eps)
        self.strategy = (strategy or "ema").lower()
        self.ema: Dict[str, float] = {}

    def update(self, values: Dict[str, float]):
        d = self.decay
        for k, v in values.items():
            if v is None:
                continue
            m = self.ema.get(k, v)
            m = d * m + (1.0 - d) * float(abs(v))
            self.ema[k] = m

    def compute_weights(
        self, base: Dict[str, float], context: Dict[str, float] | None = None
    ) -> Dict[str, float]:
        # Если ещё нет EMA, вернуть базовые веса
        if not self.ema:
            return dict(base)
        vals = [v for v in self.ema.values() if v is not None]
        if not vals:
            return dict(base)
        m_ref = sum(vals) / len(vals)
        out = {}
        for k, base_lam in base.items():
            if base_lam <= 0.0:
                out[k] = 0.0
                continue
            mi = self.ema.get(k, None)
            if mi is None:
                out[k] = base_lam
            else:
                out[k] = base_lam * (m_ref / (mi + self.eps))
        # Энтропийно-осведомлённая модификация (мягкая)
        if self.strategy == "entropy_aware" and context is not None:
            ent = (
                float(max(0.0, context.get("entropy", 0.0)))
                if isinstance(context, dict)
                else 0.0
            )
            # Нормируем эвристически: ent_scaled ~ [0..1] для типичных значений
            ent_scaled = min(1.0, ent / 1.0)
            if ent_scaled > 0.0:
                # Усиливаем L1 до +15% при высокой энтропии, ослабляем adv/perc до -20%
                if "l1" in out:
                    out["l1"] *= 1.0 + 0.15 * ent_scaled
                for k in ("adv", "perc"):
                    if k in out:
                        out[k] *= 1.0 - 0.2 * ent_scaled
        return out
