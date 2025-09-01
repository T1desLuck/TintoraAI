from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List


class LossBalancer:
    """
    Управляет весами лоссов на основе статического учебного плана (curriculum),
    заданного в конфиге. Определяет, какие лоссы активны и с какими весами на
    текущей эпохе. Необязательный DynamicLossBalancer может дополнительно
    адаптировать веса.
    """

    def __init__(self, curriculum_cfg: Dict[Any, Any], dlb: Optional[Any] = None):
        # normalize keys to int and values to dicts
        norm: List[Tuple[int, Dict[str, float] | Dict[str, Any]]] = []
        for k, v in curriculum_cfg.items():
            try:
                ik = int(k)
            except Exception:
                # skip non-int-convertible keys
                continue
            norm.append((ik, v))
        # sort by starting epoch
        self.curriculum: List[Tuple[int, Dict[str, Any]]] = sorted(
            [(k, dict(v)) for k, v in norm], key=lambda x: x[0]
        )
        self.dlb: Optional[Any] = dlb
        self.current_phase_losses: Dict[str, float] = {}
        self.active_phase: Dict[str, Any] = {}
        self.phase_num: int = -1

    def get_weights(self, epoch: int) -> Dict[str, float]:
        """
        Вычисляет и возвращает веса лоссов для текущей эпохи.
        """
        # Определяем активную фазу по номеру эпохи
        active_phase = self.curriculum[0]
        for phase in self.curriculum:
            start_epoch = int(phase[0])
            if epoch >= start_epoch:
                active_phase = phase
            else:
                break

        self.active_phase = active_phase[1]
        self.phase_num = self.active_phase.get("phase_num", -1)

        self.current_phase_losses = self.active_phase.get("losses", {})

        # Если включена динамическая балансировка, обновим её веса
        if self.dlb and "dlb_weights" in self.active_phase:
            self.dlb.update_weights(self.active_phase["dlb_weights"])

        # Если передан динамический балансировщик — скорректируем базовые веса
        if self.dlb:
            return self.dlb.compute_weights(self.current_phase_losses)

        return self.current_phase_losses

    def update_ema(self, loss_values: Dict[str, float]):
        """
        Обновляет EMA в динамическом балансировщике (если задан).
        """
        if self.dlb:
            self.dlb.update(loss_values)
