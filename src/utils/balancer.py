from __future__ import annotations
from typing import Dict, Any, Optional

class LossBalancer:
    """
    Управляет весами лоссов на основе статического учебного плана (curriculum),
    заданного в конфиге. Определяет, какие лоссы активны и с какими весами на
    текущей эпохе. Необязательный DynamicLossBalancer может дополнительно
    адаптировать веса.
    """

    def __init__(self, curriculum_cfg: Dict[str, Any], dlb: Optional[Any] = None):
        self.curriculum = sorted(curriculum_cfg.items())
        self.dlb = dlb
        self.current_phase_losses = {}
        self.active_phase = {}
        self.phase_num = -1

    def get_weights(self, epoch: int) -> Dict[str, float]:
        """
        Вычисляет и возвращает веса лоссов для текущей эпохи.
        """
        # Определяем активную фазу по номеру эпохи
        active_phase = self.curriculum[0]
        for phase in self.curriculum:
            if epoch >= phase[0]:
                active_phase = phase
            else:
                break
        
        self.active_phase = active_phase[1]
        self.phase_num = self.active_phase.get('phase_num', -1)
        
        self.current_phase_losses = self.active_phase.get('losses', {})

        # Если включена динамическая балансировка, обновим её веса
        if self.dlb and 'dlb_weights' in self.active_phase:
            self.dlb.update_weights(self.active_phase['dlb_weights'])

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
