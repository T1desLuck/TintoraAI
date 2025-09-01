# Конфигурация и новые опции

> Навигация: [README.md](README.md) • [INSTALL.md](INSTALL.md) • [TRAINING.md](TRAINING.md) • [TEST.md](TEST.md)

Этот документ описывает важные настройки, добавленные в последних обновлениях, и то, как их использовать через YAML-конфигурации.

## Важные настройки (новые опции)

```yaml
training:
  dlb:
    enabled: true
    strategy: ema            # ema | entropy_aware

gan:
  enabled: true
  loss_type: hinge          # hinge | bce
  label_smooth: 0.0         # для bce: 0.0..0.1 (например 0.05)
  r1_gamma: 0.0             # R1 регуляризация дискриминатора на real (например 10.0)

validation:
  ciede2000: false          # валидация по ΔE2000; при true доступен выбор best_metric=ciede2000

inference:
  pad_divisor: 32
  tta:
    enabled: false
    flip: true
    scales: [1.0]
```

### Пояснения
- **DLB strategy** `entropy_aware` — мягкая модуляция весов лоссов в зависимости от энтропии/неопределённости (стабильные лоссы ↑, агрессивные ↓ при высокой энтропии). По умолчанию остаётся классический EMA‑балансировщик.
- **GAN**: при `loss_type: bce` доступно `label_smooth` (смягчение меток); `r1_gamma` включает R1‑штраф на реальных изображениях (стабилизация обучения дискриминатора).
- **Validation**: помимо SSIM/DISTS добавлена метрика **CIEDE2000 (ΔE2000)**. Можно выбирать сохранение `best` по `ciede2000` в `checkpointing.best_metric`.
- **Inference**: TTA (flip+scales) и тайловый инференс теперь с косинусным окном для редукции швов.

## Где включать
Все параметры задаются в `configs/default.yaml` и/или через CLI флаги, см. также:
- `TRAINING.md` — запуск обучения и управление лоссами/фазами
- `TEST.md` — сценарии тестирования
- `README.md` — быстрый старт и обзор
