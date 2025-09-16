# ⚙️ Конфигурация и новые опции

> Навигация: [README.md](README.md) • [INSTALL.md](INSTALL.md) • [TRAINING.md](TRAINING.md) • [TEST.md](TEST.md)

## 🔗 Быстрые ссылки

- Обзор: [`README.md`](README.md)
- Обучение: [`TRAINING.md`](TRAINING.md)
- Тестирование: [`TEST.md`](TEST.md)

## 📚 Содержание

- [Важные настройки (новые опции)](#важные-настройки-новые-опции)
- [Adapter/LoRA — конфигурация](#adapterlora-—-конфигурация)
- [FAQ по слиянию](#faq-по-слиянию)
- [Примеры типовых конфигов](#примеры-типовых-конфигов)
- [Где включать](#где-включать)

Этот документ описывает важные настройки, добавленные в последних обновлениях, и то, как их использовать через YAML‑конфигурации.

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
- **GAN** — при `loss_type: bce` доступно `label_smooth` (смягчение меток); `r1_gamma` включает R1‑штраф на real (стабилизация обучения дискриминатора).
- **Validation** — помимо SSIM/DISTS добавлена метрика **CIEDE2000 (ΔE2000)**. Можно выбирать сохранение `best` по `ciede2000` в `checkpointing.best_metric`.
- **Inference** — TTA (flip+scales) и тайловый инференс с косинусным окном для редукции швов.

## Adapter/LoRA — конфигурация

Секция слияния весов и параметры LoRA размещаются в `configs/default.yaml`.

```yaml
merging:
  weights:
    base: 1.0
    adapter: 0.8
    lora:
      default: 0.5
      faces: 0.5
      animals: 0.4
      transport: 0.6

adapters:
  lora_name: "faces"  # имя для файла lora_faces_YYYY-MM-DD.pth
  rank: 8              # rank для LoRA-факторов (8–16 обычно достаточно)
```

Пояснения:
- `merging.weights.*` — коэффициенты взвешенного слияния при инференсе (порядок: base → adapter → lora). Если файлов нет — слияние не выполняется.
- `adapters.lora_name` — имя для формирования файла LoRA при обучении.
- `adapters.rank` — ранг LoRA (чем выше, тем выразительнее и дороже по памяти).

Каталоги чекпоинтов:
- Adapter: `checkpoints/adapters/adapter.pth`
- LoRA: `checkpoints/lora/lora_<lora_name>_<YYYY-MM-DD>.pth`

### Примечание про TIES‑Merging (рекомендация)
Если одновременно используется много LoRA (например, >5) и вы наблюдаете конфликты «перекраски» между тематиками, рассмотрите подход TIES‑Merging (Trim‑Elect‑Sum):

- Смысл: усечь слабые или конфликтные обновления (Trim), отобрать наиболее согласованные (Elect), просуммировать оставшиеся (Sum).
- Практика: начать с уменьшения весов отдельных LoRA в `merging.weights.lora.*`, затем экспериментально применять TIES на уровне подготовки LoRA или пользовательского слияния.
- Текущее поведение проекта — взвешенная сумма. TIES рекомендуется как следующий шаг при масштабировании числа LoRA.

## Где включать
Все параметры задаются в `configs/default.yaml` и/или через CLI флаги, см. также:
- `TRAINING.md` — запуск обучения и управление лоссами/фазами
- `TEST.md` — сценарии тестирования
- `README.md` — быстрый старт и обзор

---

## ❓ FAQ по слиянию

- Какой порядок применения весов на инференсе?
  - base → adapter → lora. Adapter применяется ко всем найденным файлам в `checkpoints/adapters/` одинаковым весом, LoRA — для каждого файла в `checkpoints/lora/` с собственным весом (по имени или `default`).

- Автоматически ли это происходит?
  - Да. Обёртка в `sitecustomize.py` перехватывает `load_state_dict` и вызывает `src/models/merge_all.scan_and_merge()`.

- Как задать вес конкретного LoRA?
  - В секции `merging.weights.lora`: ключ — логическое имя, которое берётся из файла `lora_<name>_*.pth` → `<name>`. Пример:

```yaml
merging:
  weights:
    adapter: 0.8
    lora:
      default: 0.5
      face: 0.7
      animals: 0.4
```

- Как временно отключить Adapter или LoRA?
  - Поставьте соответствующий вес в `0.0`, либо уберите файл из папки чекпоинтов.

- Что если нет базового чекпоинта?
  - Если `--checkpoint` и `paths.checkpoints/latest.pth` отсутствуют, то `inference.py` загружает случайные веса и не вызывает `load_state_dict`, поэтому автослияние не выполнится. Рекомендуется иметь `latest.pth`.

---

## 🔧 Примеры типовых конфигов

Минимальный конфиг для CPU‑обучения (фрагменты):

```yaml
runtime:
  device: cpu
  amp: false

training:
  batch_size: 2
  epochs: 4
  image_size: 256

loss:
  lambda_l1: 1.0
  lambda_perc: 0.0
  lambda_adv: 0.0
```

Включить GAN только на финальной фазе (пример куррикулума):

```yaml
gan:
  enabled: true
  loss_type: hinge

training:
  curriculum:
    enabled: true
    phases:
      - { until: 2, phase: 0 }
      - { until: 4, phase: 2 }
      - { until: 6, phase: 3 }
      - { until: 12, phase: 4 }  # финальная GAN фаза
```

Слияние нескольких LoRA с разными весами:

```yaml
merging:
  weights:
    adapter: 0.8
    lora:
      default: 0.4
      face: 0.7
      landscape: 0.5
```
