# Тестирование TintoraAI

<div align="center">

[![PyTest](https://img.shields.io/badge/Tests-PyTest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Status](https://img.shields.io/badge/Статус-Все%20тесты%20проходят-4CAF50?style=for-the-badge)](#)

</div>

Этот документ описывает, как запускать, понимать и расширять тесты проекта TintoraAI. Все команды и сообщения ориентированы на русскоязычных пользователей.

> Навигация: [README.md](README.md) • [INSTALL.md](INSTALL.md) • [TRAINING.md](TRAINING.md) • [CONFIGURATION.md](CONFIGURATION.md)

- Репозиторий: `TintoraAI`
- Лаунчер тестов: `test_project.py` (корень репозитория)
- Каталог тестов: `tests/`

## Содержание
- [Требования](#требования)
- [Быстрый старт](#быстрый-старт)
- [Лаунчер тестов](#лаунчер-тестов)
- [Цели (алиасы) запуска](#цели-алиасы-запуска)
- [Точечные запуски PyTest](#точечные-запуски-pytest)
- [Что проверяют тестовые файлы](#что-проверяют-тестовые-файлы)
- [Примеры сценариев](#примеры-сценариев)
- [Подсказки и устранение неполадок](#подсказки-и-устранение-неполадок)
- [CI и локальные проверки](#ci-и-локальные-проверки)

## Тесты Adapter/LoRA (новые)

Ниже — рекомендации по проверкам новых модулей Adapter и LoRA.

1. Без дополнительных весов (идентичность базовой модели):
   ```bash
   python test_project.py run inference
   ```
   Ожидаем, что все тесты инференса проходят без появления сторонних весов в `checkpoints/adapters` и `checkpoints/lora`.

2. Синтетические мини‑веса (корректное применение):
   - Создайте небольшой Adapter или LoRA с несколькими параметрами (можно обучить 1–2 эпохи на паре изображений).
   - Поместите файлы в `checkpoints/adapters/` и/или `checkpoints/lora/`.
   - Проверьте инференс:
     ```bash
     python test_project.py run inference
     ```
   - Ожидаем отсутствие ошибок и небольшое, но детерминированное изменение выходов.

3. Несколько LoRA (взвешенное слияние):
   - Поместите 2–3 LoRA в `checkpoints/lora/` с разными именами.
   - Настройте веса в `configs/default.yaml` → `merging.weights.lora.{default|name}`.
   - Запустите инференс‑тесты:
     ```bash
     python test_project.py run inference
     ```
   - Ожидаем, что паспортизированные тесты проходят, а слияние не вызывает артефактов.

Точечные прогоны:
```bash
python test_project.py run tests/test_inference.py::test_colorize_tiled
python test_project.py run tests/test_forward.py::test_forward_shapes
```

## Требования
- Python 3.9+
- Установленный `pytest`

Установка pytest:
```bash
python -m pip install pytest
```
Если пакет отсутствует, `test_project.py` подскажет, как его установить.

## Быстрый старт

Запуск справки и списка целей:
```bash
python test_project.py help
python test_project.py list
```
Запуск всех тестов:
```bash
python test_project.py run all
```
Запуск группы тестов (пример — модули):
```bash
python test_project.py run modules
```
Запуск точечного теста:
```bash
python test_project.py run modules::test_crb
```

## Лаунчер тестов
Используйте `test_project.py` для удобного запуска и навигации по тестам.

Команды:
- `help` — подробная справка по использованию лаунчера.
- `list` — доступные цели (алиасы) с краткими описаниями.
- `run <цель>` — запуск тестов по выбранной цели. Поддерживаются точечные тесты с `::`.

Дополнительно:
- `preview [опции]` — сохраняет предпросмотр подготовки изображения без запуска PyTest. Читает `configs/default.yaml`. См. раздел «Предпросмотр препроцессинга» ниже для примера.

В Jupyter/Colab:
```python
from test_project import run, show_help, list_targets
show_help()
list_targets()
run("modules")
run("modules::test_crb")
```

## Цели (алиасы) запуска
- `all` — весь набор тестов (`tests/`).
- `dlb` — `tests/test_dlb.py`: DynamicLossBalancer, корректность весов и их балансировки.
- `forward` — `tests/test_forward.py`: быстрый прогон `TintoraAI` вперёд, формы и ключи выходов.
- `inference` — `tests/test_inference.py`: функции инференса (одиночный/тайловый), диапазоны значений.
- `losses` — `tests/test_losses.py`: продвинутые лоссы, корректность вычислений и финитность значений.
- `modules` — `tests/test_modules.py`: отдельные модули — backbone/heads/CRB/decoder, соответствие форм.

## Точечные запуски PyTest
Можно запускать конкретные тесты или группы внутри файла:
- `modules::test_crb` — тест блока цветового рассуждения CRB.
- `modules::test_decoder_unetpp` — тест декодера U‑Net++.
- Любой `test_*` внутри `tests/test_modules.py` или других файлов.

Синтаксис общий для PyTest: `<файл или алиас>::<имя_теста>`.

## Предпросмотр препроцессинга
Скрипты предпросмотра применяют те же шаги, что и датасеты (`advanced/simple`):
- Геометрия: `training.geometry.train_mode|val_mode`
- Размер: `training.image_size`
- Интерполяция: `training.resize.filter` (lanczos/bicubic/bilinear/nearest)
- Аугментации: `training.aug.flip_p`, `training.aug.crop_scale`, `training.aug.ab_jitter`, мелкие L‑дефекты при `--enable_defects` читаются из `training.aug.defects`

Пример (лаунчер):
```bash
python test_project.py preview --input assets/color.jpg --save_L --enable_defects \
  --output experiments/exp_default/preview_from_launcher
```

Прямой вызов скрипта:
```bash
python scripts/preview_preprocessing.py --input assets/color.jpg --save_L --enable_defects \
  --config configs/default.yaml --output experiments/exp_default/preview_preprocessing
```

## Что проверяют тестовые файлы
- `tests/test_dlb.py`
  - Проверяет `DynamicLossBalancer` из `src/utils/dlb.py`.
  - Балансировку весов потерь, положительность весов, реакцию на разные масштабы лоссов.

- `tests/test_dlb_entropy.py`
  - Проверяет энтропийно-осведомлённую стратегию DLB (`strategy: entropy_aware`).
  - Корректность модуляции весов и численную стабильность.

- `tests/test_forward.py`
  - Быстрый прогон основной модели `TintoraAI` (`src/models/tintoraai.py`).
  - Проверка форм выходных тензоров и наличия ключевых предсказаний.

- `tests/test_inference.py`
  - Проверка `colorize_single` и `colorize_tiled` из `src/inference.py`.
  - Достоверность диапазонов значений, корректность сборки результата.

- `tests/test_losses.py`
  - Проверяет доступные лоссы (в проекте определён `PhotometricSmoothnessLoss` в `src/losses/advanced.py`).
  - Корректность вычислений и финитность значений на игрушечных данных.

- `tests/test_gan_r1.py`
  - Проверяет GAN‑лосс (BCE с label smoothing) и вычисление R1‑штрафа на real.
  - Цель — отсутствие рантайм‑ошибок и корректность «проводки» параметров.

- `tests/test_modules.py`
  - Набор модульных тестов: backbones, головы глубины/освещённости, CRB, UNetPP‑decoder.
  - Согласование пространственных размеров и каналов, релевантность параметров.

- `tests/test_model_api.py`
  - API модели: формы выходов, dtype/device поведение, AMP против FP32, эквивалентность TTA/tiling в простых сценах.

- `tests/test_metrics.py`
  - Санити‑проверки SSIM/DISTS/CIEDE2000, монотонность/инвариантности и численная устойчивость.

- `tests/test_config_validation.py`
  - Схема и типы в `configs/default.yaml` через `src/utils/config.py`.
  - Ошибка на неизвестные верхнеуровневые ключи, корректность CLI‑override.

- `tests/test_checkpoint_io.py`
  - Политики latest/best имен файлов (`latest.pth`, `best_*.pth`) и наличие EMA.
  - Round‑trip сохранение/загрузка: восстановление весов, оптимизатора, шагов/эпох.
  - Мини‑resume parity: после «corrupt → load» параметры идентичны сохранённым.

- `tests/test_perf_smoke.py`
  - CUDA‑только (skip на CPU):
    - Отсутствие VRAM‑утечек между итерациями (толеранс 5 МБ).
    - Слабый SLA по времени на небольшом инпуте.

- `tests/test_ddp_windows.py`
  - CPU‑DDP spawn (backend `gloo`) с file init‑методом:
    - Паритет градиентов между ранками до шага оптимизатора.
    - Паритет состояния оптимизатора (momentum buffer) после шага.
    - Паритет параметров модели между ранками.
  - Windows‑пути: сохранение/загрузка чекпоинта по смешанным разделителям `runs/exp1/checkpoints\latest.pth`.

## Примеры сценариев
- Запустить только CRB внутри модульных тестов:
  ```bash
  python test_project.py run modules::test_crb
  ```
- Прогнать только инференс‑тесты:
  ```bash
  python test_project.py run inference
  ```
- Полный прогон перед коммитом:
  ```bash
  python test_project.py run all
  ```
- Перед предпросмотром убедитесь, что в `configs/default.yaml` корректно выставлены `training.image_size`, `training.geometry.*` и `training.resize.filter`.

### Примечания Windows / DDP
- CPU‑DDP тест (`tests/test_ddp_windows.py::test_ddp_cpu_spawn_state_sync`) запускается автоматически в общем наборе и не требует GPU.
- На Windows используется старт‑метод `spawn`; тест сам включает его при необходимости.
- Инициализация через `file://` избавляет от конфликтов портов в локальной среде/CI.

## Подсказки и устранение неполадок
- **Кодировка в Windows:** если русские буквы отображаются некорректно — это не влияет на выполнение тестов. Можно сменить шрифт/кодировку консоли или использовать терминал, поддерживающий UTF‑8.
- **Долгий прогон:** используйте точечные тесты (`::<имя>`) для быстрой изоляции проблем.
- **Подробный лог PyTest:** замените `-q` в `test_project.py` на более подробный режим или добавьте флаги PyTest вручную после цели (см. ниже). 
- **Передача собственных путей:** лаунчер позволяет указывать пути напрямую вместо алиаса, например: `python test_project.py run tests/test_modules.py::test_crb`.

## CI и локальные проверки
- Рекомендуется перед пушем выполнять: `python test_project.py run all`.
- В репозитории может быть настроен GitHub Actions для автоматического прогона. Локальный прогон помогает быстрее получить обратную связь.

---

Статус набора тестов: все актуальные тесты должны проходить. Смотрите локальный прогон через `python test_project.py run all` и статусы CI.
