---

<h2 align="center">🛡 Защитите своё Android-устройство</h2>

<p align="center">
  <b>Продвинутый антивирус с AI для Android</b><br>
  Защита в реальном времени • Умное ML-сканирование • Чистый и быстрый интерфейс
</p>

<p align="center">
  <a href="https://play.google.com/store/apps/details?id=com.aiiql.armaga" target="_blank">
    <img src="https://img.shields.io/badge/Скачать_в-Google_Play-34A853?style=for-the-badge&logo=google-play&logoColor=white" />
  </a>
</p>

---
# TintoraAI: Продвинутая система колоризации изображений

<div align="center">
  <img src="assets/tintorai_logo.png" alt="TintoraAI Logo" width="512"/>
  <h3>Превращайте черно-белые изображения в яркие цветные с помощью ИИ</h3>
  <p>Умная колоризация • Гибкая настройка</p>
</div>

<div align="center">

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.9.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Version](https://img.shields.io/badge/Версия-2.0.3-4CAF50?style=for-the-badge)](https://github.com/T1desLuck/TintoraAI.git)
[![License](https://img.shields.io/badge/Лицензия-MIT-007ACC?style=for-the-badge)](https://github.com/T1desLuck/TintoraAI/blob/a588a27c8a52600bb0cfbf9eff56a0291502f78e/LICENSE)
[![Open In Colab](https://img.shields.io/badge/Open%20in-Colab-orange?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/gist/T1desLuck/2c599c68aa99e88308a2d5b0f79af634/tintoraai.ipynb)
[![Repo Size](https://img.shields.io/github/repo-size/T1desLuck/TintoraAI?style=for-the-badge)](https://github.com/T1desLuck/TintoraAI)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-downloads)
[![OS](https://img.shields.io/badge/OS-Linux-0078D6?style=for-the-badge&logo=linux&logoColor=white)](https://www.kernel.org/)
[![GitHub Forks](https://img.shields.io/github/forks/T1desLuck/TintoraAI?style=for-the-badge&logo=github&logoColor=white)](https://github.com/T1desLuck/TintoraAI/network/members)
![Made in Kazakhstan](https://img.shields.io/badge/MADE%20IN-KAZAKHSTAN-00BFFF?style=for-the-badge&labelColor=FFD700)
[![CI Status](https://img.shields.io/github/actions/workflow/status/T1desLuck/TintoraAI/python-app.yml?branch=main&style=for-the-badge)](https://github.com/T1desLuck/TintoraAI/actions/workflows/python-app.yml)

</div>

<p><small><strong>📚 Содержание:</strong> 
<a href="#-особенности">✨ Особенности</a> • 
<a href="#-быстрый-старт">🚀 Быстрый старт</a> • 
<a href="#-документация">📖 Документация</a> • 
<a href="#-архитектура-системы">🧠 Архитектура системы</a> • 
<a href="#поток-весов-при-инференсе-baseadapterlora">Поток весов при инференсе</a> • 
<a href="#️-конфигурация">🛠️ Конфигурация</a> • 
<a href="#adapter--lora-новые-возможности">Adapter / LoRA</a> • 
<a href="#️-примеры-результатов">🖼️ Примеры результатов</a> • 
<a href="#-требования">🔧 Требования</a> • 
<a href="#️-как-внести-вклад">⚙️ Как внести вклад</a> • 
<a href="#-тесты">🧪 Тесты</a> • 
<a href="#-лицензия">📄 Лицензия</a> • 
<a href="#-цитирование">📜 Цитирование</a> • 
<a href="#-контакты">📞 Контакты</a>
</small></p>

<p><small><strong>🔗 Быстрые ссылки:</strong> 
<a href="INSTALL.md">INSTALL.md</a> • 
<a href="TRAINING.md">TRAINING.md</a> • 
<a href="CONFIGURATION.md">CONFIGURATION.md</a> • 
<a href="TEST.md">TEST.md</a>
</small></p>

## ✨ Особенности

> [!IMPORTANT]
> **Гибридный энкодер** — ConvNeXt‑Tiny (локальные детали) → CoAtNet‑light (Conv+Attention) → Geometry‑Aware Transformer (глобальный контекст)
>
> **Семантическая согласованность** — OMM (банк прототипов) + CRB формируют глобальные цветовые условия для FiLM в декодере
>
> 🧩 **VGG только для обучения** — используетcя исключительно для perceptual loss; на инференсе не нужен и в модель не входит. Других внешних весов не используется.
>
> ⚡ **AMP + EMA по умолчанию** — AMP ускоряет и экономит память; EMA стабилизирует качество и применяется на инференсе.

> [!NOTE]
> 🎯 **Curriculum‑обучение** — поэтапная активация компонентов: SSL предобучение → базовое L1 → геометрия → перцептуалка → OMM чтение/ColorConsistency → GAN
>
> 👁️ **PatchGAN (опц.)** — подключается на финальной фазе для повышения реалистичности
>
> 🧠 **OMM** — объектная память с EMA‑обновлением прототипов и статистикой цветов
>
> 🧵 **Распределённое обучение (DDP)** — поддержка torchrun; синхронизация OMM выполняется на каждом шаге обучения.
>
> 🧩 **Грид‑пулинг OMM** — адаптивная сетка регионов (по умолчанию 7×7). Меняется через `omm.extra_params.grid` в `configs/default.yaml`.

> [!TIP]
> 💻 Командная строка с прогресс-индикаторами
>
> ⚙️ Настройка через YAML-конфигурации без изменения кода
>
> **Контроль качества** — поддержка расчёта метрик (SSIM, опционально DISTS и CIEDE2000) в процессе валидации
>
> 🖼 **Произвольные размеры на инференсе** — автоматический padding до кратности 32 (по умолчанию) и обратный unpad. Кратность настраивается: `configs/default.yaml -> inference.pad_divisor` или флагом CLI `--pad-div`.
>
> 🗺 **Тайловый инференс** — для очень больших изображений с overlap и feathering краёв.
>
> 🧪 **Тестирование** — см. `TEST.md` для описания тестов и сценариев запуска `test_project.py`.
>
> 🔁 **Воспроизводимость** — ключи `project_name` и `seed` в `configs/default.yaml` для фиксирования экспериментов.

## 🚀 Быстрый старт

### Установка

```bash
# Клонирование репозитория
git clone https://github.com/T1desLuck/TintoraAI.git
cd TintoraAI

# Установка зависимостей
pip install -r requirements.txt

# (опционально) Создайте виртуальное окружение и активируйте его
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
```

### Колоризация одного изображения
```bash
python -m src.inference --input path/to/image_or_dir --checkpoint checkpoints/latest.pth --config configs/default.yaml --tile 0
```

> Примечание: флаг `--checkpoint` опционален — при отсутствии явного пути инференс использует `paths.checkpoints` и имя из `checkpointing.latest_names.model` в `configs/default.yaml`.

### Пакетная обработка каталога
```bash
python -m src.inference --input data/test --checkpoint checkpoints/latest.pth --config configs/default.yaml --tile 512 --overlap 32
```

#### TTA при инференсе (включается в YAML)
Включите в `configs/default.yaml`:

```yaml
inference:
  pad_divisor: 32
  tta:
    enabled: true
    flip: true
    scales: [1.0, 0.75, 1.25]
```

После этого запустите обычную команду инференса (флаги `--tile/--overlap` совместимы с TTA и будут учтены):

```bash
python -m src.inference --input data/test --config configs/default.yaml --tile 512 --overlap 32
```

### Запуск обучения (см. подробности в TRAINING.md)
```bash
python -m src.train --config configs/default.yaml
```

#### Распределённый запуск (много‑GPU)
```bash
torchrun --standalone --nproc_per_node=NUM_GPUS -m src.train --config configs/default.yaml
```

## ✅ Поведение и договорённости (аудит)

- **Нормализация Lab**: датасеты отдают `L∈[-1,1]` (формула `L/50−1`), каналы `a,b` — в единицах Lab (≈ `[-128,127]`). `src/utils/lab_color.py` ожидает ровно такие диапазоны и конвертирует в RGB `[0,1]`.
- **OMM при инференсе**: чтение из памяти включено по умолчанию (`omm_read_only=False`) в `src/inference.py`.
- **OMM при валидации**: теперь следует учебному расписанию — в `src/train.py` валидация передаёт `omm_read_only=omm_epoch_read_only` (фаза‑зависимо).
- **TensorBoard скаляры**: добавлены `train/phase` и `train/omm_read_only` (0/1) в конце каждой эпохи обучения.
- **Padding на инференсе**: по умолчанию кратность `pad_divisor=32`, настраивается в `configs/default.yaml -> inference.pad_divisor` или через CLI `--pad-div`.

## 📖 Документация
Подробное описание всех функций, настроек и примеры использования доступны в следующих документах:

- [Инструкция по установке](INSTALL.md) — подробное руководство по установке на различных платформах
- [Руководство по обучению](TRAINING.md) — подготовка данных, конфигурация и запуск обучения
- [Руководство по конфигурации](CONFIGURATION.md) — важные настройки (DLB, GAN, TTA, CIEDE2000) и примеры YAML
- [Руководство по тестированию](TEST.md) — как запускать тесты, какие файлы что проверяют, примеры сценариев

## 🖼️ Примеры результатов
<div align="center">
  <img src="assets/bw.jpg" alt="Example 1" width="65%"/>
  <p><i>Сравнение исходного черно-белого изображения и результата колоризации</i></p>
  <img src="assets/color.jpg" alt="Example 2" width="65%"/>
  <p><i>Применение разных стилей к одному изображению</i></p>
</div>

## 🧠 Архитектура системы
TintoraAI — гибридный Conv+Attention автокодер с модулем памяти (OMM), геометрическими головами и условным декодированием через FiLM.

<div align="center">

🟠 Вход L → 🟦 ConvNeXt‑Tiny (F1) → 🟩 CoAtNet‑light (F2) → 🟪 GAT (F3) → 🧠 OMM + 🧭 Depth/Illum + ⛰️ Normals → 🎛️ CRB (FiLM) → 🧵 U‑Net++ Decoder → 🎨 a/b (+sat) → 🌈 Lab→RGB

</div>

<details>
<summary><b>Поток данных (схема)</b></summary>

```
Вход L (1×H×W)
      │
      ▼
ConvNeXt‑Tiny  →  CoAtNet‑light  →  Geometry‑Aware Transformer
   (F1 H/4)         (F2 ≈ H/8)            (F3 ≈ H/16)
      │                    │                     │
      └──────────────┬─────┴─────────────┬───────┘
                     ▼                   ▼
            Depth / Illum Heads      Region Pooling (grid 7×7)
                     │                   │
                     ▼                   ▼
                 Нормали               OMM (банк прототипов: N×D)
                     │                   │
                     └──────────┬────────┘
                                ▼
                         CRB (c_color)
                                ▼
                U‑Net++ Decoder + FiLM + PixelShuffle
                                ▼
                      Предсказания a,b (+sat)
                                ▼
                          Lab→RGB (Выход)
```

</details>

### Размерности признаков

| Уровень | Пространство | Каналы |
|---------|--------------:|-------:|
| F1 (ConvNeXt‑Tiny) | H/4 × W/4  | 96  |
| F2 (CoAtNet‑light) | H/8 × W/8  | 192 |
| F3 (GAT)           | H/16 × W/16| 384 |

### Компоненты

- __Backbone__
  - `ConvNeXt‑Tiny` → `src/models/backbone/convnext_wrapper.py`
  - `CoAtNet‑light` → `src/models/backbone/coatnet_wrapper.py`
  - `GATLight` → `src/models/backbone/gat_light.py`

- __Головы геометрии__
  - `DepthHead`, `IlluminationHead` → `src/models/heads/heads.py`
  - Нормали из глубины (Собель) → `TintoraAI.compute_normals()`

- __OMM (Object Memory Module)__
  - Региональный пулинг (по умолчанию 7×7), top‑k cosine softmax, EMA‑обновление прототипов и цветовой статистики → `src/models/omm/object_memory.py`

- __CRB (Color Reasoning Block)__
  - Слияние F3 + mem_map + D/I/Normals, проекция в контекст и генерация FiLM → `src/models/crb/crb.py`

- __Декодер__
  - U‑Net++ + FiLM на каждой стадии + PixelShuffle → `src/models/decoder/decoder_unetpp.py`

> [!NOTE]
> В текущей реализации выход ConvNeXt‑Tiny (`F1`, масштаб H/4) подаётся в CoAtNet‑light, после чего берётся карта признаков стадии `out_indices=2` (см. `CoAtNetLight`). Далее `GATLight` даунсэмплирует ×2. Это даёт фактические масштабы признаков: `F2 ≈ H/8`, `F3 ≈ H/16`. Для совместимости выполняется приведение каналов `256→192` через `coatnet_channel_fix` (см. `src/models/tintoraai.py`). Переключение на иные стадии CoAtNet упоминается как потенциальная опция, но в текущей версии не параметризовано и потребует изменения каналов и переобучения.
>
> Дополнение: явного узла FPN нет; уровни `F1/F2/F3` образуют пирамиду признаков и используются напрямую в `CRB` и декодере.

### Поток весов при инференсе (base/adapter/LoRA)
- Базовые веса (base) загружаются первыми и определяют поведение модели.
- Если присутствует `checkpoints/adapters/adapter.pth`, применяется аддитивная поправка (Adapter) к целевым слоям (decoder/CRB) с весом `merging.weights.adapter`.
- Затем последовательно применяются LoRA‑поправки из `checkpoints/lora/*.pth` (несколько файлов допускаются) с весами из `merging.weights.lora.{default|per‑name}`.
- При отсутствии Adapter/LoRA модель работает как прежде, без каких‑либо изменений.

## 🛠️ Конфигурация
Все настройки задаются в `configs/default.yaml`.

- Краткий обзор параметров и новых опций см. в `CONFIGURATION.md`.
- Подробные рекомендации по обучению и препроцессингу см. в `TRAINING.md` (раздел «Настройка конфигурации»).

### Adapter / LoRA (новые возможности)

Назначение весов:
- **Базовые веса (base)** — главный «стержень» качества. Это стабильная, полноценно обученная модель (`checkpoints/latest.pth`). Они доминируют при слиянии и отвечают за общий реализм, цветовую согласованность и универсальность.
- **Adapter** — мягкая корректировка сложных сцен и глобальных решений цвета. Реализован как аддитивные дельты к выбранным слоям (decoder/CRB). Обучается отдельно на ваших данных/подвыборках, но не меняет base. В слиянии добавляется «сверху» на base с весом `merging.weights.adapter`.
- **LoRA** — тематические «микро‑настройки» (лица, животные, транспорт и т. п.) низкого ранга для конкретных признаков. Можно иметь несколько LoRA одновременно. На инференсе каждое LoRA добавляется поверх (после Adapter) с весами `merging.weights.lora.{default|per‑name}`. Это повышает детализацию нужной тематики, не «перекрашивая» всё изображение.

Как это работает на инференсе:
1. Загружаются базовые веса модели (base).
2. Если найден `checkpoints/adapters/adapter.pth`, применяется взвешенная аддитивная дельта (Adapter).
3. Если найдены `checkpoints/lora/*.pth`, последовательно добавляются их низкоранговые поправки (LoRA), каждая со своим весом.
4. При отсутствии файлов Adapter/LoRA никакого слияния не происходит — модель работает как прежде.

- Adapter и LoRA — модульные дополнения без изменений базовой модели. Тренируются отдельно, но с использованием той же логики/конфига.
- Инференс автоматически подхватывает `checkpoints/adapters/*.pth` и `checkpoints/lora/*.pth` и выполняет взвешенное слияние: base → adapter → loRA. Если файлов нет — поведение 1‑в‑1 как раньше.
- Весовые коэффициенты слияния настраиваются в `merging.weights` (см. `CONFIGURATION.md`).

Быстрый старт:
```bash
# Обучение Adapter
python -m src.train_adapter --config configs/default.yaml

# Обучение LoRA (имя берётся из adapters.lora_name)
python -m src.train_lora --config configs/default.yaml

# Инференс: автоматически подмержит adapter/LoRA, если они есть
python -m src.inference --input data/test --config configs/default.yaml --output outputs
```
Подробнее: разделы «Adapter/LoRA» в `TRAINING.md` и «Новые опции» в `CONFIGURATION.md`.

### FAQ: Adapter/LoRA на инференсе — кратко

- **Где хранить веса?**
  - Adapter: `checkpoints/adapters/adapter.pth`
  - LoRA: `checkpoints/lora/*.pth` (например, `lora_face_2025-09-16.pth`)

- **Подхватываются автоматически?** Да. При загрузке базовых весов (`load_state_dict`) срабатывает хук из `sitecustomize.py`, который вызывает `src/models/merge_all.scan_and_merge()`.

- **Если файлов нет?** Ничего не меняется: работает чистая base‑модель.

- **Порядок слияния и вклад:** `base → adapter → lora`.
  - Вес Adapter: `merging.weights.adapter` (по умолчанию `0.8`).
  - Вес(а) LoRA: `merging.weights.lora.default` или по именам `merging.weights.lora.<name>`.
  - Имя `<name>` определяется из файла: `lora_<name>_*.pth` → `<name>`, иначе берётся целиком `stem` файла.

- **Как временно отключить?**
  - Переместить/переименовать файлы из соответствующих папок, либо
  - Поставить вес в конфиге `0.0` (например, `merging.weights.adapter: 0.0` или `merging.weights.lora.face: 0.0`).

- **Что если нет base‑чекпоинта?** Если `--checkpoint` и `paths.checkpoints/latest.pth` отсутствуют, `inference.py` работает со случайными весами и пропускает `load_state_dict`, поэтому автослияние не сработает. Рекомендуется всегда иметь базовый чекпоинт.

- **Что даёт Adapter/LoRA?**
  - Adapter — "глобальная коррекция" сложных сцен и цветовых решений без изменения base.
  - LoRA — тематические точечные нюансы (например, лица): усиливает детальность нужного домена, можно подключать несколько.

## 🔧 Требования
- Python 3.9+
- PyTorch 2.3.1 (см. `requirements.txt`), колёса с CUDA 12.1 доступны на Windows/Linux
- 8GB+ RAM (рекомендовано 16GB+)
- GPU 6GB+ VRAM для обучения на 256×256; для инференса достаточно 2–4GB

## ⚙️ Как внести вклад
1. Форкните репозиторий
2. Создайте ветку для вашей функциональности
3. Внесите изменения и протестируйте их
4. Отправьте Pull Request с подробным описанием изменений

## 📊 Целевые показатели (по ТЗ)
| Метрика       | Цель          | Примечание                       |
|---------------|---------------|----------------------------------|
| SSIM          | ≥ 0.82        | Структурное сходство            |
| DISTS         | ≤ 0.20        | Перцептуально‑структурное сходство |
| FPS (CPU)     | ~1–2          | Зависит от размера изображения  |
| FPS (GPU)     | ~20–30        | На GPU уровня RTX 2080          |

## 📜 Цитирование
Если вы используете TintoraAI в своих исследованиях, пожалуйста, процитируйте наш проект:

```bibtex
@software{TintorAI,
  author = {T1desLuck},
  title = {TintoraAI: Продвинутая система колоризации изображений},
  year = {2025},
  url = {https://github.com/T1desLuck/TintoraAI}
}
```

## 🙏 Благодарности
- PyTorch — за фреймворк глубокого обучения
- ConvNeXt — за идеи архитектуры бэкбона ранних этапов энкодера
- CoAtNet — за гибрид Conv+Attention подход для среднего уровня признаков
- ColorFormer (ECCV'22) — за концепцию memory decoder/прототипов для цветовой согласованности
- PatchGAN — за идею локального дискриминатора для адверсариального обучения (фаза 4)
- VGG — за перцептуальные признаки для perceptual loss
- SimCLR — за идеи self-supervised pre-training (contrastive) для разогрева энкодера

## 📞 Контакты
- GitHub Issues: https://github.com/T1desLuck/TintoraAI/issues
- [![Email](https://img.shields.io/badge/Email-tidesluck%40icloud.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:tidesluck@icloud.com)

## 📁 Дерево проекта (с комментариями)
Ниже приведена структура репозитория с краткими пояснениями по ключевым файлам и папкам. Папки для данных и результатов могут отсутствовать при клонировании (GitHub не хранит пустые директории) — создайте их вручную, как показано ниже.

```text
TintoraAI/
├─ README.md                      # Общее описание проекта, ссылки и быстрый старт
├─ INSTALL.md                     # Подробная установка
├─ TRAINING.md                    # Руководство по обучению модели
├─ TEST.md                        # Руководство по тестированию и сценарии запуска
├─ CONFIGURATION.md               # Справочник по настройкам и новым опциям
├─ requirements.txt               # Зависимости проекта
├─ main.py                        # Точка входа (при необходимости)
├─ test_project.py                # Русскоязычный лаунчер тестов
├─ sitecustomize.py               # Авто‑хук Python: автослияние Adapter/LoRA при инференсе (без правок core)
├─ configs/                       # Конфигурации (YAML)
│  └─ default.yaml                # Базовая конфигурация (пути, обучение, лоссы, модель)
├─ src/                           # Исходный код библиотеки TintoraAI
│  ├─ __init__.py                 # Инициализация пакета src
│  ├─ train.py                    # Скрипт обучения: python -m src.train --config ...
│  ├─ inference.py                # Скрипт инференса: python -m src.inference --input ...
│  ├─ train_adapter.py            # Обучение Adapter (дельты поверх decoder/CRB), сохраняет checkpoints/adapters/adapter.pth
│  ├─ train_lora.py               # Обучение LoRA (low‑rank A,B) поверх decoder/CRB, сохраняет checkpoints/lora/*.pth
│  ├─ datasets/                   # Датасеты и аугментации
│  │  ├─ __init__.py
│  │  ├─ simple_dataset.py        # Простой датасет (пример)
│  │  ├─ advanced_dataset.py      # Продвинутый датасет (используется по умолчанию)
│  │  └─ augmentations.py         # Базовые аугментации
│  ├─ losses/                     # Функции потерь
│  │  ├─ __init__.py
│  │  ├─ advanced.py              # PhotometricSmoothnessLoss и др.
│  │  ├─ basic.py                 # Базовые потери (L1 и т.п.)
│  │  ├─ perceptual.py            # Перцептуальные потери (VGG)
│  │  ├─ gan.py                   # Потери для GAN
│  │  └─ patchnce.py              # Настройки/заготовка для SSL PatchNCE
│  ├─ models/                     # Архитектура модели
│  │  ├─ __init__.py              # Инициализация пакета моделей
│  │  ├─ adapter.py               # Container/merge утилиты для Adapter (аддитивные дельты)
│  │  ├─ discriminator.py         # PatchGAN дискриминатор (фаза 4)
│  │  ├─ guidenet.py              # Доп. сеть-подсказчик (опционально)
│  │  ├─ lora.py                  # Container/merge утилиты для LoRA (низкоранговые факторы)
│  │  ├─ tintoraai.py             # Основной класс модели TintoraAI
│  │  ├─ merge_all.py             # Автоскан checkpoints и взвешенное слияние base→adapter→lora
│  │  ├─ backbone/                # Бэкбоны (ConvNeXt/CoAtNet/GAT)
│  │  │  ├─ __init__.py
│  │  │  ├─ convnext_tiny.py      # Реализация ConvNeXt‑Tiny (ранний этап энкодера)
│  │  │  ├─ convnext_wrapper.py   # Обёртка/унификация интерфейса ConvNeXt
│  │  │  ├─ coatnet_light.py      # Лёгкий CoAtNet (Conv+Attention) для среднего уровня
│  │  │  ├─ coatnet_wrapper.py    # Обёртка/конфигурация CoAtNet
│  │  │  └─ gat_light.py          # Geometry‑Aware Transformer (облегчённый, глобальный контекст)
│  │  ├─ heads/                   # Головы глубины/освещённости/и т.п.
│  │  │  ├─ __init__.py           # Инициализация пакета голов
│  │  │  └─ heads.py
│  │  ├─ crb/                     # Color Reasoning Block (CRB)
│  │  │  ├─ __init__.py           # Инициализация пакета CRB
│  │  │  └─ crb.py
│  │  ├─ omm/                     # Object Memory Module (OMM)
│  │  │  ├─ __init__.py           # Инициализация модуля памяти
│  │  │  └─ object_memory.py      # Банк прототипов: assign (cosine/top‑k), EMA‑обновления, min_support, статистика цветов (μ/σ), чтение/запись
│  │  └─ decoder/                 # Декодер (U‑Net++)
│  │     ├─ __init__.py           # Инициализация пакета декодера
│  │     └─ decoder_unetpp.py
│  └─ utils/                      # Утилиты и вспомогательные модули
│     ├─ __init__.py
│     ├─ balancer.py              # Альтернативный балансировщик потерь
│     ├─ config.py                # Утилиты для работы с конфигами
│     ├─ dist.py                  # Вспомогательные функции для DDP/распределёнки
│     ├─ dlb.py                   # DynamicLossBalancer
│     ├─ lab_color.py             # Преобразования Lab/RGB
│     ├─ metrics.py               # Метрики (SSIM/DISTS)
│     └─ seed.py                  # Фиксация сидов/детерминизм
├─ tests/                         # Набор автотестов (pytest)
│  ├─ test_dlb.py                 # Тесты балансировщика потерь
│  ├─ test_dlb_entropy.py         # Энтропийно‑осведомлённая стратегия DLB
│  ├─ test_forward.py             # Быстрый прогон forward модели
│  ├─ test_inference.py           # Тесты инференса (single/tiled)
│  ├─ test_losses.py              # Тесты лоссов (advanced и др.)
│  ├─ test_gan_r1.py              # GAN + R1 sanity
│  ├─ test_modules.py             # Модули: backbone/heads/CRB/decoder
│  ├─ test_model_api.py           # Проверка API модели: формы, устройство/тип, AMP/FP32, TTA/tiling
│  ├─ test_metrics.py             # SSIM/DISTS/CIEDE2000 sanity и инварианты
│  ├─ test_config_validation.py   # Валидация конфигов: схема, типы, неизвестные ключи
│  ├─ test_cli_precedence.py      # Приоритет CLI‑флагов над YAML
│  ├─ test_dataset_pipeline.py    # Загрузка/геометрия/resize/детерминизм
│  ├─ test_tiling_equivalence.py  # Эквивалентность тайлинга/цельного инференса
│  ├─ test_train_micro.py         # Микро‑интеграция тренинга (неск. шагов)
│  ├─ test_checkpoint_io.py       # Checkpoint I/O: latest/best имена, roundtrip, resume parity
│  ├─ test_perf_smoke.py          # Лёгкий перф/VRAM smoke‑тест (CUDA, skip на CPU)
│  ├─ test_ddp_windows.py         # DDP (CPU) spawn + Windows path separators
│  ├─ test_train_adapter_fast.py  # Быстрый интеграционный тест Adapter (микро‑обучение/экспорт)
│  └─ test_train_lora_fast.py     # Быстрый интеграционный тест LoRA (микро‑обучение/экспорт)
├─ scripts/                       # Утилиты/скрипты
│  ├─ preview_preprocessing.py    # Предпросмотр подготовки изображения (Lab, L‑канал, дефекты)
│  ├─ smoke_all.py                # Быстрый sequential‑smoke набора тестов/примеров
│  └─ smoke_inference.py          # Smoke‑инференс на паре изображений
├─ assets/                        # Статические ресурсы (логотип, образцы)
│  ├─ tintorai_logo.png
│  ├─ bw.jpg
│  └─ color.jpg
├─ out_main_infer_ok.png/         # Примеры/артефакты локального прогона (может отсутствовать)
├─ out_main_infer.png/            # Примеры/артефакты локального прогона (может отсутствовать)
├─ data/                          # [Папка-плейсхолдер] Данные (создайте вручную)
│  ├─ train/                      # Обучающие изображения (jpg/png) без подпапок
│  ├─ val/                        # Валидационные изображения
│  └─ test/                       # Тестовые изображения (опционально)
├─ checkpoints/                   # [Папка-плейсхолдер] Чекпоинты моделей (*.pth)
│  ├─ adapters/                   # Веса Adapter (adapter.pth)
│  └─ lora/                       # Веса LoRA (lora_<name>_<YYYY-MM-DD>.pth)
├─ logs/                          # [Папка-плейсхолдер] Логи (TensorBoard, тексты)
└─ experiments/                   # [Папка-плейсхолдер] Эксперименты/визуализации
   └─ exp_default/                # Каталог эксперимента по умолчанию
```

Примечания:
- Плейсхолдер‑папки (`data/`, `checkpoints/`, `logs/`, `experiments/…`) создаются пользователем локально — они не версионируются, если пустые.
- Пути по умолчанию настраиваются в `configs/default.yaml` секция `paths.*`.
- Для инференса минимум нужен вход (`--input`), конфиг и (обычно) чекпоинт из `checkpoints/`.

Дополнительно — предпросмотр подготовки изображений:
- Скрипт `scripts/preview_preprocessing.py` позволяет визуально проверить пайплайн подготовки (resize/crop/flip, перевод в Lab, сохранение L‑канала) и опциональные L‑дефекты.
- Удобнее запускать через лаунчер: `python test_project.py preview --input assets/color.jpg --save_L --enable_defects`.
- Выходные файлы сохраняются в `experiments/exp_default/preview_preprocess` (или папку, указанную флагом `--output`).
 - Конфиг дефектов: блок `training.aug.defects` в `configs/default.yaml`. По умолчанию выключено (`enabled: false`); активация через флаг `--enable_defects`.

## 🧪 Тесты

Коротко о запуске тестов. Полная документация перенесена в `TEST.md`.

- Запуск через лаунчер:
  ```powershell
  python test_project.py help
  python test_project.py list
  python test_project.py run all
  ```

- Точечные примеры:
  ```powershell
  python test_project.py run modules
  python test_project.py run modules::test_crb
  python test_project.py run inference
  ```

- Альтернатива (pytest напрямую):
  ```powershell
  python -m pip install pytest
  python -m pytest -q
  ```

Смотрите подробные цели запуска, список файлов и подсказки в `TEST.md`.

## 📄 Лицензия
Этот проект распространяется под лицензией MIT. См. файл LICENSE для более подробной информации.
