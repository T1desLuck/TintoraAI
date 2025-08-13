# TintoraAI: Продвинутая система колоризации изображений

<div align="center">
  <img src="assets/tintorai_logo3.png" alt="TintoraAI Logo" width="1024"/>
  <h3>Превращайте черно-белые изображения в яркие цветные с помощью ИИ</h3>
  <p>Умная колоризация • Гибкая настройка</p>
</div>

<div align="center">

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.9.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Version](https://img.shields.io/badge/Версия-1.1.0-4CAF50?style=for-the-badge)](https://github.com/T1desLuck/TintoraAI.git)
[![License](https://img.shields.io/badge/Лицензия-MIT-007ACC?style=for-the-badge)](https://github.com/T1desLuck/TintoraAI/blob/a588a27c8a52600bb0cfbf9eff56a0291502f78e/LICENSE)
[![Open In Colab](https://img.shields.io/badge/Open%20in-Colab-orange?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/T1desLuck/TintoraAI/blob/main/your_notebook.ipynb)
[![Repo Size](https://img.shields.io/github/repo-size/T1desLuck/TintoraAI?style=for-the-badge)](https://github.com/T1desLuck/TintoraAI)
[![CI Status](https://img.shields.io/github/actions/workflow/status/T1desLuck/TintoraAI/python-app.yml?branch=main&style=for-the-badge)](https://github.com/T1desLuck/TintoraAI/actions/workflows/python-app.yml)

</div>

 TintoraAI — это современная система для интеллектуальной колоризации черно-белых изображений на PyTorch. Текущая архитектура объединяет многоэтапный энкодер (ConvNeXt‑Tiny → CoAtNet‑light → Geometry‑Aware Transformer), вспомогательные головы глубины/освещённости, модуль объектной памяти (OMM), блок цветового рассуждения (CRB) и декодер U‑Net++ с FiLM и PixelShuffle. Поддерживается поэтапное обучение (curriculum), перцептуальная потеря (VGG) для обучения и опциональный PatchGAN на поздней фазе.

## ✨ Особенности

> [!IMPORTANT]
> **Гибридный энкодер** — ConvNeXt‑Tiny (локальные детали) → CoAtNet‑light (Conv+Attention) → Geometry‑Aware Transformer (глобальный контекст)
>
> **Семантическая согласованность** — OMM (банк прототипов) + CRB формируют глобальные цветовые условия для FiLM в декодере

> [!NOTE]
> 🎯 **Curriculum‑обучение** — поэтапная активация компонентов: SSL предобучение → базовое L1 → геометрия → перцептуалка → OMM чтение/ColorConsistency → GAN
>
> 👁️ **PatchGAN (опц.)** — подключается на финальной фазе для повышения реалистичности
>
> 🧠 **OMM** — объектная память с EMA‑обновлением прототипов и статистикой цветов

> [!TIP]
> 💻 Командная строка с прогресс-индикаторами
>
> ⚙️ Настройка через YAML-конфигурации без изменения кода
>
> **Контроль качества** — поддержка расчёта метрик (SSIM, PSNR, LPIPS) в процессе валидации

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

### Запуск обучения (см. подробности в TRAINING.md)
```bash
python -m src.train --config configs/default.yaml
```

#### Распределённый запуск (много‑GPU)
```bash
torchrun --standalone --nproc_per_node=NUM_GPUS -m src.train --config configs/default.yaml
```

## 📖 Документация
Подробное описание всех функций, настроек и примеры использования доступны в следующих документах:

- [Инструкция по установке](INSTALL.md) — подробное руководство по установке на различных платформах
- [Руководство по обучению](TRAINING.md) — подготовка данных, конфигурация и запуск обучения

## 🖼️ Примеры результатов
<div align="center">
  <img src="assets/examples/example1.jpg" alt="Example 1" width="80%"/>
  <p><i>Сравнение исходного черно-белого изображения и результата колоризации</i></p>
  <img src="assets/examples/example2.jpg" alt="Example 2" width="80%"/>
  <p><i>Применение разных стилей к одному изображению</i></p>
</div>

## 🧠 Архитектура системы
TintoraAI построен как гибрид Conv+Attention автокодер с модулем памяти:

```
Вход L (1×H×W)
      │
      ▼
ConvNeXt‑Tiny  →  CoAtNet‑light  →  Geometry‑Aware Transformer
   (F1 H/4)         (F2 H/8)              (F3 H/16)
      │                    │                     │
      └──────────────┬─────┴─────────────┬───────┘
                     ▼                   ▼
            Depth / Illum Heads      Region Pooling (8×8)
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
                          Lab→RGB (вывод)
```

## 🛠️ Конфигурация
Проект настраивается через один файл `configs/default.yaml`:

- `paths`: директории данных, логов, чекпоинтов
- `runtime`: AMP, DDP, num_workers, устройство
- `checkpointing`: имена и политика сохранения `latest`/`best`
- `optimizer`: тип, разные LR для backbone/decoder, weight decay
- `scheduler`: схема (например, cosine), warmup
- `training`: батч, эпохи, image_size, EMA, curriculum (фазы −1…4)
- `loss`: веса L1/Perceptual/Photometric/CC/Entropy/Cluster/Adv
- `omm`: N, D, top_k, tau, alpha, min_support
- `model`: каналы c1/c2/c3, film_dim, use_saturation_head, GuideNet (опц.)
- `gan`: параметры PatchGAN (если включён)
- `ssl`: настройки PatchNCE для предобучения (Phase −1)
- `validation`: батч, метрики (SSIM/LPIPS), параметры окон
- `logging`: TensorBoard/W&B и частота логирования

См. пример значений в `configs/default.yaml`.

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
| LPIPS         | ≤ 0.20        | Перцептуальное сходство         |
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
- Swin Transformer - за архитектуру трансформеров
- CycleGAN и pix2pix - за вдохновение в области генеративных моделей
- PyTorch - за фреймворк глубокого обучения

## 📞 Контакты
- GitHub Issues: https://github.com/T1desLuck/TintoraAI/issues
- [![Email](https://img.shields.io/badge/Email-tidesluck%40icloud.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:tidesluck@icloud.com)

## 📄 Лицензия
Этот проект распространяется под лицензией MIT. См. файл LICENSE для более подробной информации.
