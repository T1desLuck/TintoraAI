# 🎯 Руководство по обучению модели TintoraAI

Это руководство описывает полный процесс подготовки данных, настройки и обучения модели колоризации TintoraAI, включая продвинутые техники для достижения наилучших результатов.

> Навигация: [README.md](README.md) • [INSTALL.md](INSTALL.md) • [TEST.md](TEST.md)

## 📋 Содержание

- [Обзор процесса обучения](#обзор-процесса-обучения)
- [Подготовка данных](#подготовка-данных)
- [Настройка конфигурации](#настройка-конфигурации)
- [Запуск обучения](#запуск-обучения)
- [Мониторинг и визуализация](#мониторинг-и-визуализация)
- [Продвинутые техники обучения](#продвинутые-техники-обучения)
- [Оценка обученной модели](#оценка-обученной-модели)
- [Советы по оптимизации](#советы-по-оптимизации)
- [Обучение на специфических наборах данных](#обучение-на-специфических-наборах-данных)
- [Распределенное обучение](#распределенное-обучение)
- [Проблемы и решения](#проблемы-и-решения)

## 📊 Обзор процесса обучения

Обучение TintoraAI основано на следующем цикле:

1. **Подготовка данных** — Сбор RGB‑изображений в плоских каталогах `data/train` и `data/val` (L‑канал формируется на лету)
2. **Настройка гиперпараметров** — Определение архитектуры и параметров в `configs/default.yaml`
3. **Обучение модели** — Итеративное обучение с логированием и сохранением чекпоинтов
4. **Валидация** — Оценка на валидационном наборе в процессе обучения
5. **Тонкая настройка** — Корректировка параметров по результатам метрик и визуализаций
6. **Финальная оценка** — Проверка качества инференсом на отдельных данных

## 🖼️ Подготовка данных

### Структура данных

`AdvancedColorizationDataset` ожидает ПЛОСКИЕ каталоги изображений без парных подпапок. Цвет к серому преобразуется на лету.

```
data/
├── train/   # произвольные цветные изображения (jpg/png)
└── val/     # валидационные изображения
```

### Источники данных

Для обучения TintoraAI можно использовать различные наборы данных:

1. **Общедоступные наборы данных**:
   - [COCO](https://cocodataset.org/)
   - [ImageNet](https://www.image-net.org/)
   - [Places365](http://places2.csail.mit.edu/)

2. **Специализированные наборы данных**:
   - Исторические фотографии
   - Художественные произведения
   - Специфичные для предметной области изображения (медицинские, спутниковые и т.д.)

### Подготовка собственного набора данных

1. **Сбор изображений**:
```bash
# Создание директорий
mkdir -p data/raw data/train data/val
```

2. (Опционально) Быстрая заготовка train/val из "raw":
```python
# Пример скрипта для разбиения без преобразований
import os
from glob import glob
from sklearn.model_selection import train_test_split
from pathlib import Path
import shutil

raw = sorted(glob('data/raw/*.*'))
train_files, val_files = train_test_split(raw, test_size=0.1, random_state=42)
Path('data/train').mkdir(parents=True, exist_ok=True)
Path('data/val').mkdir(parents=True, exist_ok=True)
for p in train_files:
    shutil.copy(p, 'data/train')
for p in val_files:
    shutil.copy(p, 'data/val')
```

3. Аугментация данных: в `configs/default.yaml` секция `training.aug` управляет flip/crop/ab_jitter; аугментации применяются внутри датасета (advanced/simple) согласно конфигурации.

4. Проверка данных
Перед обучением убедитесь, что данные корректно подготовлены:
```bash
# Проверка количества изображений
echo "Тренировочных изображений: $(ls data/train | wc -l)"
echo "Валидационных изображений: $(ls data/val | wc -l)"
```

## ⚙️ Настройка конфигурации

Используется один файл `configs/default.yaml`.

Ключевые секции:
- `paths`: `data_root`, `train_dir`, `val_dir`, `logs`, `checkpoints`, `experiments`
- `runtime`: `device` (auto/cuda/cpu), `amp`, `num_workers`, `pin_memory`, `cudnn_benchmark`, `ddp.backend`
- `training`: `batch_size`, `epochs`, `image_size`, `dataset: advanced|simple`, `aug`, `ema`, `curriculum`
- `optimizer`: `lr_backbone`, `lr_decoder_heads`, `weight_decay_backbone`, `weight_decay_other`
- `scheduler`: `type` (cosine), `warmup_steps`
- `loss`: `lambda_l1`, `lambda_perc`, `lambda_photo`, `lambda_ds`, `lambda_cc`, `lambda_entropy`, `lambda_cluster`, `lambda_adv`
- `model`: `c1,c2,c3`, `film_dim`, `use_guidenet`, `use_saturation_head`, `omm`
- `gan`: `enabled`, `input_nc`, `ndf`, `n_layers`, `loss_type`
- `ssl`: `enabled`, `patchnce.*`
- `validation`: `enabled`, `batch_size`, `ssim_window`, `lpips`
- `logging`: TensorBoard/W&B
- `checkpointing`: политика и имена `latest`/`best`

## 🚀 Запуск обучения

### Базовое обучение
После подготовки данных и настройки конфигурации, запустите обучение:
```bash
# Активация виртуального окружения (если используется)
source venv/bin/activate  # Linux/macOS
# или
venv\Scripts\activate     # Windows

# Запуск обучения с базовыми параметрами
python -m src.train --config configs/default.yaml
```

### Дополнительные примеры
```bash
# Запуск на CPU: задайте устройство в конфиге
# способ 1: временно переопределите через отдельный yaml
# способ 2: отредактируйте `runtime.device: cpu` в configs/default.yaml
python -m src.train --config configs/default.yaml

# DDP (несколько GPU на одном узле)
torchrun --standalone --nproc_per_node=2 -m src.train --config configs/default.yaml

# Продолжение обучения
# (загрузка latest.pth из секции paths.checkpoints автоматически при наличии)
python -m src.train --config configs/default.yaml
```

### Запуск обучения на Google Colab
```python
# В Google Colab
!git clone https://github.com/T1desLuck/TintoraAI.git
%cd TintoraAI

# Установка зависимостей
!pip install -r requirements.txt

# Загрузка данных (пример с Google Drive)
from google.colab import drive
drive.mount('/content/drive')
!ln -s /content/drive/MyDrive/data data

# Запуск обучения
!python -m src.train --config configs/default.yaml
```

### Запуск обучения на Vast.ai
```bash
# На сервере Vast.ai
git clone https://github.com/T1desLuck/TintoraAI.git
cd TintoraAI
pip install -r requirements.txt

# Запуск обучения с несколькими GPU (DDP)
torchrun --standalone --nproc_per_node=2 -m src.train --config configs/default.yaml
```

## 📈 Мониторинг и визуализация

### TensorBoard
Во время обучения TintoraAI логирует метрики и визуализации в TensorBoard (если `logging.tensorboard: true`):
```bash
# Запуск TensorBoard (логдир из cfg.paths.logs, по умолчанию "logs")
tensorboard --logdir logs

# Затем откройте в браузере: http://localhost:6006
```

Что можно отслеживать:
- Метрики потерь: общая потеря и компоненты (L1, VGG, GAN, PatchNCE)
- Метрики качества: SSIM, LPIPS
- Визуализации: примеры колоризации и сравнения
- Гистограммы: распределение весов и градиентов
- Профили: использование памяти и время выполнения

### Журналы обучения
Логи обучения сохраняются в директорию, указанную в `paths.logs` (по умолчанию `logs/`):
```bash
# Просмотр последних логов обучения
Get-Content logs/training.log -Tail 50  # PowerShell
tail -50 logs/training.log              # bash
```

### Контрольные точки
Контрольные точки (чекпоинты) сохраняются в директорию `paths.checkpoints` (по умолчанию `checkpoints/`). Имена файлов берутся из `checkpointing.latest_names` и `checkpointing.best_names`.

По умолчанию (см. `configs/default.yaml`):
- latest: `latest.pth`, `latest_ema.pth`
- best (по метрике `checkpointing.best_metric`, по умолчанию `ssim`): `best_ssim.pth`, `best_ssim_ema.pth`

```bash
# Список доступных чекпоинтов
ls -lh checkpoints/
```

## 🧠 Продвинутые техники обучения

- **Curriculum‑обучение**: используйте `training.curriculum.enabled: true` и таблицу фаз `training.curriculum.phases` (см. пример в `configs/default.yaml`).
- **Перцептуальная/VGG‑потеря**: управляйте через `loss.lambda_perc` и связанные параметры модели/данных.
- **PatchGAN**: включите `gan.enabled: true` и настройте `gan.n_layers`, `gan.ndf`, `gan.loss_type`.
- **Динамическая балансировка потерь (DLB)**: включается настройками `training.dlb.*` (если используется).

### Использование OMM (Object Memory Module)
OMM хранит цветовые прототипы и используется как глобальный контекст:
```yaml
omm:
  enabled: true
  N: 2048
  D: 256
  top_k: 64
  tau: 0.07
  alpha: 0.995
  min_support: 15
  sync:
    enabled: true
    update_interval: 1
```

Примечания:
- Сетка регионального пуллинга задаётся в коде модели и по умолчанию равна 7×7. Прямого ключа `omm.grid` нет; при необходимости укажите `omm.extra_params.grid: [H, W]` в `configs/default.yaml`.
- Параметр `min_support` влияет на логику поддержания/реинициализации прототипов и статистик; он не маскирует чтение прототипов напрямую.
- `sync.update_interval` зарезервирован для будущего управления частотой синхронизации; в текущей реализации обновления памяти выполняются на каждом шаге с учётом настроек DDP.

## 🎯 Оценка обученной модели
Валидация выполняется во время обучения (если `validation.enabled: true`). Для ручной проверки визуальных результатов используйте инференс:
```bash
python -m src.inference --input data/val --config configs/default.yaml --checkpoint checkpoints/latest.pth --output outputs/val_pred
```
Если флаг `--output` не указан, путь вывода определяется из YAML: берётся `paths.experiments` (по умолчанию `experiments/exp_default`) или, если он отсутствует, `paths.logs`. В обоих случаях результаты сохраняются в подпапку `val_pred` (см. `src/inference.py`).

### Метрики качества
TintoraAI использует следующие метрики (встроенная поддержка при `validation.enabled: true`):
- SSIM — целевой ориентир ≥ 0.82
- LPIPS — целевой ориентир ≤ 0.20 (требуется пакет `lpips` и `validation.lpips: true`)


### Анализ по категориям
Подготовьте собственные списки файлов/подкаталоги и запускайте инференс по группам для сравнения метрик SSIM/LPIPS.

## 💡 Советы по оптимизации

### Оптимизация скорости обучения
- Смешанная точность (AMP): `runtime.amp: true`
- Загрузка данных: `runtime.num_workers`, `runtime.pin_memory: true`
- Gradient checkpointing: при необходимости включайте в коде отдельных модулей (в конфиге по умолчанию не задаётся)

### Оптимизация качества
- Балансировка потерь: настраивайте `loss.lambda_*` (например, `lambda_l1`, `lambda_perc`, `lambda_adv`, ...)
- Адаптивные гиперпараметры: используйте планировщик `scheduler.type: cosine`, `scheduler.warmup_steps`
- Эффективные аугментации:
```yaml
training:
  aug:
    flip_p: 0.5
    crop_scale: [0.8, 1.0]
    ab_jitter: 0.05
```

### Рекомендации по размеру батча (ориентиры)
| GPU VRAM | Размер изображения | Рекомендуемый размер батча |
|----------|--------------------|---------------------------|
| 6 GB     | 256x256           | 8-16                     |
| 8 GB     | 256x256           | 16-24                    |
| 11+ GB   | 256x256           | 24-32                    |
| 8 GB     | 512x512           | 4-8                      |
| 16+ GB   | 512x512           | 8-16                     |

## 🔍 Обучение на специфических наборах данных

### Исторические фотографии
Для специализации на исторических фотографиях:
1. Подготовьте набор данных исторических изображений
2. (Опционально) Увеличьте вес перцептуальной потери и регуляризаторов для стабильности:
```yaml
loss:
  lambda_perc: 1.5
  lambda_cc: 0.5
```
3. Запустите обучение:
```bash
python -m src.train --config configs/default.yaml
```

### Художественные произведения
Для колоризации художественных произведений:
1. Подготовьте набор данных произведений искусства
2. Рекомендуется усилить перцептуальную компоненту и ослабить GAN (если включён):
```yaml
loss:
  lambda_perc: 1.5
gan:
  enabled: false
```

## 🌐 Распределенное обучение
### Многопроцессорное обучение
Для обучения на нескольких GPU:
```bash
# Один узел, N GPU
torchrun --standalone --nproc_per_node=4 -m src.train --config configs/default.yaml
```

### Настройки распределенного обучения
```yaml
runtime:
  ddp:
    enabled: true
    backend: auto   # nccl|gloo|mpi|auto
```

## 🔧 Проблемы и решения

### Проблема: Режим коллапса (Mode Collapse)
**Симптомы**: Модель генерирует одинаковые, часто десатурированные цвета для разных изображений.

**Решение**:
1. Уменьшите вес GAN‑компоненты:
```yaml
loss:
  lambda_adv: 0.005  # пример уменьшения
```
2. Увеличьте разнообразие обучающих данных
3. При необходимости упростите архитектуру GAN (например, снизьте `gan.n_layers`).

### Проблема: Нестабильность обучения
**Симптомы**: Значительные колебания потерь, расходящееся обучение.

**Решение**:
1. Уменьшите скорости обучения:
```yaml
optimizer:
  lr_backbone: 4.0e-5
  lr_decoder_heads: 1.0e-4
```
2. Используйте градиентное отсечение (если предусмотрено в коде обучения вашей ветки).
3. Используйте плавный планировщик:
```yaml
scheduler:
  type: cosine
  warmup_steps: 1000
```

### Проблема: Недостаток памяти GPU
**Симптомы**: `CUDA out of memory` ошибки.

**Решение**:
1. Уменьшите размер батча:
```yaml
training:
  batch_size: 8  # Уменьшите с 16
```
2. Уменьшите размер входного изображения:
```yaml
training:
  image_size: 224  # Уменьшите с 256
```
3. Включите gradient checkpointing:
```yaml
# при необходимости включайте градиентный чекпоинтинг в отдельных модулях модели (если поддерживается)
```
4. Используйте смешанную точность: установите `runtime.amp: true` в конфиге.

### Проблема: Недостаточное разнообразие цветов
**Симптомы**: Модель генерирует бледные, десатурированные цвета.

**Решение**:
1. Увеличьте вклад перцептуальной потери:
```yaml
loss:
  lambda_perc: 1.5
```
2. Включите голову насыщенности (если отключена) и используйте соответствующую потерю:
```yaml
model:
  use_saturation_head: true
```
3. Усильте согласованность цвета и аугментации в ab‑пространстве:
```yaml
loss:
  lambda_cc: 0.3
training:
  aug:
    ab_jitter: 0.08
```

Для дополнительных вопросов и проблем обратитесь к Issues на GitHub или создайте новый Issue с подробным описанием вашей проблемы.
