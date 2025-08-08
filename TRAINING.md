# 🎯 Руководство по обучению модели TintoraAI

Это руководство описывает полный процесс подготовки данных, настройки и обучения модели колоризации TintoraAI, включая продвинутые техники для достижения наилучших результатов.

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

1. **Подготовка данных** - Сбор и обработка пар черно-белых и цветных изображений
2. **Настройка гиперпараметров** - Определение архитектуры и параметров обучения
3. **Обучение модели** - Итеративное обучение с отслеживанием прогресса
4. **Валидация** - Оценка результатов на отдельном наборе данных
5. **Тонкая настройка** - Корректировка параметров на основе результатов
6. **Финальная оценка** - Тестирование итоговой модели

## 🖼️ Подготовка данных

### Структура данных

Для обучения TintoraAI требуются пары изображений: черно-белые (входные) и соответствующие им цветные (целевые). Данные должны быть организованы следующим образом:

```
data/
├── train/
│   ├── grayscale/  # Черно-белые изображения для обучения
│   └── color/      # Цветные изображения для обучения
├── val/
│   ├── grayscale/  # Черно-белые изображения для валидации
│   └── color/      # Цветные изображения для валидации
├── test/
│   ├── grayscale/  # Черно-белые изображения для тестирования
│   └── color/      # Цветные изображения для тестирования
└── reference/      # Эталонные изображения для стилей
    ├── historical/ # Исторические стили (1920s, 1950s)
    ├── artistic/   # Художественные стили
    └── natural/    # Естественные цветовые палитры
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
mkdir -p data/raw data/train/color data/train/grayscale data/val/color data/val/grayscale
```

2. Преобразование в черно-белые:
```python
# Пример скрипта для преобразования
import os
import cv2
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split

# Загрузка изображений
color_images = glob('data/raw/*.jpg')

# Разделение на обучающую и валидационную выборки
train_images, val_images = train_test_split(color_images, test_size=0.1, random_state=42)

# Обработка обучающей выборки
for img_path in train_images:
    # Загрузка цветного изображения
    img = cv2.imread(img_path)
    
    # Создание черно-белой версии
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Сохранение обеих версий
    filename = os.path.basename(img_path)
    cv2.imwrite(f'data/train/color/{filename}', img)
    cv2.imwrite(f'data/train/grayscale/{filename}', gray)

# Аналогично для валидационной выборки
# ...
```

3. Аугментация данных: Расширьте набор данных с помощью аугментаций, указанных в `configs/training_config.yaml`.

4. Проверка данных
Перед обучением убедитесь, что данные корректно подготовлены:
```bash
# Проверка количества изображений
echo "Тренировочные цветные: $(ls data/train/color | wc -l)"
echo "Тренировочные черно-белые: $(ls data/train/grayscale | wc -l)"
echo "Валидационные цветные: $(ls data/val/color | wc -l)"
echo "Валидационные черно-белые: $(ls data/val/grayscale | wc -l)"

# Проверка соответствия имен файлов
python -c "import os; train_color = set(os.listdir('data/train/color')); train_gray = set(os.listdir('data/train/grayscale')); print(f'Пересечение: {len(train_color.intersection(train_gray))}'); print(f'Различия: {train_color.symmetric_difference(train_gray)}')"
```

## ⚙️ Настройка конфигурации

Для настройки процесса обучения используются конфигурационные файлы в директории `configs/`. Рассмотрим основные параметры, которые можно настроить:

### Основная конфигурация модели (`model_config.yaml`)
```yaml
# Пример изменения параметров модели
model:
  img_size: 256  # Размер входного изображения
  
# Настройка Swin-UNet
swin_unet:
  embed_dim: 96
  depths: [2, 2, 6, 2]
  num_heads: [3, 6, 12, 24]

# Настройка ViT
vit_semantic:
  embed_dim: 768
  depth: 12
  num_heads: 12

# Включение/отключение модулей
guide_net:
  enabled: true
  
discriminator:
  enabled: true
```

### Конфигурация обучения (`training_config.yaml`)
```yaml
# Параметры обучения
training:
  batch_size: 16  # Уменьшите для экономии памяти
  epochs: 100
  mixed_precision: true  # Использовать AMP для ускорения

# Оптимизатор
optimizer:
  type: "adam"
  lr: 0.0002
  beta1: 0.5
  beta2: 0.999
  
# Планировщик
scheduler:
  type: "cosine"
  warmup_epochs: 5
```

### Конфигурация функций потерь (`loss_config.yaml`)
```yaml
# Веса различных компонентов потери
l1_loss:
  enabled: true
  weight: 10.0
  
vgg_perceptual:
  enabled: true
  weight: 1.0
  
gan_loss:
  enabled: true
  weight: 0.1

# Динамическая балансировка
dynamic_balancer:
  enabled: true
  target_metric: "lpips"
```

## 🚀 Запуск обучения

### Базовое обучение
После подготовки данных и настройки конфигурации, запустите обучение:
```bash
# Активация виртуального окружения (если используется)
source venv/bin/activate  # Linux/macOS
# или
venv\Scripts\activate     # Windows

# Запуск обучения с базовыми параметрами
python scripts/train.py --config configs/training_config.yaml
```

### Дополнительные параметры обучения
```bash
# Обучение с указанием модели
python scripts/train.py --config configs/training_config.yaml --model-config configs/model_config.yaml

# Возобновление обучения с чекпоинта
python scripts/train.py --resume --checkpoint experiments/checkpoints/latest.pth

# Обучение на GPU с определенным ID
python scripts/train.py --gpu 0

# Запуск с подробным выводом
python scripts/train.py --verbose
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
!python scripts/train.py --config configs/training_config.yaml --gpu 0
```

### Запуск обучения на Vast.ai
```bash
# На сервере Vast.ai
git clone https://github.com/T1desLuck/TintoraAI.git
cd TintoraAI
pip install -r requirements.txt

# Запуск обучения с несколькими GPU
CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py --distributed --world-size 2
```

## 📈 Мониторинг и визуализация

### TensorBoard
Во время обучения TintoraAI автоматически логирует метрики и визуализации в TensorBoard:
```bash
# Запуск TensorBoard
tensorboard --logdir experiments/logs/tensorboard

# Затем откройте в браузере: http://localhost:6006
```

Что можно отслеживать:
- Метрики потерь: общая потеря и компоненты (L1, VGG, GAN, PatchNCE)
- Метрики качества: PSNR, SSIM, LPIPS
- Визуализации: примеры колоризации и сравнения
- Гистограммы: распределение весов и градиентов
- Профили: использование памяти и время выполнения

### Журналы обучения
Логи обучения сохраняются в директорию `experiments/logs/`:
```bash
# Просмотр последних логов обучения
cat experiments/logs/training.log | tail -50
```

### Контрольные точки
Контрольные точки (чекпоинты) сохраняются в директорию `experiments/checkpoints/`:
```bash
# Список доступных чекпоинтов
ls -lh experiments/checkpoints/
```

## 🧠 Продвинутые техники обучения

### Настройка системы "наград и наказаний"
Уникальная особенность TintoraAI — система "наград и наказаний", которую можно настроить в `loss_config.yaml`:
```yaml
reward_system:
  enabled: true
  guide_net:  # Настройки для GuideNet
    reward_weight: 1.0
    penalty_weight: 1.5
  discriminator:  # Настройки для Discriminator
    reward_weight: 1.0
    penalty_weight: 1.2
```

### Few-shot адаптация
Для обучения с малым количеством примеров используйте:
```bash
# Запуск few-shot адаптации
python scripts/train.py --config configs/training_config.yaml --shot-number 5 --adaptation-steps 10
```

Настройки в конфигурационном файле:
```yaml
adaptable:
  enabled: true
  adapter_type: "standard"
  bottleneck_dim: 64
  shot_number: 5
```

### Перенос стиля во время обучения
Для обучения с переносом стиля:
```bash
python scripts/train.py --config configs/training_config.yaml --style-transfer
```

Настройки стилей:
```yaml
style_transfer:
  enabled: true
  content_weight: 1.0
  style_weight: 100.0
```

### Использование Memory Bank
Memory Bank позволяет модели учиться на исторических примерах:
```yaml
memory_bank:
  enabled: true
  feature_dim: 256
  max_items: 1000
  use_fusion: true
```

## 🎯 Оценка обученной модели
После обучения важно оценить качество модели:
```bash
# Оценка модели на тестовом наборе данных
python scripts/evaluate.py --checkpoint experiments/checkpoints/best_model.pth --test-dir data/test

# Визуализация результатов оценки
python scripts/evaluate.py --checkpoint experiments/checkpoints/best_model.pth --test-dir data/test --save-comparison --plot-metrics
```

### Метрики качества
TintoraAI использует следующие метрики для оценки качества колоризации:
- PSNR (Peak Signal-to-Noise Ratio) - чем выше, тем лучше
- SSIM (Structural Similarity Index) - цель ≥ 0.82
- LPIPS (Learned Perceptual Image Patch Similarity) - цель ≤ 0.20
- FID (Fréchet Inception Distance) - чем ниже, тем лучше

### Анализ по категориям
Для более детального анализа используйте категории изображений:
```bash
python scripts/evaluate.py --checkpoint experiments/checkpoints/best_model.pth --test-dir data/test --categories data/test/categories.json
```

## 💡 Советы по оптимизации

### Оптимизация скорости обучения
- Использование смешанной точности (AMP):
```yaml
training:
  mixed_precision: true
```
- Оптимизация загрузки данных:
```yaml
data:
  num_workers: 8
  prefetch_factor: 2
  pin_memory: true
```
- Gradient checkpointing для экономии памяти:
```yaml
swin_unet:
  use_checkpoint: true
```

### Оптимизация качества
- Правильная балансировка потерь:
```yaml
dynamic_balancer:
  enabled: true
  learning_rate: 0.01
```
- Адаптивное изменение гиперпараметров:
```yaml
scheduler:
  type: "cosine"
  warmup_epochs: 5
```
- Эффективные аугментации:
```yaml
augmentations:
  horizontal_flip: true
  rotate: true
  brightness: 0.1
  contrast: 0.1
```

### Рекомендации по размеру батча
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
2. Настройте параметры в конфигурации:
```yaml
# В training_config.yaml
data:
  reference_dataset_path: "data/reference/historical"
  
# В model_config.yaml
memory_bank:
  enabled: true
  feature_dim: 256
```
3. Запустите обучение с дополнительными параметрами:
```bash
python scripts/train.py --config configs/training_config.yaml --batch-size 8 --style historical
```

### Художественные произведения
Для колоризации художественных произведений:
1. Подготовьте набор данных произведений искусства
2. Настройте параметры:
```yaml
# В training_config.yaml
data:
  reference_dataset_path: "data/reference/artistic"
  
# В model_config.yaml
style_transfer:
  enabled: true
  content_weight: 1.0
  style_weight: 150.0  # Увеличенный вес для стиля
```

## 🌐 Распределенное обучение
### Многопроцессорное обучение
Для обучения на нескольких GPU:
```bash
# С указанием GPU-устройств
python scripts/train.py --distributed --device-ids 0,1,2,3

# Автоматическое определение устройств
python scripts/train.py --distributed --world-size 4
```

### Настройки распределенного обучения
```yaml
# В training_config.yaml
training:
  distributed: true
  sync_bn: true
```

## 🔧 Проблемы и решения

### Проблема: Режим коллапса (Mode Collapse)
**Симптомы**: Модель генерирует одинаковые, часто десатурированные цвета для разных изображений.

**Решение**:
1. Уменьшите вес GAN-потери:
```yaml
gan_loss:
  weight: 0.05  # Уменьшите с 0.1
```
2. Увеличьте разнообразие обучающих данных
3. Используйте spectrum normalization в дискриминаторе:
```yaml
discriminator:
  use_spectral_norm: true
```

### Проблема: Нестабильность обучения
**Симптомы**: Значительные колебания потерь, расходящееся обучение.

**Решение**:
1. Уменьшите скорость обучения:
```yaml
optimizer:
  lr: 0.0001  # Уменьшите с 0.0002
```
2. Используйте градиентное отсечение:
```yaml
training:
  max_grad_norm: 1.0
```
3. Используйте более плавный планировщик:
```yaml
scheduler:
  type: "step"
  step_size: 30
  gamma: 0.5
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
model:
  img_size: 224  # Уменьшите с 256
```
3. Включите gradient checkpointing:
```yaml
swin_unet:
  use_checkpoint: true
```
4. Используйте смешанную точность:
```bash
python scripts/train.py --mixed-precision
```

### Проблема: Недостаточное разнообразие цветов
**Симптомы**: Модель генерирует бледные, десатурированные цвета.

**Решение**:
1. Увеличьте вес VGG perceptual loss:
```yaml
vgg_perceptual:
  weight: 1.5  # Увеличьте с 1.0
```
2. Увеличьте параметр насыщенности:
```yaml
postprocessing:
  saturation: 1.2  # Увеличьте с 1.0
```
3. Усильте PatchNCE loss для более контрастного обучения:
```yaml
patch_nce:
  weight: 1.5
```

Для дополнительных вопросов и проблем обратитесь к Issues на GitHub или создайте новый Issue с подробным описанием вашей проблемы.