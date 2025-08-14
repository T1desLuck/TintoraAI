# 📥 Руководство по установке TintoraAI

В этом документе представлены подробные инструкции по установке и настройке системы TintoraAI на различных платформах. Для удобства инструкции разделены по средам: локальная установка, облачные платформы и инструкции для различных операционных систем.

> Навигация: [README.md](README.md) • [TRAINING.md](TRAINING.md) • [TEST.md](TEST.md)

## 📋 Содержание

- [Системные требования](#системные-требования)
- [Локальная установка](#локальная-установка)
  - [Windows](#windows)
  - [macOS](#macos)
  - [Linux](#linux)
- [Облачная установка](#облачная-установка)
  - [Google Colab](#google-colab)
  - [Vast.ai](#vastai)
- [Установка с использованием Docker](#установка-с-использованием-docker)
- [Устранение проблем](#устранение-проблем)
- [Проверка установки](#проверка-установки)

## 💻 Системные требования

### Минимальные требования
- Python 3.9+
- 8GB RAM
- 2GB свободного дискового пространства
- NVIDIA GPU с 2–4GB VRAM для инференса (опционально)

### Рекомендуемые требования
- Python 3.9+
- 16GB RAM
- 10GB свободного дискового пространства
- NVIDIA GPU с 8GB+ VRAM для обучения (колёса PyTorch с CUDA 12.1)

### Зависимости
Основные зависимости проекта (точные версии см. `requirements.txt`):
- PyTorch 2.3.1 (+ CUDA 12.1 при наличии GPU)
- torchvision совместимой версии
- numpy, pillow, scikit-image, tqdm, tensorboard

## ⚡ Быстрый старт (локально)

1. Клонируйте и перейдите в каталог:
```bash
git clone https://github.com/T1desLuck/TintoraAI.git && cd TintoraAI
```
2. Создайте и активируйте окружение, установите зависимости:
```bash
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
3. Проверьте инференс (без чекпоинта будут случайные веса):
```bash
python -m src.inference --input data --config configs/default.yaml --output output
```
4. Результаты:
```bash
ls output   # или Get-ChildItem output на Windows
```

## 🖥️ Локальная установка

### Общие шаги

1. Клонирование репозитория:
```bash
git clone https://github.com/T1desLuck/TintoraAI.git
cd TintoraAI
```

2. Создание виртуального окружения (рекомендуется):
```bash
python -m venv venv
```

3. Установка зависимостей:
```bash
pip install -r requirements.txt
```

4. (Опционально) Проверьте установку CUDA/torch:
```bash
python -c "import torch;print(torch.__version__, 'cuda available:', torch.cuda.is_available())"
```

### Windows

#### Предварительные требования
Установите Python (версии 3.9 или выше)
Установите Git для Windows
Для использования GPU установите PyTorch с поддержкой CUDA 12.1 (колёса включают необходимые библиотеки). Отдельная установка CUDA Toolkit/cuDNN не требуется. При наличии системной CUDA рекомендуется версия 12.1.

#### Пошаговая установка
1. Откройте обычный PowerShell или CMD (права администратора не требуются):
2. Создайте директорию для проекта и перейдите в неё:
```powershell
mkdir C:\Projects
cd C:\Projects
```
3. Клонируйте репозиторий:
```powershell
git clone https://github.com/T1desLuck/TintoraAI.git
cd TintoraAI
```
4. Создайте и активируйте виртуальное окружение:
```powershell
# Создание виртуального окружения
python -m venv venv

# Активация в CMD
venv\Scripts\activate.bat

# Активация в PowerShell
venv\Scripts\Activate.ps1
```
Если PowerShell блокирует активацию скриптов, разрешите выполнение только для текущего пользователя и затем перезапустите PowerShell:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
5. Установите зависимости:
```powershell
pip install -r requirements.txt
```
6. (Опционально) Проверьте CUDA:
```powershell
python -c "import torch; print(torch.__version__, 'cuda available:', torch.cuda.is_available())"
```

### macOS

#### Предварительные требования
- Установите Homebrew (если не установлен):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
- Установите Python и Git:
```bash
brew install python git
```

#### Пошаговая установка
1. Откройте Terminal:
2. Создайте директорию для проекта и перейдите в неё:
```bash
mkdir -p ~/Projects
cd ~/Projects
```
3. Клонируйте репозиторий:
```bash
git clone https://github.com/T1desLuck/TintoraAI.git
cd TintoraAI
```
4. Создайте и активируйте виртуальное окружение:
```bash
# Создание виртуального окружения
python3 -m venv venv

# Активация
source venv/bin/activate
```
5. Установите зависимости:
```bash
pip install -r requirements.txt
```
6. (Опционально) Проверьте CUDA:
```bash
python -c "import torch; print(torch.__version__, 'cuda available:', torch.cuda.is_available())"
```

### Linux

#### Предварительные требования
- Обновите пакетный менеджер:
```bash
sudo apt update && sudo apt upgrade
```
- Установите необходимые пакеты:
```bash
sudo apt install python3-dev python3-pip git
```
- Для GPU-ускорения: убедитесь, что установлен драйвер NVIDIA, а PyTorch поставляйте через колёса с CUDA 12.1. Системная установка CUDA не обязательна.
```bash
# Пример установки PyTorch 2.3.1 с CUDA 12.1 (официальный индекс колёс)
pip3 install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```
Примечание: файл `requirements.txt` уже закрепляет совместимые версии `torch`/`torchvision`. Если вы предварительно установили PyTorch через индекс `cu121`, следующий шаг `pip install -r requirements.txt` оставит совместимые версии без переустановки.

#### Пошаговая установка
1. Создайте директорию для проекта и перейдите в неё:
```bash
mkdir -p ~/projects
cd ~/projects
```
2. Клонируйте репозиторий:
```bash
git clone https://github.com/T1desLuck/TintoraAI.git
cd TintoraAI
```
3. Создайте и активируйте виртуальное окружение:
```bash
# Создание виртуального окружения
python3 -m venv venv

# Активация
source venv/bin/activate
```
4. Установите зависимости:
```bash
pip install -r requirements.txt
```
5. (Опционально) Проверьте CUDA:
```bash
python -c "import torch; print(torch.__version__, 'cuda available:', torch.cuda.is_available())"
```

## ☁️ Облачная установка

### Google Colab
Вы можете запустить TintoraAI на Google Colab без локальной установки. Для этого:
1. Перейдите на Google Colab
2. Создайте новый блокнот
3. Вставьте и выполните следующий код:
```python
# Клонирование репозитория
!git clone https://github.com/T1desLuck/TintoraAI.git
%cd TintoraAI

# Установка зависимостей
!pip install -r requirements.txt

# Пример инференса (без чекпоинта будут случайные веса)
from google.colab import files
uploaded = files.upload()  # Загрузите изображение
fn = list(uploaded.keys())[0]
!python -m src.inference --input "$fn" --config configs/default.yaml --output output

# Скачивание результата
import os, glob
out = sorted(glob.glob('output/*'))[-1]
from google.colab import files as colab_files
colab_files.download(out)
```

### Vast.ai
Vast.ai предлагает доступные GPU для обучения и инференса моделей:
1. Создайте учетную запись на Vast.ai
2. Создайте новый экземпляр с подходящим GPU (рекомендуется минимум 8GB VRAM)
3. Выберите Docker-образ `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime` (или эквивалент с CUDA 12.1)
4. После подключения к экземпляру выполните:
```bash
# Установка необходимых пакетов
apt update && apt install -y git python3-pip python3-venv

# Клонирование репозитория
git clone https://github.com/T1desLuck/TintoraAI.git
cd TintoraAI

# Создание виртуального окружения
python3 -m venv venv
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt

# Для обучения модели (пример):
python -m src.train --config configs/default.yaml
```

## 🐳 Установка с использованием Docker
Для изолированной установки вы можете использовать Docker:
1. Установите Docker и Docker Compose
2. Клонируйте репозиторий:
```bash
git clone https://github.com/T1desLuck/TintoraAI.git
cd TintoraAI
```
3. Создайте файл `Dockerfile`:
```dockerfile
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

# Пример команды по умолчанию (можете переопределить при запуске)
CMD ["python", "-m", "src.inference", "--input", "data", "--config", "configs/default.yaml", "--output", "output"]
```
4. Соберите и запустите контейнер:
```bash
# Сборка образа
docker build -t tintoraai .

# Запуск контейнера
docker run --gpus all -it -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output tintoraai
```

## 🔧 Устранение проблем

### Ошибки CUDA
**Проблема**: `RuntimeError: CUDA error: no kernel image is available for execution on the device`

**Решение**:
```bash
# Проверьте доступность CUDA в PyTorch и версию сборки
python - << 'PY'
import torch
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA build:', torch.version.cuda)
PY

# Переустановите PyTorch с корректной сборкой CUDA 12.1 при необходимости
pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# Если ваша видеокарта слишком старая для текущих колёс — используйте CPU или тайловый инференс
```

### Ошибки установки зависимостей
**Проблема**: Ошибки при установке зависимостей из `requirements.txt`

**Решение**:
```bash
# Установите пакеты по одному
pip install --no-cache-dir numpy
pip install --no-cache-dir torch==1.10.0
# ...и так далее

# Для проблем с компиляцией C-расширений на Windows
pip install --upgrade pip setuptools wheel
```

### Недостаточно памяти GPU
**Проблема**: `CUDA out of memory`

**Решение**:
```bash
# Используйте тайловый инференс (плитки) для экономии памяти
python -m src.inference --input input.jpg --config configs/default.yaml --tile 256 --overlap 32

# Или уменьшите размер входного изображения заранее
# например, до 128–256 по меньшей стороне
```

## ✅ Проверка установки
Для проверки корректности установки выполните следующие команды:

1. Проверка доступности CUDA:
```bash
python -c "import torch; print('CUDA доступен:', torch.cuda.is_available())"
```

2. Проверка базового функционала:
```bash
python -m src.inference --help
```

3. Запуск тестов:
```bash
pytest tests/
```

4. Проверка инференса на своём изображении:
```bash
# Поместите одно или несколько ваших изображений в каталог data/ и запустите инференс
python -m src.inference --input data --config configs/default.yaml --output output
```

Примечания:
- Если не указать `--output`, по умолчанию результаты сохраняются в каталог, заданный конфигом: `paths.experiments` (или `paths.logs` как резерв) с подпапкой `val_pred`. Это соответствует логике `src/inference.py`.
- По умолчанию загрузится чекпоинт `checkpoints/latest.pth` (путь можно задать в `configs/default.yaml` через `paths` и `checkpointing`). Если файл отсутствует, скрипт честно предупредит и запустит модель со случайными весами.

5. Проверьте результат (файлы появятся в каталоге output):
```bash
# Для Linux/macOS
ls output

# Для Windows (PowerShell)
Get-ChildItem output
```

Если все шаги выполнились без ошибок и вы видите колоризованное изображение, значит установка прошла успешно!

## 🔄 Обновление
Для обновления TintoraAI до последней версии:
```bash
# Перейдите в директорию проекта
cd path/to/TintoraAI

# Получите последние изменения
git pull

# Обновите зависимости
pip install -r requirements.txt --upgrade

# Дополнительная настройка не требуется — структура актуальна
```

Если у вас остались вопросы или возникли проблемы с установкой, пожалуйста, создайте Issue на GitHub или обратитесь в нашу команду поддержки.