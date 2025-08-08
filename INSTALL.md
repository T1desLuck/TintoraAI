# 📥 Руководство по установке TintoraAI

В этом документе представлены подробные инструкции по установке и настройке системы TintoraAI на различных платформах. Для удобства инструкции разделены по средам: локальная установка, облачные платформы и инструкции для различных операционных систем.

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
- Python 3.7 или выше
- 8GB RAM
- 2GB свободного дискового пространства
- NVIDIA GPU с 2GB VRAM для инференса (опционально)

### Рекомендуемые требования
- Python 3.8 или выше
- 16GB RAM
- 10GB свободного дискового пространства
- NVIDIA GPU с 8GB+ VRAM для обучения (CUDA 11.1+)

### Зависимости
Основные зависимости проекта:
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- NumPy >= 1.19.5
- OpenCV >= 4.5.3
- Pillow >= 9.0.0
- scikit-image >= 0.18.0
- и другие, указанные в файле `requirements.txt`

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

4. Запуск скрипта настройки для создания структуры директорий:
```bash
python scripts/setup.py
```

### Windows

#### Предварительные требования
- Установите Python (версии 3.7 или выше)
- Установите Git для Windows
- Для использования GPU установите CUDA Toolkit и cuDNN

#### Пошаговая установка
1. Откройте командную строку Windows (CMD) или PowerShell от имени администратора:
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
5. Установите зависимости:
```powershell
pip install -r requirements.txt
```
6. Запустите скрипт настройки:
```powershell
python scripts\setup.py
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
6. Запустите скрипт настройки:
```bash
python scripts/setup.py
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
- Для GPU-ускорения установите CUDA и cuDNN (для Ubuntu):
```bash
# Добавление репозитория NVIDIA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"

# Установка CUDA
sudo apt update
sudo apt install cuda-11-3
```

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
5. Запустите скрипт настройки:
```bash
python scripts/setup.py
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

# Настройка проекта
!python scripts/setup.py

# Проверка установки
!python scripts/demo.py --checkpoint demo_model.pth --console

# Пример инференса
from google.colab import files
uploaded = files.upload()  # Загрузите изображение
import os
filename = list(uploaded.keys())[0]
!python scripts/inference.py --single {filename} --checkpoint demo_model.pth --output-dir output

# Скачивание результата
files.download('output/colorized/' + os.path.splitext(filename)[0] + '.png')
```

### Vast.ai
Vast.ai предлагает доступные GPU для обучения и инференса моделей:
1. Создайте учетную запись на Vast.ai
2. Создайте новый экземпляр с подходящим GPU (рекомендуется минимум 8GB VRAM)
3. Выберите Docker-образ `nvidia/cuda:11.3-cudnn8-runtime-ubuntu20.04`
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

# Настройка проекта
python scripts/setup.py

# Для обучения модели (пример):
python scripts/train.py --config configs/training_config.yaml
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
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN python scripts/setup.py

CMD ["python", "scripts/demo.py", "--console"]
```
4. Соберите и запустите контейнер:
```bash
# Сборка образа
docker build -t tintorai .

# Запуск контейнера
docker run --gpus all -it -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output tintorai
```

## 🔧 Устранение проблем

### Ошибки CUDA
**Проблема**: `RuntimeError: CUDA error: no kernel image is available for execution on the device`

**Решение**:
```bash
# Проверьте версию CUDA
nvcc --version

# Убедитесь, что PyTorch установлен с поддержкой соответствующей версии CUDA
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
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
# Уменьшите размер батча в configs/inference_config.yaml
# batch_size: 4  # уменьшите это значение

# Или запустите с меньшим размером изображения
python scripts/inference.py --single input.jpg --img-size 128
```

## ✅ Проверка установки
Для проверки корректности установки выполните следующие команды:

1. Проверка доступности CUDA:
```bash
python -c "import torch; print('CUDA доступен:', torch.cuda.is_available())"
```

2. Проверка базового функционала:
```bash
python scripts/setup.py --verbose --gpu-check
```

3. Запуск тестов:
```bash
pytest tests/
```

4. Проверка инференса на примере:
```bash
# Скопируйте тестовое изображение
cp data/samples/example1.jpg input/single/

# Выполните инференс
python scripts/inference.py --single input/single/example1.jpg --output-dir output
```

5. Проверьте результат:
```bash
# Для Linux/macOS
open output/colorized/example1.png

# Для Windows
start output\colorized\example1.png
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

# Обновите структуру директорий
python scripts/setup.py --force
```

Если у вас остались вопросы или возникли проблемы с установкой, пожалуйста, создайте Issue на GitHub или обратитесь в нашу команду поддержки.