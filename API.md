# 🌐 API документация TintoraAI

В этом документе представлена полная документация по использованию API системы TintoraAI для колоризации изображений. API позволяет интегрировать функциональность колоризации в ваши приложения, сервисы и рабочие процессы.

## 📋 Содержание

- [Обзор API](#обзор-api)
- [Начало работы](#начало-работы)
- [Endpoints](#endpoints)
- [Запуск API-сервера](#запуск-api-сервера)
- [Примеры запросов](#примеры-запросов)
- [SDK для Python](#sdk-для-python)
- [Пакетная обработка](#пакетная-обработка)
- [Обработка потоковых данных](#обработка-потоковых-данных)
- [Настройка и конфигурация](#настройка-и-конфигурация)
- [Обработка ошибок](#обработка-ошибок)
- [Примеры интеграции](#примеры-интеграции)

## 📝 Обзор API

API TintoraAI предоставляет REST-интерфейс для колоризации изображений с возможностью выбора различных стилей, настройки параметров обработки и получения детальной информации о результатах. API также поддерживает пакетную обработку и асинхронные запросы.

### Основные возможности API

- 🖼️ Колоризация отдельных изображений
- 📚 Пакетная обработка множества изображений
- 🎨 Выбор и настройка стилей колоризации
- 📊 Получение карт неопределенности и метаданных
- 📱 Поддержка уведомлений о завершении обработки
- 🔐 Аутентификация и ограничение доступа (опционально)

## 🚀 Начало работы

### Требования

- Python 3.7+
- FastAPI
- PyTorch 1.9+
- TintoraAI с предварительно обученной моделью

### Установка

```bash
# Клонирование репозитория
git clone https://github.com/T1desLuck/TintoraAI.git
cd TintoraAI

# Установка зависимостей
pip install -r requirements.txt

# Запуск API-сервера
python API.py --checkpoint experiments/checkpoints/model.pth --port 8000
```

## 🔄 Endpoints

API TintoraAI предоставляет следующие endpoints:

1. **GET /**
   Возвращает базовую информацию об API.

   **Ответ**:
   ```json
   {
     "name": "TintoraAI API",
     "version": "1.0.0",
     "status": "active",
     "model_loaded": true
   }
   ```

2. **GET /health**
   Проверяет состояние API и модели.

   **Ответ**:
   ```json
   {
     "status": "healthy",
     "model_status": "loaded",
     "load_error": null,
     "active_batches": 0,
     "queue_length": 0,
     "timestamp": "2023-08-08T12:34:56"
   }
   ```

3. **POST /colorize**
   Колоризует одно изображение и возвращает результат.

   **Запрос**:
   ```json
   {
     "image_data": "base64_encoded_image_data",
     "style": "default",
     "color_space": "lab",
     "saturation": 1.0,
     "contrast": 1.0,
     "enhance": false,
     "show_uncertainty": false,
     "style_strength": 1.0,
     "guidance": "ЦветСтиль:винтаж"
   }
   ```

   **Ответ**:
   ```json
   {
     "status": "success",
     "message": "Изображение успешно колоризовано",
     "request_id": "550e8400-e29b-41d4-a716-446655440000",
     "processing_time": 0.856,
     "colorized_image": "base64_encoded_colorized_image",
     "uncertainty_map": "base64_encoded_uncertainty_map",
     "metadata": {
       "style": "default",
       "color_space": "lab",
       "saturation": 1.0,
       "contrast": 1.0,
       "enhance": false,
       "device": "cuda:0"
     }
   }
   ```

4. **POST /batch/colorize**
   Начинает пакетную колоризацию нескольких изображений.

   **Запрос**:
   ```json
   {
     "images": ["base64_image1", "base64_image2", "base64_image3"],
     "is_urls": false,
     "style": "vintage",
     "color_space": "lab",
     "saturation": 1.2,
     "contrast": 1.1,
     "enhance": true,
     "notify_url": "https://example.com/webhook",
     "callback_id": "job123",
     "priority": 1
   }
   ```

   **Ответ**:
   ```json
   {
     "status": "accepted",
     "batch_id": "550e8400-e29b-41d4-a716-446655440000",
     "message": "Пакет добавлен в очередь. Для проверки статуса используйте /batch/status/550e8400-e29b-41d4-a716-446655440000"
   }
   ```

5. **GET /batch/status/{batch_id}**
   Получает статус пакетной обработки.

   **Ответ**:
   ```json
   {
     "batch_id": "550e8400-e29b-41d4-a716-446655440000",
     "status": "processing",
     "total_images": 3,
     "processed_images": 1,
     "progress_percent": 33.33,
     "estimated_time_left": 4.5,
     "errors": []
   }
   ```

6. **GET /batch/results/{batch_id}**
   Получает результаты пакетной обработки.

   **Ответ**:
   ```json
   {
     "batch_id": "550e8400-e29b-41d4-a716-446655440000",
     "status": "completed",
     "total_images": 3,
     "processed_images": 3,
     "results": [
       {
         "index": 0,
         "colorized_image": "base64_encoded_image"
       },
       {
         "index": 1,
         "colorized_image": "base64_encoded_image"
       },
       {
         "index": 2,
         "colorized_image": "base64_encoded_image"
       }
     ],
     "errors": [],
     "processing_time": 2.56
   }
   ```

7. **GET /styles**
   Получает список доступных стилей колоризации.

   **Ответ**:
   ```json
   {
     "styles": [
       {
         "id": "default",
         "name": "Стандартный",
         "description": "Естественная колоризация с нейтральными цветами"
       },
       {
         "id": "vintage",
         "name": "Винтаж",
         "description": "Старинный вид с мягкими пастельными тонами"
       },
       {
         "id": "vivid",
         "name": "Яркий",
         "description": "Насыщенные цвета для яркого впечатления"
       }
     ]
   }
   ```

## ⚙️ Запуск API-сервера

### Локальный запуск
```bash
# Базовый запуск
python API.py --checkpoint experiments/checkpoints/model.pth

# Запуск с указанием хоста и порта
python API.py --checkpoint experiments/checkpoints/model.pth --host 0.0.0.0 --port 8000

# Запуск с несколькими рабочими процессами
python API.py --checkpoint experiments/checkpoints/model.pth --workers 4
```

### Запуск в Docker
```bash
# Сборка Docker-образа
docker build -t tintorai -f Dockerfile.api .

# Запуск контейнера
docker run -p 8000:8000 -v $(pwd)/experiments:/app/experiments tintorai --checkpoint experiments/checkpoints/model.pth
```

### Запуск на облачных платформах

#### Heroku
```bash
# Создание Heroku app
heroku create tintorai-api

# Настройка buildpacks
heroku buildpacks:add https://github.com/heroku/heroku-buildpack-python

# Деплой
git push heroku main
```

#### Google Cloud Run
```bash
# Сборка образа для Cloud Run
gcloud builds submit --tag gcr.io/your-project/tintorai-api

# Деплой на Cloud Run
gcloud run deploy tintorai-api --image gcr.io/your-project/tintorai-api --platform managed
```

## 📝 Примеры запросов

### Колоризация изображения (curl)
```bash
curl -X POST "http://localhost:8000/colorize" \
     -H "Content-Type: application/json" \
     -d '{
           "image_data": "'$(base64 -w0 input.jpg)'",
           "style": "vintage",
           "saturation": 1.2,
           "show_uncertainty": true
         }' \
     --output response.json

# Сохранение результата
cat response.json | jq -r '.colorized_image' | sed 's/data:image\/png;base64,//' | base64 -d > colorized_output.png
```

### Колоризация изображения (Python)
```python
import requests
import base64
import json
from PIL import Image
from io import BytesIO

# Загрузка изображения
with open("input.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

# Подготовка запроса
request_data = {
    "image_data": image_data,
    "style": "vintage",
    "saturation": 1.2,
    "contrast": 1.1,
    "show_uncertainty": True
}

# Отправка запроса
response = requests.post(
    "http://localhost:8000/colorize",
    json=request_data
)

# Обработка ответа
if response.status_code == 200:
    data = response.json()
    
    # Сохранение колоризованного изображения
    colorized_image_data = data["colorized_image"].split(",")[1]
    colorized_image = Image.open(BytesIO(base64.b64decode(colorized_image_data)))
    colorized_image.save("colorized_output.png")
    
    # Сохранение карты неопределенности, если доступна
    if "uncertainty_map" in data:
        uncertainty_map_data = data["uncertainty_map"].split(",")[1]
        uncertainty_map = Image.open(BytesIO(base64.b64decode(uncertainty_map_data)))
        uncertainty_map.save("uncertainty_map.png")
    
    print(f"Время обработки: {data['processing_time']:.3f} секунд")
else:
    print(f"Ошибка: {response.status_code} - {response.text}")
```

### Пакетная колоризация (Python)
```python
import requests
import base64
import json
import time
import os
from PIL import Image
from io import BytesIO

# Функция для кодирования изображений в base64
def encode_images(image_paths):
    encoded_images = []
    for path in image_paths:
        with open(path, "rb") as f:
            encoded_images.append(base64.b64encode(f.read()).decode("utf-8"))
    return encoded_images

# Изображения для колоризации
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
encoded_images = encode_images(image_paths)

# Отправляем запрос на пакетную колоризацию
response = requests.post(
    "http://localhost:8000/batch/colorize",
    json={
        "images": encoded_images,
        "is_urls": False,
        "style": "vintage",
        "saturation": 1.2,
        "contrast": 1.1
    }
)

batch_id = response.json()["batch_id"]
print(f"Batch ID: {batch_id}")

# Проверяем статус обработки
while True:
    status_response = requests.get(f"http://localhost:8000/batch/status/{batch_id}")
    status_data = status_response.json()
    
    print(f"Прогресс: {status_data['processed_images']}/{status_data['total_images']} "
          f"({status_data['progress_percent']:.1f}%)")
    
    if status_data["status"] in ["completed", "failed"]:
        break
        
    time.sleep(1)

# Получаем результаты
results_response = requests.get(f"http://localhost:8000/batch/results/{batch_id}")
results_data = results_response.json()

# Сохраняем результаты
if results_data["status"] == "completed":
    os.makedirs("batch_results", exist_ok=True)
    
    for i, result in enumerate(results_data["results"]):
        img_data = result["colorized_image"].split(",")[1]
        img = Image.open(BytesIO(base64.b64decode(img_data)))
        img.save(f"batch_results/result_{i}.png")
    
    print(f"Результаты сохранены в директорию 'batch_results/'")
else:
    print(f"Ошибка при обработке: {results_data['errors']}")
```

## 🛠️ SDK для Python

TintoraAI предоставляет Python SDK для удобной интеграции:
```python
from API import TintoraAISDK

# Инициализация SDK
sdk = TintoraAISDK(checkpoint_path="experiments/checkpoints/model.pth")

# Колоризация одного изображения
result = sdk.colorize_image(
    image_path="input.jpg",
    output_path="colorized_output.png",
    style="vintage",
    saturation=1.2,
    contrast=1.1
)

# Колоризация всех изображений в директории
stats = sdk.colorize_directory(
    input_dir="input/batch",
    output_dir="output/colorized",
    style="default",
    recursive=True
)

print(f"Успешно обработано {stats['success']} из {stats['total']} изображений")
```

## 📚 Пакетная обработка

### Отправка пакета на обработку
```python
import requests
import base64
import json
import time
import os

# Функция для кодирования изображений
def encode_images_from_dir(directory):
    encoded_images = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            with open(os.path.join(directory, filename), "rb") as f:
                encoded_images.append(base64.b64encode(f.read()).decode("utf-8"))
    return encoded_images

# Получение списка изображений
encoded_images = encode_images_from_dir("input/batch")

# Отправка запроса
response = requests.post(
    "http://localhost:8000/batch/colorize",
    json={
        "images": encoded_images,
        "is_urls": False,
        "style": "vintage",
        "saturation": 1.2,
        "contrast": 1.1,
        "notify_url": "https://example.com/webhook",
        "callback_id": "batch_job_123",
        "priority": 2
    }
)

batch_id = response.json()["batch_id"]
print(f"Пакетная обработка запущена. ID: {batch_id}")

# Дождитесь завершения или периодически проверяйте статус
```

## 🌊 Обработка потоковых данных

TintoraAI API поддерживает обработку потоковых данных для случаев, когда изображения поступают из постоянного источника.

### Настройка очереди
Для непрерывной обработки используйте директорию очереди:
1. Поместите изображения в директорию `input/queue/`
2. API будет автоматически обрабатывать новые файлы и сохранять результаты в `output/colorized/`
3. Для активации этого режима:
```bash
# Запуск API в режиме наблюдения за очередью
python API.py --checkpoint experiments/checkpoints/model.pth --watch-queue --queue-interval 5
```

### WebSocket API для потоковой передачи
API также предоставляет WebSocket-интерфейс для потоковых данных:
```python
import asyncio
import websockets
import json
import base64

async def stream_images():
    uri = "ws://localhost:8000/ws/colorize"
    
    async with websockets.connect(uri) as websocket:
        # Настройка сессии
        await websocket.send(json.dumps({
            "type": "config",
            "style": "vintage",
            "saturation": 1.2
        }))
        
        # Отправка изображений в потоке
        for i in range(5):
            # Чтение изображения
            with open(f"input/stream/image{i}.jpg", "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            
            # Отправка изображения
            await websocket.send(json.dumps({
                "type": "image",
                "image_data": image_data,
                "frame_id": i
            }))
            
            # Получение результата
            response = await websocket.recv()
            data = json.loads(response)
            
            print(f"Получено колоризованное изображение для кадра {data['frame_id']}")
            
            # Сохранение результата
            colorized_data = data["colorized_image"].split(",")[1]
            with open(f"output/stream/colorized_{i}.jpg", "wb") as f:
                f.write(base64.b64decode(colorized_data))

asyncio.run(stream_images())
```

## ⚙️ Настройка и конфигурация

### Конфигурация API через YAML
API TintoraAI можно настроить через конфигурационный файл `configs/inference_config.yaml`:
```yaml
api:
  # Настройки API
  enable_server: true
  host: "0.0.0.0"
  port: 8000
  workers: 1
  
  # Безопасность
  cors_origins: ["*"]
  api_key_required: false
  api_key: ""
  
  # Ограничения
  max_file_size_mb: 10
  rate_limit: 60
  timeout: 30
```

### Переопределение настроек через переменные окружения
```bash
# Установка порта через переменные окружения
export TINTORAI_API_PORT=9000
export TINTORAI_API_HOST=0.0.0.0
export TINTORAI_API_KEY_REQUIRED=true
export TINTORAI_API_KEY=your_secret_api_key

# Запуск API с настройками из переменных окружения
python API.py --checkpoint experiments/checkpoints/model.pth
```

### Настройка аутентификации
Для включения аутентификации добавьте API-ключ в конфигурацию:
```yaml
api:
  api_key_required: true
  api_key: "your_secret_api_key"
```

И при выполнении запросов используйте заголовок:
```bash
curl -X POST "http://localhost:8000/colorize" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: your_secret_api_key" \
     -d '{
           "image_data": "base64_encoded_image",
           "style": "default"
         }'
```

## 🚨 Обработка ошибок

API TintoraAI использует стандартные HTTP коды ответов:

| Код | Значение              | Описание                            |
|-----|-----------------------|-------------------------------------|
| 200 | OK                    | Запрос успешно обработан            |
| 202 | Accepted              | Запрос принят на пакетную обработку |
| 400 | Bad Request           | Неверные параметры запроса          |
| 401 | Unauthorized          | Не предоставлен или неверный API-ключ |
| 404 | Not Found             | Ресурс не найден                   |
| 413 | Payload Too Large     | Слишком большой файл изображения    |
| 429 | Too Many Requests     | Превышен лимит запросов            |
| 500 | Internal Server Error | Внутренняя ошибка сервера          |
| 503 | Service Unavailable   | Сервис временно недоступен         |

### Примеры обработки ошибок
```python
import requests

try:
    response = requests.post(
        "http://localhost:8000/colorize",
        json={
            "image_data": "invalid_base64_data",
            "style": "default"
        }
    )
    
    # Проверка статуса
    response.raise_for_status()
    
    data = response.json()
    print(f"Успешно: {data['message']}")
    
except requests.exceptions.HTTPError as e:
    status_code = e.response.status_code
    error_data = e.response.json()
    
    if status_code == 400:
        print(f"Ошибка в запросе: {error_data['detail']}")
    elif status_code == 401:
        print("Ошибка авторизации: Неверный API-ключ")
    elif status_code == 429:
        print(f"Превышен лимит запросов. Повторите через {error_data.get('retry_after', 60)} секунд")
    else:
        print(f"Ошибка сервера: {error_data['detail']}")
```

## 🔌 Примеры интеграции

### Интеграция с веб-приложением (JavaScript)
```javascript
// Функция для колоризации изображения из формы
async function colorizeImage() {
    const fileInput = document.getElementById('imageInput');
    const styleSelect = document.getElementById('styleSelect');
    const resultImage = document.getElementById('resultImage');
    const loadingIndicator = document.getElementById('loading');
    
    if (!fileInput.files || !fileInput.files[0]) {
        alert('Пожалуйста, выберите изображение');
        return;
    }
    
    const file = fileInput.files[0];
    const reader = new FileReader();
    
    reader.onload = async function(e) {
        // Получаем base64 содержимое
        const base64Image = e.target.result.split(',')[1];
        
        // Показываем индикатор загрузки
        loadingIndicator.style.display = 'block';
        resultImage.style.display = 'none';
        
        try {
            // Отправляем запрос к API
            const response = await fetch('http://localhost:8000/colorize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image_data: base64Image,
                    style: styleSelect.value,
                    saturation: 1.2,
                    show_uncertainty: false
                })
            });
            
            // Проверяем статус ответа
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Ошибка при колоризации');
            }
            
            // Получаем результат
            const data = await response.json();
            
            // Отображаем результат
            resultImage.src = data.colorized_image;
            resultImage.style.display = 'block';
            
            // Сохраняем метаданные
            document.getElementById('metadata').textContent = 
                JSON.stringify(data.metadata, null, 2);
                
        } catch (error) {
            console.error('Ошибка:', error);
            alert(`Ошибка: ${error.message}`);
        } finally {
            // Скрываем индикатор загрузки
            loadingIndicator.style.display = 'none';
        }
    };
    
    reader.readAsDataURL(file);
}
```

### Интеграция с мобильным приложением (Flutter)
```dart
import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

Future<void> colorizeImage(BuildContext context) async {
  final picker = ImagePicker();
  final pickedFile = await picker.getImage(source: ImageSource.gallery);
  
  if (pickedFile != null) {
    // Показываем индикатор загрузки
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (BuildContext context) {
        return Center(child: CircularProgressIndicator());
      },
    );
    
    try {
      // Читаем файл
      File imageFile = File(pickedFile.path);
      List<int> imageBytes = await imageFile.readAsBytes();
      String base64Image = base64Encode(imageBytes);
      
      // Подготавливаем данные запроса
      var requestData = {
        'image_data': base64Image,
        'style': 'vintage',
        'saturation': 1.2,
        'contrast': 1.0
      };
      
      // Отправляем запрос
      final response = await http.post(
        Uri.parse('http://api.example.com/colorize'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(requestData),
      );
      
      // Закрываем индикатор загрузки
      Navigator.pop(context);
      
      if (response.statusCode == 200) {
        var data = jsonDecode(response.body);
        
        // Показываем результат
        showDialog(
          context: context,
          builder: (BuildContext context) {
            return AlertDialog(
              title: Text('Результат колоризации'),
              content: Image.memory(base64Decode(
                data['colorized_image'].split(',')[1]
              )),
              actions: <Widget>[
                TextButton(
                  child: Text('Закрыть'),
                  onPressed: () => Navigator.pop(context),
                ),
                TextButton(
                  child: Text('Сохранить'),
                  onPressed: () {
                    // Код для сохранения изображения
                  },
                ),
              ],
            );
          },
        );
      } else {
        // Обработка ошибок
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Ошибка: ${response.statusCode}')),
        );
      }
    } catch (e) {
      // Закрываем индикатор загрузки
      Navigator.pop(context);
      
      // Показываем ошибку
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Ошибка: $e')),
      );
    }
  }
}
```

### Интеграция с бэкенд-сервисом (Node.js)
```javascript
const express = require('express');
const multer = require('multer');
const axios = require('axios');
const fs = require('fs');
const path = require('path');

const app = express();
const upload = multer({ dest: 'uploads/' });

// Эндпоинт для колоризации через TintoraAI API
app.post('/api/colorize', upload.single('image'), async (req, res) => {
    try {
        // Проверяем наличие файла
        if (!req.file) {
            return res.status(400).json({ error: 'Не выбрано изображение' });
        }
        
        // Читаем файл
        const imageBuffer = fs.readFileSync(req.file.path);
        const base64Image = imageBuffer.toString('base64');
        
        // Подготовка данных для запроса
        const requestData = {
            image_data: base64Image,
            style: req.body.style || 'default',
            saturation: parseFloat(req.body.saturation) || 1.0,
            contrast: parseFloat(req.body.contrast) || 1.0,
            show_uncertainty: req.body.show_uncertainty === 'true'
        };
        
        // Отправляем запрос к TintoraAI API
        const response = await axios.post('http://localhost:8000/colorize', requestData, {
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': process.env.TINTORAI_API_KEY || ''
            },
            timeout: 30000 // 30 секунд
        });
        
        // Удаляем временный файл
        fs.unlinkSync(req.file.path);
        
        // Отправляем результат
        res.json(response.data);
        
    } catch (error) {
        console.error('Ошибка при колоризации:', error);
        
        // Удаляем временный файл в случае ошибки
        if (req.file && fs.existsSync(req.file.path)) {
            fs.unlinkSync(req.file.path);
        }
        
        // Проверяем тип ошибки
        if (error.response) {
            // Ошибка от API TintoraAI
            res.status(error.response.status).json({
                error: error.response.data.detail || 'Ошибка от API TintoraAI'
            });
        } else {
            // Другие ошибки
            res.status(500).json({
                error: 'Внутренняя ошибка сервера',
                details: error.message
            });
        }
    }
});

// Запуск сервера
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Сервер запущен на порту ${PORT}`);
});
```

Если у вас есть вопросы по использованию API TintoraAI или вам нужны дополнительные примеры интеграции, обратитесь к команде разработчиков через GitHub Issues или по электронной почте.