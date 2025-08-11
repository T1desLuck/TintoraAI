# Инструкции по запуску тестов TintoraAI

## Обзор

Данный документ содержит инструкции по запуску тестов для проверки исправлений сигнатур классов TintoraAI, выполненных для прохождения CI/CD тестов.

## Исправленные компоненты

### Core модули
- **SwinUNet** (`core/swin_unet.py`) - добавлена поддержка `in_channels`, `out_channels`
- **ViTSemantic** (`core/vit_semantic.py`) - добавлена поддержка `in_channels`
- **FPNPyramid** (`core/fpn_pyramid.py`) - добавлена поддержка `in_channels`, `out_channels`

### Модули
- **GuideNet** (`modules/guide_net.py`) - добавлена поддержка `in_channels`, `advice_channels`, `device`
- **MotivationalDiscriminator** (`modules/discriminator.py`) - добавлена поддержка `in_channels`, `device`
- **StyleTransferModule** (`modules/style_transfer.py`) - добавлена поддержка `in_channels`, `out_channels`, `device`
- **MemoryBankModule** (`modules/memory_bank.py`) - добавлена поддержка `device`
- **AdaptableColorizer** (`modules/few_shot_adapter.py`) - добавлена поддержка `adapter_type`, `device`

### Функции потерь
- **PatchNCELoss** (`losses/patch_nce.py`) - уже поддерживает альтернативные параметры
- **VGGPerceptualLoss** (`losses/vgg_perceptual.py`) - уже поддерживает альтернативные параметры
- **GANLoss** (`losses/gan_loss.py`) - уже поддерживает альтернативные параметры, включая `device`
- **DynamicLossBalancer** (`losses/dynamic_balancer.py`) - уже поддерживает альтернативные параметры

## Запуск тестов

### 1. Быстрая валидация исправлений

```bash
python validate_fixes.py
```

Этот скрипт проверяет, что все исправленные классы корректно инициализируются с альтернативными параметрами.

### 2. Полный набор CI/CD тестов

```bash
python run_ci_tests.py
```

Этот скрипт запускает:
- Проверку импортов всех модулей
- Валидацию исправлений сигнатур
- Unittest тесты
- Pytest тесты (если доступен)

### 3. Отдельные тесты

```bash
# Тесты core компонентов
python -m pytest tests/test_core.py -v

# Тесты модулей
python -m pytest tests/test_modules.py -v

# Тесты функций потерь
python -m pytest tests/test_losses.py -v

# Все тесты
python -m pytest tests/ -v
```

### 4. Unittest тесты

```bash
# Запуск отдельного теста
python tests/test_core.py

# Запуск с unittest модулем
python -m unittest tests.test_core -v
```

## Ожидаемые результаты

После исправлений все тесты должны проходить успешно:

- ✅ Все классы инициализируются с альтернативными параметрами
- ✅ Сохранена обратная совместимость с оригинальными параметрами
- ✅ Архитектура и логика не нарушены
- ✅ Все импорты работают корректно
- ✅ CI/CD тесты проходят без ошибок

## Структура исправлений

Все исправления выполнены по единому паттерну:

```python
def __init__(self, original_param1, original_param2, 
             # Альтернативные параметры для совместимости с тестами
             alternative_param1=None, alternative_param2=None, **kwargs):
    super().__init__()
    
    # Обработка альтернативных параметров
    if alternative_param1 is not None:
        original_param1 = alternative_param1
    if alternative_param2 is not None:
        original_param2 = alternative_param2
    
    # Оригинальная логика инициализации
    self.param1 = original_param1
    self.param2 = original_param2
```

## Устранение проблем

Если тесты не проходят:

1. **Проверьте Python окружение**: убедитесь, что Python 3.7+ установлен
2. **Установите зависимости**: `pip install -r requirements.txt`
3. **Проверьте PYTHONPATH**: убедитесь, что корневая директория проекта в пути
4. **Запустите валидацию**: `python validate_fixes.py` для быстрой диагностики

## Контакты

При возникновении проблем с тестами обращайтесь к документации проекта или создавайте issue в репозитории.
