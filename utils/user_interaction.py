"""
User Interaction Module: Модуль для взаимодействия с пользователем через консольные команды.

Данный модуль обеспечивает интерактивное взаимодействие с пользователем через систему
консольных команд формата "Команда:Текст". Это позволяет удобно управлять процессом 
колоризации, настраивать параметры, получать обратную связь и выполнять другие действия
без изменения исходного кода.

Ключевые особенности:
- Система команд в формате "Команда:Текст" для интуитивного управления
- Расширяемый интерфейс для добавления новых команд и функциональности
- Поддержка различных типов команд: параметры, действия, запросы, настройки и т.д.
- Интеграция с другими модулями системы для комплексного управления
- Возможность создания скриптов из последовательности команд

Преимущества для колоризации:
- Интерактивная настройка параметров колоризации в реальном времени
- Выбор различных стилей и палитр без перезапуска программы
- Возможность корректировки результатов на основе обратной связи
- Подготовка к интеграции с внешними интерфейсами (API, веб-интерфейс и т.д.)
"""

import os
import sys
import re
import json
import time
import threading
import logging
import inspect
from typing import Dict, List, Union, Optional, Any, Callable
from enum import Enum
import readline  # Для улучшенного ввода в консоли


class CommandType(Enum):
    """Типы команд для классификации и обработки."""
    PARAMETER = "parameter"  # Установка параметра
    ACTION = "action"        # Выполнение действия
    QUERY = "query"          # Запрос информации
    HELP = "help"            # Справка
    SYSTEM = "system"        # Системные команды
    CUSTOM = "custom"        # Пользовательские команды
    STYLE = "style"          # Команды настройки стиля
    METADATA = "metadata"    # Команды работы с метаданными
    BATCH = "batch"          # Команды пакетной обработки
    INTERACTIVE = "interactive"  # Интерактивные команды


class CommandResult:
    """
    Результат выполнения команды.
    
    Args:
        success (bool): Успешно ли выполнена команда
        message (str): Сообщение о результате выполнения
        data (Any, optional): Дополнительные данные результата
    """
    def __init__(self, success: bool, message: str, data: Any = None):
        self.success = success
        self.message = message
        self.data = data
        self.timestamp = time.time()
        
    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"{status} {self.message}"
    
    def to_dict(self) -> Dict:
        """Преобразует результат в словарь для JSON-сериализации."""
        return {
            "success": self.success,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp
        }


class Command:
    """
    Базовый класс для команды.
    
    Args:
        name (str): Имя команды
        description (str): Описание команды
        usage (str): Пример использования
        cmd_type (CommandType): Тип команды
    """
    def __init__(self, name: str, description: str, usage: str, cmd_type: CommandType):
        self.name = name
        self.description = description
        self.usage = usage
        self.cmd_type = cmd_type
        
    def execute(self, args: str, context: Dict = None) -> CommandResult:
        """
        Выполняет команду.
        
        Args:
            args (str): Аргументы команды
            context (Dict, optional): Контекст выполнения команды
            
        Returns:
            CommandResult: Результат выполнения команды
        """
        raise NotImplementedError("Метод execute должен быть реализован в подклассах")
    
    def get_help(self) -> str:
        """
        Возвращает справку по команде.
        
        Returns:
            str: Справка по команде
        """
        return f"{self.name}: {self.description}\nИспользование: {self.usage}"


class SetParameterCommand(Command):
    """
    Команда для установки значения параметра.
    
    Args:
        name (str): Имя команды
        description (str): Описание команды
        usage (str): Пример использования
        parameter_key (str): Ключ параметра в контексте
        validator (Callable, optional): Функция валидации значения
        converter (Callable, optional): Функция преобразования значения
    """
    def __init__(self, name: str, description: str, usage: str, parameter_key: str,
                validator: Callable = None, converter: Callable = None):
        super().__init__(name, description, usage, CommandType.PARAMETER)
        self.parameter_key = parameter_key
        self.validator = validator or (lambda x: True)
        self.converter = converter or (lambda x: x)
        
    def execute(self, args: str, context: Dict = None) -> CommandResult:
        """
        Устанавливает значение параметра.
        
        Args:
            args (str): Значение параметра
            context (Dict, optional): Контекст выполнения команды
            
        Returns:
            CommandResult: Результат выполнения команды
        """
        if context is None:
            return CommandResult(False, "Не указан контекст для сохранения параметра")
        
        try:
            # Преобразуем значение
            value = self.converter(args.strip())
            
            # Проверяем значение
            if not self.validator(value):
                return CommandResult(False, f"Некорректное значение для параметра {self.name}: {args}")
            
            # Сохраняем значение в контекст
            keys = self.parameter_key.split('.')
            current = context
            
            # Проходим по всем ключам, кроме последнего
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Устанавливаем значение для последнего ключа
            current[keys[-1]] = value
            
            return CommandResult(True, f"Установлен параметр {self.name} = {value}", value)
        except Exception as e:
            return CommandResult(False, f"Ошибка при установке параметра {self.name}: {e}")


class ActionCommand(Command):
    """
    Команда для выполнения действия.
    
    Args:
        name (str): Имя команды
        description (str): Описание команды
        usage (str): Пример использования
        action_func (Callable): Функция действия
    """
    def __init__(self, name: str, description: str, usage: str, action_func: Callable):
        super().__init__(name, description, usage, CommandType.ACTION)
        self.action_func = action_func
        
    def execute(self, args: str, context: Dict = None) -> CommandResult:
        """
        Выполняет действие.
        
        Args:
            args (str): Аргументы действия
            context (Dict, optional): Контекст выполнения команды
            
        Returns:
            CommandResult: Результат выполнения действия
        """
        try:
            # Получаем информацию о параметрах функции
            sig = inspect.signature(self.action_func)
            
            # Проверяем, нужен ли контекст
            needs_context = 'context' in sig.parameters
            needs_args = 'args' in sig.parameters
            
            # Формируем аргументы вызова
            call_args = {}
            if needs_context:
                call_args['context'] = context
            if needs_args:
                call_args['args'] = args
                
            # Вызываем функцию
            result = self.action_func(**call_args)
            
            # Если функция вернула CommandResult, используем его
            if isinstance(result, CommandResult):
                return result
            
            # Иначе создаем новый CommandResult
            return CommandResult(True, f"Действие {self.name} выполнено успешно", result)
        except Exception as e:
            return CommandResult(False, f"Ошибка при выполнении действия {self.name}: {e}")


class QueryCommand(Command):
    """
    Команда для запроса информации.
    
    Args:
        name (str): Имя команды
        description (str): Описание команды
        usage (str): Пример использования
        query_func (Callable): Функция запроса
    """
    def __init__(self, name: str, description: str, usage: str, query_func: Callable):
        super().__init__(name, description, usage, CommandType.QUERY)
        self.query_func = query_func
        
    def execute(self, args: str, context: Dict = None) -> CommandResult:
        """
        Выполняет запрос информации.
        
        Args:
            args (str): Аргументы запроса
            context (Dict, optional): Контекст выполнения команды
            
        Returns:
            CommandResult: Результат запроса
        """
        try:
            # Получаем информацию о параметрах функции
            sig = inspect.signature(self.query_func)
            
            # Проверяем, нужен ли контекст
            needs_context = 'context' in sig.parameters
            needs_args = 'args' in sig.parameters
            
            # Формируем аргументы вызова
            call_args = {}
            if needs_context:
                call_args['context'] = context
            if needs_args:
                call_args['args'] = args
                
            # Вызываем функцию
            result = self.query_func(**call_args)
            
            # Если функция вернула CommandResult, используем его
            if isinstance(result, CommandResult):
                return result
            
            # Иначе создаем новый CommandResult
            return CommandResult(True, f"Запрос {self.name} выполнен успешно", result)
        except Exception as e:
            return CommandResult(False, f"Ошибка при выполнении запроса {self.name}: {e}")


class HelpCommand(Command):
    """
    Команда для получения справки.
    
    Args:
        name (str): Имя команды
        description (str): Описание команды
        usage (str): Пример использования
        command_registry (Dict): Реестр команд
    """
    def __init__(self, name: str, description: str, usage: str, command_registry: Dict):
        super().__init__(name, description, usage, CommandType.HELP)
        self.command_registry = command_registry
        
    def execute(self, args: str, context: Dict = None) -> CommandResult:
        """
        Возвращает справку по команде или по всем командам.
        
        Args:
            args (str): Имя команды для получения справки или пусто для справки по всем командам
            context (Dict, optional): Контекст выполнения команды
            
        Returns:
            CommandResult: Справка
        """
        args = args.strip()
        
        if args:
            # Справка по конкретной команде
            if args in self.command_registry:
                command = self.command_registry[args]
                return CommandResult(True, command.get_help())
            else:
                return CommandResult(False, f"Команда '{args}' не найдена")
        else:
            # Справка по всем командам
            help_text = "Доступные команды:\n"
            
            # Группируем команды по типам
            commands_by_type = {}
            for name, command in sorted(self.command_registry.items()):
                cmd_type = command.cmd_type
                if cmd_type not in commands_by_type:
                    commands_by_type[cmd_type] = []
                commands_by_type[cmd_type].append((name, command.description))
            
            # Выводим команды по группам
            for cmd_type in sorted(commands_by_type.keys(), key=lambda x: x.value):
                help_text += f"\n{cmd_type.value.upper()}:\n"
                for name, description in sorted(commands_by_type[cmd_type]):
                    help_text += f"  {name}: {description}\n"
            
            help_text += "\nДля получения подробной справки по команде используйте: help:команда"
            return CommandResult(True, help_text)


class StyleCommand(Command):
    """
    Команда для настройки стиля колоризации.
    
    Args:
        name (str): Имя команды
        description (str): Описание команды
        usage (str): Пример использования
        style_key (str): Ключ стиля
        style_values (List): Допустимые значения стиля
    """
    def __init__(self, name: str, description: str, usage: str, style_key: str, style_values: List = None):
        super().__init__(name, description, usage, CommandType.STYLE)
        self.style_key = style_key
        self.style_values = style_values or []
        
    def execute(self, args: str, context: Dict = None) -> CommandResult:
        """
        Устанавливает стиль колоризации.
        
        Args:
            args (str): Значение стиля
            context (Dict, optional): Контекст выполнения команды
            
        Returns:
            CommandResult: Результат установки стиля
        """
        if context is None:
            return CommandResult(False, "Не указан контекст для сохранения стиля")
        
        args = args.strip().lower()
        
        # Проверяем, что значение допустимо
        if self.style_values and args not in [v.lower() for v in self.style_values]:
            values_str = ", ".join(self.style_values)
            return CommandResult(False, f"Недопустимое значение стиля '{args}'. Допустимые значения: {values_str}")
        
        # Создаем секцию стилей, если её нет
        if 'style' not in context:
            context['style'] = {}
            
        # Устанавливаем стиль
        context['style'][self.style_key] = args
        
        return CommandResult(True, f"Установлен стиль {self.style_key} = {args}", args)


class CommandRegistry:
    """
    Реестр команд для регистрации и получения команд.
    """
    def __init__(self):
        self.commands = {}
        
    def register_command(self, command: Command):
        """
        Регистрирует команду.
        
        Args:
            command (Command): Команда
        """
        self.commands[command.name] = command
        
    def get_command(self, name: str) -> Optional[Command]:
        """
        Возвращает команду по имени.
        
        Args:
            name (str): Имя команды
            
        Returns:
            Optional[Command]: Команда или None, если команда не найдена
        """
        return self.commands.get(name)
    
    def list_commands(self) -> List[str]:
        """
        Возвращает список имен всех зарегистрированных команд.
        
        Returns:
            List[str]: Список имен команд
        """
        return list(self.commands.keys())
    
    def register_parameter(self, name: str, description: str, usage: str, parameter_key: str,
                          validator: Callable = None, converter: Callable = None):
        """
        Регистрирует команду для установки параметра.
        
        Args:
            name (str): Имя команды
            description (str): Описание команды
            usage (str): Пример использования
            parameter_key (str): Ключ параметра в контексте
            validator (Callable, optional): Функция валидации значения
            converter (Callable, optional): Функция преобразования значения
        """
        command = SetParameterCommand(name, description, usage, parameter_key, validator, converter)
        self.register_command(command)
        
    def register_action(self, name: str, description: str, usage: str, action_func: Callable):
        """
        Регистрирует команду для выполнения действия.
        
        Args:
            name (str): Имя команды
            description (str): Описание команды
            usage (str): Пример использования
            action_func (Callable): Функция действия
        """
        command = ActionCommand(name, description, usage, action_func)
        self.register_command(command)
        
    def register_query(self, name: str, description: str, usage: str, query_func: Callable):
        """
        Регистрирует команду для запроса информации.
        
        Args:
            name (str): Имя команды
            description (str): Описание команды
            usage (str): Пример использования
            query_func (Callable): Функция запроса
        """
        command = QueryCommand(name, description, usage, query_func)
        self.register_command(command)
        
    def register_style(self, name: str, description: str, usage: str, style_key: str, style_values: List = None):
        """
        Регистрирует команду для настройки стиля.
        
        Args:
            name (str): Имя команды
            description (str): Описание команды
            usage (str): Пример использования
            style_key (str): Ключ стиля
            style_values (List, optional): Допустимые значения стиля
        """
        command = StyleCommand(name, description, usage, style_key, style_values)
        self.register_command(command)


class CommandProcessor:
    """
    Процессор команд для парсинга и выполнения команд.
    
    Args:
        registry (CommandRegistry): Реестр команд
        context (Dict, optional): Начальный контекст
    """
    def __init__(self, registry: CommandRegistry, context: Dict = None):
        self.registry = registry
        self.context = context or {}
        self.command_history = []
        
        # Регистрируем базовую команду help
        help_command = HelpCommand("help", "Показать справку по доступным командам", "help:[команда]", registry.commands)
        registry.register_command(help_command)
        
        # Настраиваем логирование
        self.logger = logging.getLogger("CommandProcessor")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def parse_command(self, command_str: str) -> Tuple[Optional[str], str]:
        """
        Парсит строку команды.
        
        Args:
            command_str (str): Строка команды
            
        Returns:
            Tuple[Optional[str], str]: Кортеж (имя_команды, аргументы) или (None, "") если ошибка парсинга
        """
        if not command_str or ":" not in command_str:
            return None, ""
            
        # Разбиваем на имя команды и аргументы
        parts = command_str.split(":", 1)
        command_name = parts[0].strip().lower()
        args = parts[1] if len(parts) > 1 else ""
        
        return command_name, args
        
    def process_command(self, command_str: str) -> CommandResult:
        """
        Обрабатывает команду.
        
        Args:
            command_str (str): Строка команды
            
        Returns:
            CommandResult: Результат выполнения команды
        """
        # Парсим команду
        command_name, args = self.parse_command(command_str)
        
        if command_name is None:
            return CommandResult(False, f"Неправильный формат команды: {command_str}. Используйте формат 'Команда:Аргументы'")
        
        # Получаем команду из реестра
        command = self.registry.get_command(command_name)
        
        if command is None:
            return CommandResult(False, f"Неизвестная команда: {command_name}")
        
        # Выполняем команду
        result = command.execute(args, self.context)
        
        # Добавляем в историю
        self.command_history.append((command_str, result))
        
        # Логируем результат
        if result.success:
            self.logger.info(f"Команда '{command_name}' выполнена успешно")
        else:
            self.logger.warning(f"Ошибка выполнения команды '{command_name}': {result.message}")
        
        return result
    
    def get_command_history(self) -> List[Tuple[str, CommandResult]]:
        """
        Возвращает историю выполнения команд.
        
        Returns:
            List[Tuple[str, CommandResult]]: История команд
        """
        return self.command_history
    
    def clear_history(self):
        """Очищает историю команд."""
        self.command_history = []


class InteractiveConsole:
    """
    Интерактивная консоль для ввода команд.
    
    Args:
        processor (CommandProcessor): Процессор команд
        prompt (str, optional): Строка приглашения к вводу
    """
    def __init__(self, processor: CommandProcessor, prompt: str = "TintoraAI> "):
        self.processor = processor
        self.prompt = prompt
        self.running = False
        
    def start(self):
        """Запускает интерактивную консоль."""
        self.running = True
        print("TintoraAI Interactive Console")
        print("Введите 'help:' для получения списка команд или 'exit:' для выхода")
        
        while self.running:
            try:
                # Получаем команду от пользователя
                command_str = input(self.prompt)
                
                # Проверяем команду выхода
                if command_str.lower() in ("exit:", "quit:", "q:"):
                    self.running = False
                    print("Выход из интерактивного режима")
                    break
                
                # Обрабатываем команду
                result = self.processor.process_command(command_str)
                
                # Выводим результат
                print(result)
                
                # Если есть дополнительные данные, выводим их
                if result.data is not None and isinstance(result.data, (dict, list)):
                    print(json.dumps(result.data, indent=2, ensure_ascii=False))
                
            except KeyboardInterrupt:
                print("\nПрервано пользователем. Для выхода введите 'exit:'")
            except Exception as e:
                print(f"Ошибка: {e}")
    
    def stop(self):
        """Останавливает интерактивную консоль."""
        self.running = False


class CommandScript:
    """
    Скрипт команд для пакетного выполнения команд.
    
    Args:
        processor (CommandProcessor): Процессор команд
    """
    def __init__(self, processor: CommandProcessor):
        self.processor = processor
        self.commands = []
        
    def load_from_file(self, filepath: str) -> bool:
        """
        Загружает команды из файла.
        
        Args:
            filepath (str): Путь к файлу
            
        Returns:
            bool: True, если загрузка успешна, иначе False
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        self.commands.append(line)
            return True
        except Exception as e:
            print(f"Ошибка загрузки скрипта: {e}")
            return False
            
    def add_command(self, command_str: str):
        """
        Добавляет команду в скрипт.
        
        Args:
            command_str (str): Строка команды
        """
        self.commands.append(command_str)
        
    def execute(self, stop_on_error: bool = False) -> List[CommandResult]:
        """
        Выполняет все команды в скрипте.
        
        Args:
            stop_on_error (bool): Остановка при ошибке
            
        Returns:
            List[CommandResult]: Результаты выполнения команд
        """
        results = []
        
        for command_str in self.commands:
            result = self.processor.process_command(command_str)
            results.append(result)
            
            if not result.success and stop_on_error:
                break
                
        return results
    
    def save_to_file(self, filepath: str) -> bool:
        """
        Сохраняет команды в файл.
        
        Args:
            filepath (str): Путь к файлу
            
        Returns:
            bool: True, если сохранение успешно, иначе False
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("# TintoraAI Command Script\n")
                f.write(f"# Created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for command in self.commands:
                    f.write(f"{command}\n")
                    
            return True
        except Exception as e:
            print(f"Ошибка сохранения скрипта: {e}")
            return False


class UserInteractionModule:
    """
    Основной модуль взаимодействия с пользователем, объединяющий все компоненты.
    
    Args:
        initial_context (Dict, optional): Начальный контекст
        auto_register_commands (bool): Автоматическая регистрация стандартных команд
    """
    def __init__(self, initial_context: Dict = None, auto_register_commands: bool = True):
        self.registry = CommandRegistry()
        self.context = initial_context or {}
        self.processor = CommandProcessor(self.registry, self.context)
        self.console = InteractiveConsole(self.processor)
        
        # Автоматическая регистрация стандартных команд
        if auto_register_commands:
            self._register_standard_commands()
        
    def _register_standard_commands(self):
        """Регистрирует стандартные команды."""
        # Системные команды
        self.registry.register_action(
            "status", 
            "Показать текущий статус системы", 
            "status:", 
            lambda context: CommandResult(True, "Система активна", {"context_size": len(context)})
        )
        
        self.registry.register_action(
            "clear", 
            "Очистить консоль", 
            "clear:", 
            lambda: os.system('cls' if os.name == 'nt' else 'clear') or CommandResult(True, "Консоль очищена")
        )
        
        self.registry.register_action(
            "save_context", 
            "Сохранить текущий контекст в файл", 
            "save_context:path/to/file.json", 
            self._save_context
        )
        
        self.registry.register_action(
            "load_context", 
            "Загрузить контекст из файла", 
            "load_context:path/to/file.json", 
            self._load_context
        )
        
        # Параметры колоризации
        self.registry.register_parameter(
            "alpha", 
            "Установить интенсивность колоризации (0.0-1.0)", 
            "alpha:0.8", 
            "colorization.alpha",
            lambda x: 0.0 <= x <= 1.0,
            float
        )
        
        self.registry.register_parameter(
            "temperature", 
            "Установить температуру для вариативности цветов (0.0-1.0)", 
            "temperature:0.5", 
            "colorization.temperature",
            lambda x: 0.0 <= x <= 1.0,
            float
        )
        
        # Стили колоризации
        self.registry.register_style(
            "era", 
            "Установить историческую эпоху для стиля колоризации", 
            "era:1950s", 
            "era", 
            ["1920s", "1950s", "1970s", "1980s", "modern"]
        )
        
        self.registry.register_style(
            "artistic_style", 
            "Установить художественный стиль колоризации", 
            "artistic_style:impressionism", 
            "artistic", 
            ["natural", "impressionism", "expressionism", "pop-art", "noir", "vintage"]
        )
        
        # Действия для обработки изображений
        self.registry.register_action(
            "colorize", 
            "Колоризовать изображение", 
            "colorize:path/to/image.jpg", 
            self._colorize_image
        )
        
        self.registry.register_action(
            "batch_colorize", 
            "Колоризовать все изображения в директории", 
            "batch_colorize:path/to/directory", 
            self._batch_colorize
        )
        
        self.registry.register_action(
            "compare", 
            "Создать сравнение оригинального и колоризованного изображения", 
            "compare:path/to/image.jpg", 
            self._create_comparison
        )
        
        # Запросы
        self.registry.register_query(
            "list_styles", 
            "Показать список доступных стилей", 
            "list_styles:", 
            self._list_styles
        )
        
        self.registry.register_query(
            "show_parameters", 
            "Показать текущие параметры колоризации", 
            "show_parameters:", 
            self._show_parameters
        )
        
    def _save_context(self, args: str, context: Dict) -> CommandResult:
        """
        Сохраняет контекст в файл.
        
        Args:
            args (str): Путь к файлу
            context (Dict): Контекст
            
        Returns:
            CommandResult: Результат сохранения
        """
        filepath = args.strip()
        if not filepath:
            return CommandResult(False, "Не указан путь для сохранения контекста")
            
        try:
            # Создаем директорию, если не существует
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Сохраняем контекст
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(context, f, indent=2, ensure_ascii=False)
                
            return CommandResult(True, f"Контекст сохранен в {filepath}")
        except Exception as e:
            return CommandResult(False, f"Ошибка при сохранении контекста: {e}")
            
    def _load_context(self, args: str, context: Dict) -> CommandResult:
        """
        Загружает контекст из файла.
        
        Args:
            args (str): Путь к файлу
            context (Dict): Текущий контекст
            
        Returns:
            CommandResult: Результат загрузки
        """
        filepath = args.strip()
        if not filepath:
            return CommandResult(False, "Не указан путь для загрузки контекста")
            
        try:
            # Проверяем существование файла
            if not os.path.isfile(filepath):
                return CommandResult(False, f"Файл {filepath} не существует")
                
            # Загружаем контекст
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_context = json.load(f)
                
            # Обновляем текущий контекст
            context.update(loaded_context)
                
            return CommandResult(True, f"Контекст загружен из {filepath}")
        except Exception as e:
            return CommandResult(False, f"Ошибка при загрузке контекста: {e}")
            
    def _colorize_image(self, args: str, context: Dict) -> CommandResult:
        """
        Колоризует изображение (заглушка).
        
        Args:
            args (str): Путь к изображению
            context (Dict): Контекст
            
        Returns:
            CommandResult: Результат колоризации
        """
        filepath = args.strip()
        if not filepath:
            return CommandResult(False, "Не указан путь к изображению")
            
        # Проверяем существование файла
        if not os.path.isfile(filepath):
            return CommandResult(False, f"Файл {filepath} не существует")
            
        # Здесь должен быть код колоризации изображения
        # В текущей реализации это заглушка
        
        return CommandResult(True, f"Изображение {filepath} колоризовано", {
            "input_path": filepath,
            "output_path": filepath.replace('.jpg', '_colorized.jpg'),
            "parameters": context.get('colorization', {})
        })
            
    def _batch_colorize(self, args: str, context: Dict) -> CommandResult:
        """
        Колоризует все изображения в директории (заглушка).
        
        Args:
            args (str): Путь к директории
            context (Dict): Контекст
            
        Returns:
            CommandResult: Результат колоризации
        """
        directory = args.strip()
        if not directory:
            return CommandResult(False, "Не указан путь к директории")
            
        # Проверяем существование директории
        if not os.path.isdir(directory):
            return CommandResult(False, f"Директория {directory} не существует")
            
        # Здесь должен быть код пакетной колоризации изображений
        # В текущей реализации это заглушка
        
        return CommandResult(True, f"Изображения в директории {directory} колоризованы", {
            "input_directory": directory,
            "output_directory": os.path.join(directory, "colorized"),
            "processed_images": 0,
            "parameters": context.get('colorization', {})
        })
            
    def _create_comparison(self, args: str, context: Dict) -> CommandResult:
        """
        Создает сравнение оригинального и колоризованного изображения (заглушка).
        
        Args:
            args (str): Путь к изображению
            context (Dict): Контекст
            
        Returns:
            CommandResult: Результат создания сравнения
        """
        filepath = args.strip()
        if not filepath:
            return CommandResult(False, "Не указан путь к изображению")
            
        # Проверяем существование файла
        if not os.path.isfile(filepath):
            return CommandResult(False, f"Файл {filepath} не существует")
            
        # Здесь должен быть код создания сравнения
        # В текущей реализации это заглушка
        
        return CommandResult(True, f"Создано сравнение для изображения {filepath}", {
            "input_path": filepath,
            "comparison_path": filepath.replace('.jpg', '_comparison.jpg')
        })
            
    def _list_styles(self, context: Dict) -> CommandResult:
        """
        Возвращает список доступных стилей.
        
        Args:
            context (Dict): Контекст
            
        Returns:
            CommandResult: Список стилей
        """
        styles = {
            "eras": ["1920s", "1950s", "1970s", "1980s", "modern"],
            "artistic": ["natural", "impressionism", "expressionism", "pop-art", "noir", "vintage"]
        }
        
        return CommandResult(True, "Список доступных стилей", styles)
            
    def _show_parameters(self, context: Dict) -> CommandResult:
        """
        Возвращает текущие параметры колоризации.
        
        Args:
            context (Dict): Контекст
            
        Returns:
            CommandResult: Параметры колоризации
        """
        parameters = context.get('colorization', {})
        style = context.get('style', {})
        
        return CommandResult(True, "Текущие параметры колоризации", {
            "colorization": parameters,
            "style": style
        })
        
    def register_command(self, command: Command):
        """
        Регистрирует команду.
        
        Args:
            command (Command): Команда
        """
        self.registry.register_command(command)
        
    def register_parameter(self, name: str, description: str, usage: str, parameter_key: str,
                          validator: Callable = None, converter: Callable = None):
        """
        Регистрирует команду для установки параметра.
        
        Args:
            name (str): Имя команды
            description (str): Описание команды
            usage (str): Пример использования
            parameter_key (str): Ключ параметра в контексте
            validator (Callable, optional): Функция валидации значения
            converter (Callable, optional): Функция преобразования значения
        """
        self.registry.register_parameter(name, description, usage, parameter_key, validator, converter)
        
    def register_action(self, name: str, description: str, usage: str, action_func: Callable):
        """
        Регистрирует команду для выполнения действия.
        
        Args:
            name (str): Имя команды
            description (str): Описание команды
            usage (str): Пример использования
            action_func (Callable): Функция действия
        """
        self.registry.register_action(name, description, usage, action_func)
        
    def register_query(self, name: str, description: str, usage: str, query_func: Callable):
        """
        Регистрирует команду для запроса информации.
        
        Args:
            name (str): Имя команды
            description (str): Описание команды
            usage (str): Пример использования
            query_func (Callable): Функция запроса
        """
        self.registry.register_query(name, description, usage, query_func)
        
    def register_style(self, name: str, description: str, usage: str, style_key: str, style_values: List = None):
        """
        Регистрирует команду для настройки стиля.
        
        Args:
            name (str): Имя команды
            description (str): Описание команды
            usage (str): Пример использования
            style_key (str): Ключ стиля
            style_values (List, optional): Допустимые значения стиля
        """
        self.registry.register_style(name, description, usage, style_key, style_values)
        
    def process_command(self, command_str: str) -> CommandResult:
        """
        Обрабатывает команду.
        
        Args:
            command_str (str): Строка команды
            
        Returns:
            CommandResult: Результат выполнения команды
        """
        return self.processor.process_command(command_str)
        
    def start_interactive(self):
        """Запускает интерактивную консоль."""
        self.console.start()
        
    def create_script(self) -> CommandScript:
        """
        Создает новый скрипт команд.
        
        Returns:
            CommandScript: Новый скрипт команд
        """
        return CommandScript(self.processor)
        
    def load_script(self, filepath: str) -> Optional[CommandScript]:
        """
        Загружает скрипт команд из файла.
        
        Args:
            filepath (str): Путь к файлу
            
        Returns:
            Optional[CommandScript]: Загруженный скрипт или None, если ошибка
        """
        script = CommandScript(self.processor)
        if script.load_from_file(filepath):
            return script
        return None
        
    def get_context(self) -> Dict:
        """
        Возвращает текущий контекст.
        
        Returns:
            Dict: Контекст
        """
        return self.context


# Функция для создания модуля взаимодействия с пользователем
def create_user_interaction_module(initial_context: Dict = None, auto_register_commands: bool = True) -> UserInteractionModule:
    """
    Создает модуль взаимодействия с пользователем.
    
    Args:
        initial_context (Dict, optional): Начальный контекст
        auto_register_commands (bool): Автоматическая регистрация стандартных команд
        
    Returns:
        UserInteractionModule: Модуль взаимодействия с пользователем
    """
    return UserInteractionModule(initial_context, auto_register_commands)


if __name__ == "__main__":
    # Пример использования модуля взаимодействия с пользователем
    
    # Создаем модуль
    module = create_user_interaction_module()
    
    # Запускаем интерактивную консоль
    module.start_interactive()