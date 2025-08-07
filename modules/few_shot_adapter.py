"""
Few-shot Adapter: Адаптация между доменами и датасетами с малым количеством примеров.

Данный модуль обеспечивает быструю адаптацию модели колоризации между различными доменами
и датасетами с использованием всего нескольких примеров (few-shot learning). Это особенно
полезно для колоризации специфических типов изображений, редких объектов, исторических
фотографий или адаптации к художественным стилям.

Ключевые особенности:
- Быстрая адаптация к новым доменам с малым количеством примеров
- Мета-обучение для эффективного переноса знаний
- Модульные адаптеры для тонкой настройки без изменения основной модели
- Методы доменной адаптации для согласования распределений признаков
- Механизмы прототипного обучения для улавливания сущностных характеристик домена

Преимущества для колоризации:
- Возможность специализации под конкретные жанры фотографии или изображений
- Сохранение общих знаний о колоризации при адаптации к новому домену
- Улучшение качества колоризации для нетипичных или специфических сценариев
- Эффективное использование ограниченных обучающих данных
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
from typing import Dict, List, Tuple, Union, Optional
from collections import OrderedDict


class FeatureAdapter(nn.Module):
    """
    Адаптер признаков для доменной адаптации.
    
    Модифицирует признаки модели для лучшей работы в целевом домене
    без полной переподгонки базовой модели.
    
    Args:
        in_features (int): Количество входных признаков
        bottleneck_dim (int): Размерность "бутылочного горлышка" для эффективной адаптации
        dropout_rate (float): Вероятность dropout
        init_scale (float): Масштаб инициализации для скейлинговых параметров
    """
    def __init__(self, in_features, bottleneck_dim=64, dropout_rate=0.0, init_scale=1e-3):
        super(FeatureAdapter, self).__init__()
        
        self.in_features = in_features
        self.bottleneck_dim = bottleneck_dim
        
        # Слои адаптера
        self.down_proj = nn.Linear(in_features, bottleneck_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_dim, in_features)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Параметры масштабирования для остаточного соединения
        self.scale = nn.Parameter(torch.ones(1) * init_scale)
        
        # Инициализация с небольшими значениями для минимального начального влияния
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=init_scale)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=init_scale)
        nn.init.zeros_(self.up_proj.bias)
        
    def forward(self, x):
        """
        Применяет адаптацию к признакам.
        
        Args:
            x (torch.Tensor): Входные признаки [B, ..., C]
            
        Returns:
            torch.Tensor: Адаптированные признаки [B, ..., C]
        """
        # Сохраняем оригинальную форму
        original_shape = x.shape
        
        # Преобразуем в 2D для операций линейных слоев
        if len(original_shape) > 2:
            x_flat = x.view(-1, self.in_features)
        else:
            x_flat = x
            
        # Прямое распространение через адаптер
        h = self.down_proj(x_flat)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.up_proj(h)
        
        # Применяем масштабирующий параметр
        h = h * self.scale
        
        # Остаточное соединение
        output = x_flat + h
        
        # Восстанавливаем исходную форму
        if len(original_shape) > 2:
            output = output.view(original_shape)
            
        return output


class AdapterBlock(nn.Module):
    """
    Блок адаптера для свёрточных признаков.
    
    Args:
        in_channels (int): Количество входных каналов
        bottleneck_dim (int): Размерность "бутылочного горлышка" для эффективной адаптации
        kernel_size (int): Размер ядра свёртки
        dropout_rate (float): Вероятность dropout
    """
    def __init__(self, in_channels, bottleneck_dim=None, kernel_size=1, dropout_rate=0.0):
        super(AdapterBlock, self).__init__()
        
        if bottleneck_dim is None:
            bottleneck_dim = max(1, in_channels // 4)
            
        self.adapter = nn.Sequential(
            # Сжатие каналов
            nn.Conv2d(in_channels, bottleneck_dim, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            # Восстановление каналов
            nn.Conv2d(bottleneck_dim, in_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        
        # Параметры масштабирования для остаточного соединения
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        
        # Инициализация с небольшими значениями
        self._init_weights()
        
    def _init_weights(self):
        """
        Инициализирует веса для слоев адаптера.
        """
        for m in self.adapter.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        """
        Применяет адаптацию к свёрточным признакам.
        
        Args:
            x (torch.Tensor): Входные свёрточные признаки [B, C, H, W]
            
        Returns:
            torch.Tensor: Адаптированные признаки [B, C, H, W]
        """
        # Адаптируем признаки и применяем масштабирование
        adapted = self.adapter(x) * self.scale
        
        # Остаточное соединение
        return x + adapted


class DomainAdaptationLayer(nn.Module):
    """
    Слой для адаптации признаков между доменами.
    
    Args:
        in_channels (int): Количество входных каналов
        domain_code_dim (int): Размерность кода домена
        use_instance_norm (bool): Использовать ли Instance Normalization
        use_film (bool): Использовать ли Feature-wise Linear Modulation (FiLM)
    """
    def __init__(self, in_channels, domain_code_dim=16, use_instance_norm=True, use_film=True):
        super(DomainAdaptationLayer, self).__init__()
        
        self.in_channels = in_channels
        self.use_instance_norm = use_instance_norm
        self.use_film = use_film
        
        # Instance Normalization для нормализации статистик домена
        if use_instance_norm:
            self.instance_norm = nn.InstanceNorm2d(in_channels, affine=False)
            
        # FiLM для условной модуляции признаков
        if use_film:
            self.film_generator = nn.Sequential(
                nn.Linear(domain_code_dim, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, in_channels * 2)  # gamma и beta для каждого канала
            )
            
    def forward(self, x, domain_code=None):
        """
        Применяет доменную адаптацию к признакам.
        
        Args:
            x (torch.Tensor): Входные признаки [B, C, H, W]
            domain_code (torch.Tensor, optional): Код домена [B, domain_code_dim]
            
        Returns:
            torch.Tensor: Адаптированные признаки [B, C, H, W]
        """
        # Применяем Instance Normalization
        if self.use_instance_norm:
            x = self.instance_norm(x)
            
        # Применяем FiLM модуляцию
        if self.use_film and domain_code is not None:
            # Генерируем параметры модуляции
            film_params = self.film_generator(domain_code)
            
            # Разделяем на gamma и beta
            gamma, beta = film_params.chunk(2, dim=1)
            
            # Изменяем форму для применения к признакам
            gamma = gamma.view(-1, self.in_channels, 1, 1)
            beta = beta.view(-1, self.in_channels, 1, 1)
            
            # Применяем модуляцию
            x = gamma * x + beta
            
        return x


class PrototypicalNetwork(nn.Module):
    """
    Прототипическая сеть для few-shot обучения.
    
    Args:
        embed_dim (int): Размерность пространства признаков
        prototype_dim (int): Размерность прототипов
        num_prototypes (int): Количество прототипов
        temperature (float): Температурный параметр для мягкой классификации
    """
    def __init__(self, embed_dim=512, prototype_dim=128, num_prototypes=16, temperature=0.5):
        super(PrototypicalNetwork, self).__init__()
        
        self.embed_dim = embed_dim
        self.prototype_dim = prototype_dim
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        
        # Проекция признаков в пространство прототипов
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, prototype_dim)
        )
        
        # Инициализация прототипов
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, prototype_dim))
        
        # Нормализуем прототипы
        with torch.no_grad():
            self.prototypes.copy_(F.normalize(self.prototypes, dim=1))
            
    def forward(self, features):
        """
        Вычисляет сходство с прототипами.
        
        Args:
            features (torch.Tensor): Входные признаки [B, embed_dim]
            
        Returns:
            dict: {
                'logits': torch.Tensor,  # Логиты сходства с прототипами [B, num_prototypes]
                'projected_features': torch.Tensor,  # Спроецированные признаки [B, prototype_dim]
                'prototypes': torch.Tensor  # Нормализованные прототипы [num_prototypes, prototype_dim]
            }
        """
        # Проецируем признаки
        projected = self.projection(features)
        
        # Нормализуем признаки
        projected_norm = F.normalize(projected, dim=1)
        
        # Нормализуем прототипы
        prototypes_norm = F.normalize(self.prototypes, dim=1)
        
        # Вычисляем логиты сходства (косинусную близость)
        logits = torch.matmul(projected_norm, prototypes_norm.t()) / self.temperature
        
        return {
            'logits': logits,
            'projected_features': projected_norm,
            'prototypes': prototypes_norm
        }
        
    def update_prototypes(self, features, labels):
        """
        Обновляет прототипы на основе новых примеров.
        
        Args:
            features (torch.Tensor): Признаки примеров [N, embed_dim]
            labels (torch.Tensor): Метки классов примеров [N]
            
        Returns:
            dict: {
                'updated_prototypes': torch.Tensor,  # Обновленные прототипы
                'num_examples_per_class': list  # Количество примеров для каждого класса
            }
        """
        # Проецируем признаки
        projected = self.projection(features)
        projected_norm = F.normalize(projected, dim=1)
        
        # Статистика для отчета
        num_examples = [0] * self.num_prototypes
        
        # Копируем текущие прототипы
        new_prototypes = self.prototypes.clone()
        
        # Обновляем прототипы
        for i in range(self.num_prototypes):
            mask = (labels == i)
            if mask.sum() > 0:
                # Усредняем признаки для класса
                class_features = projected_norm[mask]
                class_mean = torch.mean(class_features, dim=0)
                
                # Обновляем прототип
                new_prototypes[i] = F.normalize(class_mean, dim=0)
                
                # Обновляем статистику
                num_examples[i] = mask.sum().item()
                
        # Обновляем параметр прототипов
        with torch.no_grad():
            self.prototypes.copy_(new_prototypes)
            
        return {
            'updated_prototypes': new_prototypes,
            'num_examples_per_class': num_examples
        }


class AdaptableColorizer(nn.Module):
    """
    Адаптируемый колоризатор с поддержкой few-shot обучения.
    
    Args:
        colorizer (nn.Module): Базовая модель колоризации
        adapter_config (dict): Конфигурация адаптеров
        prototype_config (dict): Конфигурация прототипической сети
    """
    def __init__(self, colorizer, adapter_config=None, prototype_config=None):
        super(AdaptableColorizer, self).__init__()
        
        self.colorizer = colorizer
        
        # Конфигурация по умолчанию для адаптеров
        if adapter_config is None:
            adapter_config = {
                'enabled': True,
                'adapter_locations': [],  # Будет заполнено автоматически
                'bottleneck_dim': 64,
                'dropout_rate': 0.1
            }
            
        # Конфигурация по умолчанию для прототипической сети
        if prototype_config is None:
            prototype_config = {
                'enabled': True,
                'embed_dim': 512,
                'prototype_dim': 128,
                'num_prototypes': 16,
                'temperature': 0.5
            }
            
        self.adapter_config = adapter_config
        self.prototype_config = prototype_config
        
        # Создаем адаптеры
        self.adapters = None
        if adapter_config['enabled']:
            self.adapters = self._create_adapters()
            
        # Создаем прототипическую сеть
        self.prototype_network = None
        if prototype_config['enabled']:
            self.prototype_network = PrototypicalNetwork(
                embed_dim=prototype_config['embed_dim'],
                prototype_dim=prototype_config['prototype_dim'],
                num_prototypes=prototype_config['num_prototypes'],
                temperature=prototype_config['temperature']
            )
            
        # Регистрируем хуки для извлечения промежуточных признаков
        self.hooks = []
        self.feature_maps = {}
        self._register_hooks()
        
        # Режим адаптации (True - с адаптерами, False - без)
        self.adaptation_mode = True
        
    def _create_adapters(self):
        """
        Создает адаптеры для модели.
        
        Returns:
            nn.ModuleDict: Словарь адаптеров
        """
        adapters = nn.ModuleDict()
        
        # Если расположения адаптеров не указаны, находим их автоматически
        if not self.adapter_config['adapter_locations']:
            self._auto_detect_adapter_locations()
            
        # Создаем адаптеры для каждого указанного расположения
        for location in self.adapter_config['adapter_locations']:
            # Находим модуль по имени
            module = self._get_module_by_name(self.colorizer, location)
            
            if module is None:
                print(f"Предупреждение: модуль не найден для адаптера {location}")
                continue
                
            # Определяем тип адаптера в зависимости от типа модуля
            if isinstance(module, nn.Conv2d):
                # Для сверточных слоев используем AdapterBlock
                in_channels = module.out_channels
                adapters[location] = AdapterBlock(
                    in_channels=in_channels,
                    bottleneck_dim=min(in_channels // 2, self.adapter_config['bottleneck_dim']),
                    dropout_rate=self.adapter_config['dropout_rate']
                )
            elif isinstance(module, nn.Linear):
                # Для линейных слоев используем FeatureAdapter
                in_features = module.out_features
                adapters[location] = FeatureAdapter(
                    in_features=in_features,
                    bottleneck_dim=self.adapter_config['bottleneck_dim'],
                    dropout_rate=self.adapter_config['dropout_rate']
                )
            else:
                print(f"Предупреждение: неподдерживаемый тип модуля для адаптера {location}: {type(module)}")
                
        return adapters
    
    def _auto_detect_adapter_locations(self):
        """
        Автоматически определяет расположения для адаптеров.
        """
        # Список расположений для адаптеров
        locations = []
        
        # Находим подходящие слои
        for name, module in self.colorizer.named_modules():
            # Пропускаем корневой модуль
            if name == '':
                continue
                
            # Добавляем сверточные и линейные слои в стратегических позициях
            if isinstance(module, nn.Conv2d):
                # Добавляем слои с большим количеством каналов
                if module.out_channels >= 128:
                    locations.append(name)
            elif isinstance(module, nn.Linear):
                # Добавляем линейные слои с большим количеством выходов
                if module.out_features >= 256:
                    locations.append(name)
                    
        # Ограничиваем количество адаптеров
        max_adapters = 8
        if len(locations) > max_adapters:
            # Выбираем равномерно распределенные адаптеры
            indices = np.linspace(0, len(locations) - 1, max_adapters, dtype=int)
            locations = [locations[i] for i in indices]
            
        self.adapter_config['adapter_locations'] = locations
        print(f"Автоматически обнаружены расположения адаптеров: {locations}")
    
    def _get_module_by_name(self, model, name):
        """
        Получает модуль по имени.
        
        Args:
            model (nn.Module): Модель
            name (str): Имя модуля
            
        Returns:
            nn.Module: Найденный модуль или None, если не найден
        """
        names = name.split('.')
        module = model
        
        for n in names:
            if not hasattr(module, n):
                return None
            module = getattr(module, n)
            
        return module
    
    def _register_hooks(self):
        """
        Регистрирует хуки для извлечения промежуточных признаков.
        """
        # Очищаем предыдущие хуки
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # Словарь для хранения признаков
        self.feature_maps = {}
        
        # Регистрируем хуки для всех адаптеров
        if self.adapters is not None:
            for location in self.adapters.keys():
                module = self._get_module_by_name(self.colorizer, location)
                
                if module is not None:
                    def hook_fn(module, input, output, location=location):
                        # Сохраняем выход модуля
                        self.feature_maps[location] = output
                        
                        # Применяем адаптер, если он включен
                        if self.adaptation_mode and location in self.adapters:
                            return self.adapters[location](output)
                        else:
                            return output
                            
                    # Регистрируем хук
                    hook = module.register_forward_hook(hook_fn)
                    self.hooks.append(hook)
                    
        # Регистрируем хук для извлечения глобальных признаков
        # Пытаемся найти финальный слой энкодера или последний сверточный слой
        for name, module in self.colorizer.named_modules():
            if isinstance(module, nn.AdaptiveAvgPool2d) or isinstance(module, nn.AdaptiveMaxPool2d):
                def global_features_hook(module, input, output, name=name):
                    # Сохраняем выход модуля как глобальные признаки
                    self.feature_maps['global'] = input[0]  # input[0] содержит признаки до пулинга
                    return output
                    
                # Регистрируем хук
                hook = module.register_forward_hook(global_features_hook)
                self.hooks.append(hook)
                break
        
    def extract_global_features(self, x):
        """
        Извлекает глобальные признаки из изображения.
        
        Args:
            x (torch.Tensor): Входное изображение [B, C, H, W]
            
        Returns:
            torch.Tensor: Глобальные признаки [B, embed_dim]
        """
        # Очищаем сохраненные признаки
        self.feature_maps = {}
        
        # Отключаем адаптацию для извлечения оригинальных признаков
        original_mode = self.adaptation_mode
        self.adaptation_mode = False
        
        # Прямое распространение через модель
        with torch.no_grad():
            _ = self.colorizer(x)
            
        # Восстанавливаем режим адаптации
        self.adaptation_mode = original_mode
        
        # Извлекаем глобальные признаки
        if 'global' in self.feature_maps:
            global_features = self.feature_maps['global']
            
            # Применяем глобальный пулинг
            global_features = F.adaptive_avg_pool2d(global_features, 1).flatten(1)
            
            return global_features
        else:
            raise RuntimeError("Не удалось извлечь глобальные признаки. Проверьте регистрацию хуков.")
            
    def set_adaptation_mode(self, mode):
        """
        Устанавливает режим адаптации.
        
        Args:
            mode (bool): True - с адаптерами, False - без
        """
        self.adaptation_mode = mode
        
    def forward(self, x, domain_code=None):
        """
        Прямое распространение через адаптируемый колоризатор.
        
        Args:
            x (torch.Tensor): Входное изображение [B, C, H, W]
            domain_code (torch.Tensor, optional): Код домена для FiLM модуляции
            
        Returns:
            dict: {
                'colorized': torch.Tensor,  # Колоризованное изображение
                'prototype_logits': torch.Tensor (optional),  # Логиты сходства с прототипами
                (и другие выходы базовой модели)
            }
        """
        # Прямое распространение через базовую модель
        # Адаптеры применяются автоматически через хуки
        result = self.colorizer(x)
        
        # Если результат не словарь, преобразуем его
        if not isinstance(result, dict):
            result = {'colorized': result}
            
        # Если прототипическая сеть включена и глобальные признаки доступны
        if self.prototype_network is not None and 'global' in self.feature_maps:
            # Извлекаем глобальные признаки
            global_features = self.feature_maps['global']
            
            # Применяем глобальный пулинг
            global_features = F.adaptive_avg_pool2d(global_features, 1).flatten(1)
            
            # Вычисляем сходство с прототипами
            prototype_result = self.prototype_network(global_features)
            
            # Добавляем результаты в выход
            result['prototype_logits'] = prototype_result['logits']
            result['projected_features'] = prototype_result['projected_features']
            
        return result
    
    def few_shot_adapt(self, support_images, support_labels=None, num_steps=100, learning_rate=1e-4):
        """
        Адаптирует модель на основе нескольких опорных изображений.
        
        Args:
            support_images (torch.Tensor): Опорные изображения [N, C, H, W]
            support_labels (torch.Tensor, optional): Метки опорных изображений [N]
            num_steps (int): Количество шагов обновления
            learning_rate (float): Скорость обучения
            
        Returns:
            dict: Результаты адаптации
        """
        # Проверяем, что адаптеры включены
        if self.adapters is None:
            print("Предупреждение: адаптеры не включены, адаптация не будет применена")
            return {'status': 'error', 'message': 'Адаптеры не включены'}
        
        # Включаем режим обучения для адаптеров
        for adapter in self.adapters.values():
            adapter.train()
            
        # Отключаем градиенты для базовой модели
        for param in self.colorizer.parameters():
            param.requires_grad = False
            
        # Включаем градиенты для адаптеров
        for adapter in self.adapters.values():
            for param in adapter.parameters():
                param.requires_grad = True
                
        # Создаем оптимизатор
        optimizer = torch.optim.Adam(
            [p for adapter in self.adapters.values() for p in adapter.parameters()],
            lr=learning_rate
        )
        
        # Если метки не предоставлены, используем прототипическую сеть для кластеризации
        if support_labels is None and self.prototype_network is not None:
            # Извлекаем признаки
            with torch.no_grad():
                features = self.extract_global_features(support_images)
                
            # Получаем логиты
            with torch.no_grad():
                prototype_result = self.prototype_network(features)
                logits = prototype_result['logits']
                
            # Используем argmax как псевдо-метки
            support_labels = torch.argmax(logits, dim=1)
            
        # Обновляем прототипы, если они включены и метки предоставлены
        if self.prototype_network is not None and support_labels is not None:
            with torch.no_grad():
                features = self.extract_global_features(support_images)
                self.prototype_network.update_prototypes(features, support_labels)
            
        # Выполняем несколько шагов обновления
        losses = []
        
        for step in range(num_steps):
            # Обнуляем градиенты
            optimizer.zero_grad()
            
            # Прямое распространение
            output = self.forward(support_images)
            colorized = output['colorized']
            
            # Вычисляем потерю (L1 по отношению к истинным цветам)
            if colorized.shape[1] > 1:  # Если это цветное изображение
                # Предполагаем, что первый канал - яркость, остальные - цветность
                color_channels = colorized[:, 1:]
                target_colors = support_images[:, 1:] if support_images.shape[1] > 1 else support_images
                loss = F.l1_loss(color_channels, target_colors)
            else:
                loss = F.l1_loss(colorized, support_images)
                
            # Обратное распространение
            loss.backward()
            
            # Обновление параметров
            optimizer.step()
            
            # Сохраняем потерю
            losses.append(loss.item())
            
            if step % 10 == 0:
                print(f"Шаг {step}/{num_steps}, потеря: {loss.item():.6f}")
                
        # Возвращаем модель в режим оценки
        for adapter in self.adapters.values():
            adapter.eval()
            
        return {
            'status': 'success',
            'losses': losses,
            'num_steps': num_steps,
            'final_loss': losses[-1]
        }
    
    def save_adapters(self, path):
        """
        Сохраняет адаптеры в файл.
        
        Args:
            path (str): Путь для сохранения
            
        Returns:
            bool: True в случае успеха, иначе False
        """
        if self.adapters is None:
            print("Предупреждение: адаптеры не включены, нечего сохранять")
            return False
            
        # Создаем словарь с состояниями адаптеров
        adapter_states = {
            'adapters': {name: adapter.state_dict() for name, adapter in self.adapters.items()},
            'adapter_config': self.adapter_config
        }
        
        # Добавляем прототипы, если они включены
        if self.prototype_network is not None:
            adapter_states['prototype_network'] = self.prototype_network.state_dict()
            adapter_states['prototype_config'] = self.prototype_config
            
        # Сохраняем в файл
        try:
            torch.save(adapter_states, path)
            print(f"Адаптеры успешно сохранены в {path}")
            return True
        except Exception as e:
            print(f"Ошибка при сохранении адаптеров: {e}")
            return False
            
    def load_adapters(self, path):
        """
        Загружает адаптеры из файла.
        
        Args:
            path (str): Путь для загрузки
            
        Returns:
            bool: True в случае успеха, иначе False
        """
        try:
            # Загружаем состояния
            adapter_states = torch.load(path)
            
            # Проверяем формат
            if 'adapters' not in adapter_states or 'adapter_config' not in adapter_states:
                print(f"Некорректный формат файла адаптеров: {path}")
                return False
                
            # Обновляем конфигурацию
            self.adapter_config = adapter_states['adapter_config']
            
            # Создаем новые адаптеры, если их нет или конфигурация изменилась
            if self.adapters is None:
                self.adapters = self._create_adapters()
                
            # Загружаем состояния адаптеров
            for name, state in adapter_states['adapters'].items():
                if name in self.adapters:
                    self.adapters[name].load_state_dict(state)
                else:
                    print(f"Предупреждение: адаптер {name} не найден в текущей модели")
                    
            # Загружаем прототипы, если они включены
            if 'prototype_network' in adapter_states and self.prototype_network is not None:
                self.prototype_network.load_state_dict(adapter_states['prototype_network'])
                self.prototype_config = adapter_states['prototype_config']
                
            # Перерегистрируем хуки
            self._register_hooks()
            
            print(f"Адаптеры успешно загружены из {path}")
            return True
        except Exception as e:
            print(f"Ошибка при загрузке адаптеров: {e}")
            return False


class MetaLearningAdapter(nn.Module):
    """
    Адаптер с использованием мета-обучения для быстрой адаптации.
    
    Args:
        colorizer (nn.Module): Базовая модель колоризации
        feature_dim (int): Размерность признаков
        inner_lr (float): Скорость обучения для внутреннего обновления
        meta_lr (float): Скорость обучения для мета-обновления
    """
    def __init__(self, colorizer, feature_dim=512, inner_lr=0.01, meta_lr=0.001):
        super(MetaLearningAdapter, self).__init__()
        
        self.colorizer = colorizer
        self.feature_dim = feature_dim
        self.inner_lr = inner_lr
        
        # Создаем адаптеры для мета-обучения
        self.meta_adapters = nn.ModuleDict({
            'encoder': nn.Linear(feature_dim, feature_dim),
            'decoder': nn.Linear(feature_dim, feature_dim)
        })
        
        # Создаем оптимизатор для мета-обучения
        self.meta_optimizer = torch.optim.Adam(self.meta_adapters.parameters(), lr=meta_lr)
        
        # Регистрируем хуки для извлечения промежуточных признаков
        self.hooks = []
        self.feature_maps = {}
        self._register_hooks()
        
    def _register_hooks(self):
        """
        Регистрирует хуки для извлечения промежуточных признаков.
        """
        # Очищаем предыдущие хуки
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # Словарь для хранения признаков
        self.feature_maps = {}
        
        # Находим слои для адаптации
        encoder_found = False
        decoder_found = False
        
        for name, module in self.colorizer.named_modules():
            # Ищем подходящий слой энкодера
            if not encoder_found and ("encoder" in name.lower() or "down" in name.lower()):
                if isinstance(module, nn.Conv2d) and module.out_channels >= self.feature_dim:
                    def encoder_hook(module, input, output, name=name):
                        self.feature_maps['encoder_input'] = input[0]
                        self.feature_maps['encoder_output'] = output
                        return output
                        
                    hook = module.register_forward_hook(encoder_hook)
                    self.hooks.append(hook)
                    encoder_found = True
                    print(f"Зарегистрирован хук энкодера для слоя: {name}")
                    
            # Ищем подходящий слой декодера
            if not decoder_found and ("decoder" in name.lower() or "up" in name.lower()):
                if isinstance(module, nn.Conv2d) and module.in_channels >= self.feature_dim:
                    def decoder_hook(module, input, output, name=name):
                        self.feature_maps['decoder_input'] = input[0]
                        self.feature_maps['decoder_output'] = output
                        return output
                        
                    hook = module.register_forward_hook(decoder_hook)
                    self.hooks.append(hook)
                    decoder_found = True
                    print(f"Зарегистрирован хук декодера для слоя: {name}")
                    
        if not encoder_found or not decoder_found:
            print("Предупреждение: не удалось найти подходящие слои для хуков")
            
    def meta_learn(self, support_set, query_set, num_tasks=10, inner_steps=5):
        """
        Выполняет мета-обучение на наборе задач.
        
        Args:
            support_set (list): Список опорных наборов [тензор изображений] для каждой задачи
            query_set (list): Список тестовых наборов [тензор изображений] для каждой задачи
            num_tasks (int): Количество задач для мета-обучения
            inner_steps (int): Количество шагов внутреннего обновления
            
        Returns:
            dict: Результаты мета-обучения
        """
        # Проверяем данные
        if len(support_set) < num_tasks or len(query_set) < num_tasks:
            print("Предупреждение: недостаточно задач для мета-обучения")
            num_tasks = min(len(support_set), len(query_set))
            
        # Переводим модель в режим обучения
        self.train()
        
        # Мета-обучение
        meta_losses = []
        
        for task_idx in range(num_tasks):
            # Получаем данные для текущей задачи
            support_images = support_set[task_idx]
            query_images = query_set[task_idx]
            
            # Создаем копию параметров для внутреннего обновления
            task_adapters = {
                'encoder': self.meta_adapters['encoder'].weight.clone().requires_grad_(True),
                'decoder': self.meta_adapters['decoder'].weight.clone().requires_grad_(True)
            }
            
            # Внутреннее обновление на опорном наборе
            for _ in range(inner_steps):
                # Прямое распространение через модель
                _ = self.colorizer(support_images)
                
                # Извлекаем признаки
                if 'encoder_output' in self.feature_maps and 'decoder_input' in self.feature_maps:
                    # Применяем адаптеры
                    encoder_out = self._apply_adapter(self.feature_maps['encoder_output'], task_adapters['encoder'])
                    decoder_in = self._apply_adapter(self.feature_maps['decoder_input'], task_adapters['decoder'])
                    
                    # Вычисляем потерю (L1 по отношению к истинным цветам)
                    loss = F.l1_loss(encoder_out, self.feature_maps['encoder_output']) + \
                           F.l1_loss(decoder_in, self.feature_maps['decoder_input'])
                           
                    # Обновляем параметры адаптеров
                    encoder_grad = torch.autograd.grad(loss, task_adapters['encoder'], retain_graph=True)[0]
                    decoder_grad = torch.autograd.grad(loss, task_adapters['decoder'])[0]
                    
                    task_adapters['encoder'] = task_adapters['encoder'] - self.inner_lr * encoder_grad
                    task_adapters['decoder'] = task_adapters['decoder'] - self.inner_lr * decoder_grad
                    
            # Оценка на тестовом наборе
            self.meta_optimizer.zero_grad()
            
            # Прямое распространение через модель
            _ = self.colorizer(query_images)
            
            # Извлекаем признаки и применяем адаптеры
            if 'encoder_output' in self.feature_maps and 'decoder_input' in self.feature_maps:
                # Применяем адаптеры
                encoder_out = self._apply_adapter(self.feature_maps['encoder_output'], task_adapters['encoder'])
                decoder_in = self._apply_adapter(self.feature_maps['decoder_input'], task_adapters['decoder'])
                
                # Вычисляем мета-потерю
                meta_loss = F.l1_loss(encoder_out, self.feature_maps['encoder_output']) + \
                           F.l1_loss(decoder_in, self.feature_maps['decoder_input'])
                           
                # Обратное распространение и обновление мета-параметров
                meta_loss.backward()
                self.meta_optimizer.step()
                
                # Сохраняем мета-потерю
                meta_losses.append(meta_loss.item())
                
        # Возвращаем результаты
        return {
            'status': 'success',
            'meta_losses': meta_losses,
            'avg_meta_loss': sum(meta_losses) / len(meta_losses) if meta_losses else 0.0
        }
    
    def _apply_adapter(self, features, adapter_weight):
        """
        Применяет адаптер к признакам.
        
        Args:
            features (torch.Tensor): Признаки [B, C, H, W]
            adapter_weight (torch.Tensor): Веса адаптера
            
        Returns:
            torch.Tensor: Адаптированные признаки [B, C, H, W]
        """
        # Сохраняем оригинальную форму
        original_shape = features.shape
        
        # Преобразуем признаки для применения линейного преобразования
        features_flat = features.view(-1, original_shape[1])
        
        # Применяем адаптер
        adapted = torch.matmul(features_flat, adapter_weight.T)
        
        # Восстанавливаем исходную форму
        adapted = adapted.view(original_shape)
        
        return adapted
    
    def forward(self, x):
        """
        Прямое распространение через адаптер.
        
        Args:
            x (torch.Tensor): Входное изображение [B, C, H, W]
            
        Returns:
            dict: Результат колоризации
        """
        # Прямое распространение через базовую модель
        result = self.colorizer(x)
        
        # Если это режим обучения, просто возвращаем результат
        if self.training:
            return result
            
        # В режиме оценки применяем адаптеры к промежуточным признакам
        if 'encoder_output' in self.feature_maps and 'decoder_input' in self.feature_maps:
            # Извлекаем признаки
            encoder_features = self.feature_maps['encoder_output']
            decoder_features = self.feature_maps['decoder_input']
            
            # Применяем адаптеры
            encoder_adapted = self._apply_adapter(encoder_features, self.meta_adapters['encoder'].weight)
            decoder_adapted = self._apply_adapter(decoder_features, self.meta_adapters['decoder'].weight)
            
            # TODO: Здесь можно добавить логику для модификации результата
            # на основе адаптированных признаков
            
        return result


# Создаем функцию для создания адаптера
def create_few_shot_adapter(colorizer=None, config=None):
    """
    Создает адаптер для few-shot обучения.
    
    Args:
        colorizer (nn.Module, optional): Базовая модель колоризации
        config (dict, optional): Конфигурация адаптера
        
    Returns:
        nn.Module: Few-shot адаптер
    """
    if colorizer is None:
        raise ValueError("Необходимо предоставить базовую модель колоризации")
        
    if config is None:
        config = {}
        
    # Параметры по умолчанию
    adapter_type = config.get('adapter_type', 'standard')  # standard, meta
    
    # Конфигурация для стандартного адаптера
    adapter_config = config.get('adapter_config', {
        'enabled': True,
        'adapter_locations': [],  # Автоопределение
        'bottleneck_dim': 64,
        'dropout_rate': 0.1
    })
    
    # Конфигурация для прототипической сети
    prototype_config = config.get('prototype_config', {
        'enabled': True,
        'embed_dim': 512,
        'prototype_dim': 128,
        'num_prototypes': 16,
        'temperature': 0.5
    })
    
    # Конфигурация для мета-обучения
    meta_config = config.get('meta_config', {
        'feature_dim': 512,
        'inner_lr': 0.01,
        'meta_lr': 0.001
    })
    
    # Создаем адаптер нужного типа
    if adapter_type == 'standard':
        return AdaptableColorizer(
            colorizer=colorizer,
            adapter_config=adapter_config,
            prototype_config=prototype_config
        )
    elif adapter_type == 'meta':
        return MetaLearningAdapter(
            colorizer=colorizer,
            feature_dim=meta_config['feature_dim'],
            inner_lr=meta_config['inner_lr'],
            meta_lr=meta_config['meta_lr']
        )
    else:
        raise ValueError(f"Неизвестный тип адаптера: {adapter_type}")


if __name__ == "__main__":
    # Пример использования few-shot адаптера
    
    # Создаем простую модель колоризации для демонстрации
    class SimpleColorizer(nn.Module):
        def __init__(self):
            super(SimpleColorizer, self).__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            self.decoder = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 2, kernel_size=3, padding=1)
            )
            
        def forward(self, x):
            features = self.encoder(x)
            output = self.decoder(features)
            return {'colorized': torch.cat([x, output], dim=1)}
    
    # Создаем модель колоризации
    colorizer = SimpleColorizer()
    
    # Создаем few-shot адаптер
    adapter = create_few_shot_adapter(
        colorizer=colorizer,
        config={
            'adapter_type': 'standard',
            'adapter_config': {
                'enabled': True,
                'bottleneck_dim': 32
            }
        }
    )
    
    # Создаем тестовые данные
    batch_size = 2
    grayscale_images = torch.randn(batch_size, 1, 64, 64)
    color_images = torch.randn(batch_size, 3, 64, 64)
    
    # Адаптируем модель на основе нескольких примеров
    adaptation_result = adapter.few_shot_adapt(
        support_images=color_images,
        num_steps=10
    )
    
    # Применяем адаптированную модель
    with torch.no_grad():
        result = adapter(grayscale_images)
    
    # Выводим информацию о результате
    print("\nРезультаты адаптации:")
    for key, value in adaptation_result.items():
        if isinstance(value, list):
            print(f"  {key}: список из {len(value)} значений")
        else:
            print(f"  {key}: {value}")
    
    print("\nРезультаты колоризации:")
    for key, value in result.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")