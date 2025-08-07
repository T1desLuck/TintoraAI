"""
Uncertainty Estimation: Модуль для оценки неопределенности в процессе колоризации изображений.

Данный модуль предоставляет методы для оценки неопределенности при колоризации изображений,
что помогает определить области, где модель наименее уверена в выборе цвета. Это особенно
полезно для редких объектов, двусмысленных сцен или объектов, которые могут иметь
множество допустимых цветовых вариаций.

Ключевые особенности:
- Байесовские методы для оценки эпистемической неопределенности (незнание)
- Прямые методы для оценки алеаторической неопределенности (внутренняя вариативность)
- MC Dropout для получения распределения возможных результатов
- Ансамблевые методы для более надежных оценок
- Визуализация карт неопределенности для анализа и взаимодействия с пользователем

Преимущества для колоризации:
- Выявление проблемных областей для интерактивного уточнения
- Предотвращение неестественных цветов в зонах с высокой неопределенностью
- Более информированные решения при слиянии нескольких результатов
- Возможность генерации нескольких правдоподобных вариантов колоризации
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Union, Optional


class DropoutColorizerWrapper(nn.Module):
    """
    Обертка для модели колоризации, добавляющая dropout для байесовской оценки неопределенности.
    
    Args:
        colorizer (nn.Module): Базовая модель колоризации
        dropout_rate (float): Вероятность выключения нейронов при dropout
        dropout_layers (list): Список имен слоев, к которым применяется dropout
    """
    def __init__(self, colorizer, dropout_rate=0.1, dropout_layers=None):
        super(DropoutColorizerWrapper, self).__init__()
        
        self.colorizer = colorizer
        self.dropout_rate = dropout_rate
        self.dropout_layers = dropout_layers or []
        
        # Добавляем dropout к указанным слоям
        self._add_dropout_layers()
        
    def _add_dropout_layers(self):
        """
        Добавляет слои dropout к указанным слоям модели.
        """
        if not self.dropout_layers:
            # Если слои не указаны, пытаемся найти их автоматически
            for name, module in self.colorizer.named_modules():
                # Добавляем dropout после сверточных слоев и перед активациями
                if isinstance(module, nn.Conv2d) and hasattr(module, 'add_dropout_after'):
                    setattr(self.colorizer, name + '_dropout', nn.Dropout2d(self.dropout_rate))
                    self.dropout_layers.append(name)
        else:
            # Добавляем dropout к указанным слоям
            for layer_name in self.dropout_layers:
                if hasattr(self.colorizer, layer_name):
                    parent_name = '.'.join(layer_name.split('.')[:-1])
                    if parent_name:
                        parent = getattr(self.colorizer, parent_name)
                        setattr(parent, layer_name.split('.')[-1] + '_dropout', 
                                nn.Dropout2d(self.dropout_rate))
                    else:
                        setattr(self.colorizer, layer_name + '_dropout', 
                                nn.Dropout2d(self.dropout_rate))
                        
    def forward(self, x, **kwargs):
        """
        Прямое распространение через модель с включенным dropout.
        
        Args:
            x (torch.Tensor): Входное изображение
            **kwargs: Дополнительные аргументы для модели колоризации
            
        Returns:
            dict or torch.Tensor: Результат работы модели колоризации
        """
        # Включаем режим обучения, чтобы dropout работал
        self.train()
        
        # Прямое распространение через базовую модель
        return self.colorizer(x, **kwargs)


class MCDropoutUncertainty(nn.Module):
    """
    Оценка неопределенности методом Monte Carlo Dropout.
    
    Args:
        colorizer (nn.Module): Модель колоризации
        num_samples (int): Количество выборок для оценки неопределенности
        dropout_rate (float): Вероятность выключения нейронов при dropout
        dropout_layers (list): Список имен слоев, к которым применяется dropout
    """
    def __init__(self, colorizer, num_samples=10, dropout_rate=0.1, dropout_layers=None):
        super(MCDropoutUncertainty, self).__init__()
        
        # Оборачиваем колоризатор в DropoutColorizerWrapper
        self.colorizer_with_dropout = DropoutColorizerWrapper(
            colorizer=colorizer,
            dropout_rate=dropout_rate,
            dropout_layers=dropout_layers
        )
        
        self.num_samples = num_samples
        
    def forward(self, x, **kwargs):
        """
        Генерирует несколько сэмплов колоризации и вычисляет неопределенность.
        
        Args:
            x (torch.Tensor): Входное изображение [B, C, H, W]
            **kwargs: Дополнительные аргументы для модели колоризации
            
        Returns:
            dict: {
                'colorized': torch.Tensor,  # Среднее значение колоризации [B, C, H, W]
                'uncertainty': torch.Tensor,  # Карта неопределенности [B, 1, H, W]
                'samples': torch.Tensor,  # Сэмплы колоризации [B, num_samples, C, H, W]
                'variance': torch.Tensor,  # Дисперсия по каналам [B, C, H, W]
                'entropy': torch.Tensor  # Энтропия [B, 1, H, W]
            }
        """
        batch_size = x.shape[0]
        
        # Список для хранения результатов
        samples = []
        
        # Генерируем несколько сэмплов
        for _ in range(self.num_samples):
            # Получаем результат от модели
            result = self.colorizer_with_dropout(x, **kwargs)
            
            # Извлекаем колоризованное изображение (предполагаем, что результат либо тензор, либо словарь)
            if isinstance(result, dict) and 'colorized' in result:
                colorized = result['colorized']
            elif isinstance(result, dict) and 'output' in result:
                colorized = result['output']
            elif isinstance(result, torch.Tensor):
                colorized = result
            else:
                raise ValueError("Неизвестный формат вывода модели колоризации")
                
            samples.append(colorized)
            
        # Объединяем сэмплы в один тензор [B, num_samples, C, H, W]
        samples_tensor = torch.stack(samples, dim=1)
        
        # Вычисляем среднее значение по сэмплам
        mean_colorized = torch.mean(samples_tensor, dim=1)
        
        # Вычисляем дисперсию по сэмплам
        variance = torch.var(samples_tensor, dim=1)
        
        # Вычисляем общую неопределенность как среднюю дисперсию по цветовым каналам
        # Обычно нас интересует неопределенность в цветовых каналах (a и b в Lab)
        if mean_colorized.size(1) > 1:  # Если есть цветовые каналы
            color_channels = mean_colorized.size(1) - 1  # Исключаем канал L (яркость)
            color_variance = variance[:, 1:]  # Берем только цветовые каналы (a и b)
            uncertainty = torch.mean(color_variance, dim=1, keepdim=True)
        else:
            uncertainty = torch.mean(variance, dim=1, keepdim=True)
            
        # Вычисляем энтропию (более сложная мера неопределенности)
        # Для этого предполагаем нормальное распределение и используем формулу энтропии
        # для нормального распределения: H(X) = 0.5 * log(2 * pi * e * sigma^2)
        entropy = 0.5 * torch.log(2 * math.pi * math.e * uncertainty)
        
        return {
            'colorized': mean_colorized,
            'uncertainty': uncertainty,
            'samples': samples_tensor,
            'variance': variance,
            'entropy': entropy
        }


class EnsembleUncertainty(nn.Module):
    """
    Оценка неопределенности методом ансамбля моделей.
    
    Args:
        colorizers (list): Список моделей колоризации
    """
    def __init__(self, colorizers):
        super(EnsembleUncertainty, self).__init__()
        
        self.colorizers = nn.ModuleList(colorizers)
        
    def forward(self, x, **kwargs):
        """
        Генерирует колоризации от разных моделей ансамбля и вычисляет неопределенность.
        
        Args:
            x (torch.Tensor): Входное изображение [B, C, H, W]
            **kwargs: Дополнительные аргументы для моделей колоризации
            
        Returns:
            dict: {
                'colorized': torch.Tensor,  # Среднее значение колоризации [B, C, H, W]
                'uncertainty': torch.Tensor,  # Карта неопределенности [B, 1, H, W]
                'samples': torch.Tensor,  # Колоризации от разных моделей [B, num_models, C, H, W]
                'variance': torch.Tensor,  # Дисперсия по каналам [B, C, H, W]
            }
        """
        # Список для хранения результатов
        samples = []
        
        # Получаем результаты от каждой модели
        for colorizer in self.colorizers:
            # Получаем результат от модели
            result = colorizer(x, **kwargs)
            
            # Извлекаем колоризованное изображение
            if isinstance(result, dict) and 'colorized' in result:
                colorized = result['colorized']
            elif isinstance(result, dict) and 'output' in result:
                colorized = result['output']
            elif isinstance(result, torch.Tensor):
                colorized = result
            else:
                raise ValueError("Неизвестный формат вывода модели колоризации")
                
            samples.append(colorized)
            
        # Объединяем результаты в один тензор [B, num_models, C, H, W]
        samples_tensor = torch.stack(samples, dim=1)
        
        # Вычисляем среднее значение по моделям
        mean_colorized = torch.mean(samples_tensor, dim=1)
        
        # Вычисляем дисперсию по моделям
        variance = torch.var(samples_tensor, dim=1)
        
        # Вычисляем общую неопределенность как среднюю дисперсию по цветовым каналам
        if mean_colorized.size(1) > 1:  # Если есть цветовые каналы
            color_variance = variance[:, 1:]  # Берем только цветовые каналы (a и b)
            uncertainty = torch.mean(color_variance, dim=1, keepdim=True)
        else:
            uncertainty = torch.mean(variance, dim=1, keepdim=True)
            
        return {
            'colorized': mean_colorized,
            'uncertainty': uncertainty,
            'samples': samples_tensor,
            'variance': variance
        }


class DirectUncertaintyEstimator(nn.Module):
    """
    Модуль для прямой оценки неопределенности, обучаемый одновременно с моделью колоризации.
    
    Args:
        input_channels (int): Количество входных каналов
        base_channels (int): Базовое количество каналов в свёрточных слоях
    """
    def __init__(self, input_channels=512, base_channels=64):
        super(DirectUncertaintyEstimator, self).__init__()
        
        # Сеть для оценки неопределенности
        self.uncertainty_network = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, kernel_size=1),
            nn.Sigmoid()  # Нормализуем выход в диапазон [0, 1]
        )
        
    def forward(self, features):
        """
        Оценивает неопределенность на основе признаков.
        
        Args:
            features (torch.Tensor): Признаки из модели колоризации [B, C, H, W]
            
        Returns:
            torch.Tensor: Карта неопределенности [B, 1, H, W]
        """
        return self.uncertainty_network(features)


class ProbabilisticColorizer(nn.Module):
    """
    Вероятностный колоризатор, предсказывающий не только цвета, но и их распределения.
    
    Args:
        colorizer (nn.Module): Базовая модель колоризации
        input_channels (int): Количество входных каналов для оценки неопределенности
        base_channels (int): Базовое количество каналов для оценки неопределенности
    """
    def __init__(self, colorizer, input_channels=512, base_channels=64):
        super(ProbabilisticColorizer, self).__init__()
        
        self.colorizer = colorizer
        
        # Модуль для прямой оценки неопределенности
        self.uncertainty_estimator = DirectUncertaintyEstimator(
            input_channels=input_channels,
            base_channels=base_channels
        )
        
        # Для хука, извлекающего промежуточные признаки
        self.features = None
        self._register_hook()
        
    def _register_hook(self):
        """
        Регистрирует хук для извлечения промежуточных признаков из колоризатора.
        """
        # Найдем последний слой энкодера или предпоследний слой модели
        target_module = None
        target_name = ""
        
        # Ищем слой, к которому можно прикрепить хук
        # Это может быть последний слой энкодера или предпоследний слой всей модели
        for name, module in self.colorizer.named_modules():
            if isinstance(module, nn.Conv2d):
                target_module = module
                target_name = name
            elif "encoder" in name and isinstance(module, nn.Sequential):
                # Если нашли энкодер, берем его последний слой
                target_module = module[-1] if isinstance(module[-1], nn.Conv2d) else None
                if target_module:
                    target_name = name + ".[-1]"
                    break
        
        if target_module is None:
            raise ValueError("Не удалось найти подходящий слой для хука")
        
        # Регистрируем хук
        def hook_fn(module, input, output):
            self.features = output
            
        self.hook = target_module.register_forward_hook(hook_fn)
        print(f"Хук зарегистрирован для слоя: {target_name}")
        
    def forward(self, x, **kwargs):
        """
        Выполняет колоризацию и оценивает неопределенность.
        
        Args:
            x (torch.Tensor): Входное изображение [B, C, H, W]
            **kwargs: Дополнительные аргументы для модели колоризации
            
        Returns:
            dict: {
                'colorized': torch.Tensor,  # Колоризованное изображение [B, C, H, W]
                'uncertainty': torch.Tensor,  # Карта неопределенности [B, 1, H, W]
                'features': torch.Tensor,  # Промежуточные признаки [B, C, H, W]
                (и другие ключи из результата базовой модели)
            }
        """
        # Обнуляем промежуточные признаки
        self.features = None
        
        # Получаем результат от базовой модели
        result = self.colorizer(x, **kwargs)
        
        # Проверяем, что признаки были извлечены
        if self.features is None:
            raise RuntimeError("Не удалось извлечь промежуточные признаки. Хук не сработал.")
            
        # Оцениваем неопределенность
        uncertainty = self.uncertainty_estimator(self.features)
        
        # Преобразуем результат в словарь, если это не словарь
        if not isinstance(result, dict):
            result = {'colorized': result}
            
        # Добавляем оценку неопределенности
        result['uncertainty'] = uncertainty
        
        # Добавляем промежуточные признаки
        result['features'] = self.features
        
        return result


class UncertaintyGuidedColorizer(nn.Module):
    """
    Колоризатор с учетом неопределенности для улучшения качества колоризации в неопределенных областях.
    
    Args:
        colorizer (nn.Module): Базовая модель колоризации
        uncertainty_threshold (float): Порог неопределенности для переключения на альтернативные методы
        use_memory_bank (bool): Использовать ли банк памяти для неопределенных областей
        use_guide_net (bool): Использовать ли советник по цветам для неопределенных областей
    """
    def __init__(self, colorizer, uncertainty_threshold=0.5, use_memory_bank=True, use_guide_net=True):
        super(UncertaintyGuidedColorizer, self).__init__()
        
        # Базовая модель колоризации
        self.colorizer = colorizer
        
        # Порог неопределенности
        self.uncertainty_threshold = uncertainty_threshold
        
        # Вероятностная обертка
        self.probabilistic_colorizer = ProbabilisticColorizer(colorizer)
        
        # Флаги для использования дополнительных модулей
        self.use_memory_bank = use_memory_bank
        self.use_guide_net = use_guide_net
        
        # Здесь должны быть инициализированы дополнительные модули, если они используются
        # Однако, так как они зависят от других файлов, оставляем их на момент полной интеграции
        self.memory_bank = None  # Инициализируется позже
        self.guide_net = None    # Инициализируется позже
        
    def set_memory_bank(self, memory_bank):
        """
        Устанавливает модуль банка памяти.
        
        Args:
            memory_bank (nn.Module): Модуль банка памяти
        """
        self.memory_bank = memory_bank
        
    def set_guide_net(self, guide_net):
        """
        Устанавливает модуль советника по цветам.
        
        Args:
            guide_net (nn.Module): Модуль советника по цветам
        """
        self.guide_net = guide_net
        
    def forward(self, x, reference_image=None, **kwargs):
        """
        Выполняет колоризацию с учетом неопределенности.
        
        Args:
            x (torch.Tensor): Входное изображение [B, C, H, W]
            reference_image (torch.Tensor, optional): Референсное изображение [B, C, H, W]
            **kwargs: Дополнительные аргументы для модели колоризации
            
        Returns:
            dict: {
                'colorized': torch.Tensor,  # Финальное колоризованное изображение [B, C, H, W]
                'uncertainty': torch.Tensor,  # Карта неопределенности [B, 1, H, W]
                'base_colorized': torch.Tensor,  # Результат базовой модели [B, C, H, W]
                (и другие ключи в зависимости от использованных модулей)
            }
        """
        # Получаем результат от вероятностной модели
        result = self.probabilistic_colorizer(x, **kwargs)
        
        # Извлекаем колоризованное изображение и карту неопределенности
        colorized = result['colorized']
        uncertainty = result['uncertainty']
        
        # Создаем маску для неопределенных областей
        uncertain_mask = uncertainty > self.uncertainty_threshold
        
        # Если нет неопределенных областей или дополнительные модули не используются, возвращаем результат как есть
        if not torch.any(uncertain_mask) or (not self.use_memory_bank and not self.use_guide_net):
            return result
        
        # Сохраняем оригинальный результат
        result['base_colorized'] = colorized.clone()
        
        # Используем банк памяти для неопределенных областей, если доступен
        if self.use_memory_bank and self.memory_bank is not None:
            memory_result = self.memory_bank(x)
            
            if 'colorized' in memory_result:
                # Создаем смесь на основе неопределенности
                blend_weight = uncertainty.clone().detach()  # [B, 1, H, W]
                # Нормализуем веса
                blend_weight = (blend_weight - self.uncertainty_threshold) / (1 - self.uncertainty_threshold)
                blend_weight = torch.clamp(blend_weight, 0, 1)
                
                # Применяем смешивание только к неопределенным областям
                colorized = torch.where(
                    uncertain_mask.expand_as(colorized),
                    colorized * (1 - blend_weight) + memory_result['colorized'] * blend_weight,
                    colorized
                )
                
                # Добавляем результат банка памяти в общий результат
                result['memory_colorized'] = memory_result['colorized']
            
        # Используем советника по цветам для неопределенных областей, если доступен
        if self.use_guide_net and self.guide_net is not None:
            guide_result = self.guide_net(x, reference_image)
            
            if 'color_advice' in guide_result:
                # Получаем советы по цветам
                color_advice = guide_result['color_advice']
                confidence = guide_result.get('confidence', torch.ones_like(uncertainty))
                
                # Создаем смесь на основе неопределенности и уверенности советника
                blend_weight = uncertainty.clone().detach() * confidence
                blend_weight = torch.clamp(blend_weight, 0, 1)
                
                # Применяем смешивание только к неопределенным областям
                color_advice_expanded = F.interpolate(
                    color_advice, 
                    size=colorized.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
                
                colorized = torch.where(
                    uncertain_mask.expand_as(colorized),
                    colorized * (1 - blend_weight) + color_advice_expanded * blend_weight,
                    colorized
                )
                
                # Добавляем результат советника в общий результат
                result['guide_colorized'] = color_advice_expanded
            
        # Обновляем колоризованное изображение в результате
        result['colorized'] = colorized
        
        return result


class MultiLevelUncertainty(nn.Module):
    """
    Оценка неопределенности на разных уровнях абстракции.
    
    Args:
        feature_extractors (list): Список модулей для извлечения признаков на разных уровнях
        input_channels_list (list): Список количества каналов для каждого уровня
    """
    def __init__(self, feature_extractors, input_channels_list):
        super(MultiLevelUncertainty, self).__init__()
        
        assert len(feature_extractors) == len(input_channels_list), \
            "Количество экстракторов признаков должно соответствовать количеству входных каналов"
        
        self.feature_extractors = nn.ModuleList(feature_extractors)
        
        # Создаем оценщики неопределенности для каждого уровня
        self.uncertainty_estimators = nn.ModuleList([
            DirectUncertaintyEstimator(input_channels=channels)
            for channels in input_channels_list
        ])
        
    def forward(self, x, feature_maps=None):
        """
        Оценивает неопределенность на разных уровнях абстракции.
        
        Args:
            x (torch.Tensor): Входное изображение [B, C, H, W]
            feature_maps (list, optional): Предварительно извлеченные карты признаков
            
        Returns:
            dict: {
                'uncertainty_maps': list,  # Карты неопределенности на разных уровнях
                'combined_uncertainty': torch.Tensor,  # Объединенная карта неопределенности [B, 1, H, W]
                'feature_maps': list  # Извлеченные карты признаков
            }
        """
        uncertainty_maps = []
        
        # Если карты признаков не предоставлены, извлекаем их
        if feature_maps is None:
            feature_maps = []
            current_x = x
            
            for extractor in self.feature_extractors:
                current_x = extractor(current_x)
                feature_maps.append(current_x)
                
        # Оцениваем неопределенность на каждом уровне
        for i, (feature_map, estimator) in enumerate(zip(feature_maps, self.uncertainty_estimators)):
            uncertainty = estimator(feature_map)
            
            # Изменяем размер карты неопределенности для соответствия входному изображению
            if uncertainty.shape[2:] != x.shape[2:]:
                uncertainty = F.interpolate(
                    uncertainty, 
                    size=x.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
                
            uncertainty_maps.append(uncertainty)
            
        # Объединяем карты неопределенности
        # Используем взвешенное среднее с большим весом для высокоуровневых карт
        weights = torch.linspace(0.5, 1.5, len(uncertainty_maps), device=x.device)
        weights = weights / weights.sum()
        
        combined_uncertainty = torch.zeros_like(uncertainty_maps[0])
        for i, uncertainty in enumerate(uncertainty_maps):
            combined_uncertainty += weights[i] * uncertainty
            
        return {
            'uncertainty_maps': uncertainty_maps,
            'combined_uncertainty': combined_uncertainty,
            'feature_maps': feature_maps
        }


class UncertaintyVisualization:
    """
    Методы для визуализации неопределенности колоризации.
    """
    @staticmethod
    def create_uncertainty_heatmap(uncertainty_map, colormap='inferno'):
        """
        Создает тепловую карту неопределенности.
        
        Args:
            uncertainty_map (torch.Tensor): Карта неопределенности [B, 1, H, W]
            colormap (str): Название цветовой карты ('inferno', 'viridis', 'plasma', 'magma')
            
        Returns:
            np.ndarray: Тепловая карта неопределенности [B, H, W, 3] в RGB
        """
        # Импортируем matplotlib только при необходимости
        import matplotlib.pyplot as plt
        from matplotlib import cm
        
        # Преобразуем в numpy
        uncertainty_np = uncertainty_map.detach().cpu().numpy()
        batch_size = uncertainty_np.shape[0]
        
        # Создаем цветовую карту
        cmap = cm.get_cmap(colormap)
        
        # Создаем тепловые карты для каждого изображения в пакете
        heatmaps = []
        
        for i in range(batch_size):
            # Сжимаем до 2D [H, W]
            uncertainty_2d = uncertainty_np[i, 0]
            
            # Применяем цветовую карту
            heatmap = cmap(uncertainty_2d)[:, :, :3]  # Отбрасываем альфа-канал
            
            heatmaps.append(heatmap)
            
        # Объединяем в один массив [B, H, W, 3]
        return np.stack(heatmaps, axis=0)
    
    @staticmethod
    def overlay_uncertainty(image, uncertainty_map, alpha=0.5, colormap='inferno'):
        """
        Накладывает карту неопределенности на изображение.
        
        Args:
            image (torch.Tensor): Изображение [B, C, H, W]
            uncertainty_map (torch.Tensor): Карта неопределенности [B, 1, H, W]
            alpha (float): Прозрачность наложения (0.0 - 1.0)
            colormap (str): Название цветовой карты
            
        Returns:
            np.ndarray: Изображение с наложенной картой неопределенности [B, H, W, 3] в RGB
        """
        # Преобразуем изображение в numpy RGB
        if image.shape[1] == 1:  # Если это ЧБ изображение
            image_np = image.detach().cpu().repeat(1, 3, 1, 1).numpy()
            image_np = np.transpose(image_np, (0, 2, 3, 1))  # [B, H, W, 3]
        else:  # Если это цветное изображение
            image_np = image.detach().cpu().numpy()
            image_np = np.transpose(image_np, (0, 2, 3, 1))  # [B, H, W, C]
            
            # Если это Lab изображение, преобразуем в RGB
            if image_np.shape[3] == 3 and np.min(image_np) < 0:
                # Это упрощенное преобразование, для точности нужна специальная функция
                image_np = (image_np + 1) / 2  # Нормализуем в [0, 1]
                
        # Нормализуем изображение в [0, 1], если нужно
        if np.max(image_np) > 1.0:
            image_np = image_np / 255.0
            
        # Создаем тепловую карту неопределенности
        heatmap = UncertaintyVisualization.create_uncertainty_heatmap(uncertainty_map, colormap)
        
        # Накладываем тепловую карту на изображение
        overlaid = []
        
        for i in range(image_np.shape[0]):
            # Смешиваем изображение и тепловую карту
            overlay = image_np[i] * (1 - alpha) + heatmap[i] * alpha
            
            # Обрезаем значения до [0, 1]
            overlay = np.clip(overlay, 0, 1)
            
            overlaid.append(overlay)
            
        # Объединяем в один массив [B, H, W, 3]
        return np.stack(overlaid, axis=0)
    
    @staticmethod
    def create_uncertainty_variants(image, samples, uncertainty_map, num_variants=3):
        """
        Создает несколько вариантов колоризации для областей с высокой неопределенностью.
        
        Args:
            image (torch.Tensor): Исходное изображение [B, C, H, W]
            samples (torch.Tensor): Сэмплы колоризации [B, N, C, H, W]
            uncertainty_map (torch.Tensor): Карта неопределенности [B, 1, H, W]
            num_variants (int): Количество вариантов для создания
            
        Returns:
            list: Список вариантов колоризации [B, num_variants, C, H, W]
        """
        # Выбираем наиболее разные сэмплы
        batch_size, num_samples = samples.shape[:2]
        num_variants = min(num_variants, num_samples)
        
        variants = []
        
        for b in range(batch_size):
            # Вычисляем матрицу расстояний между сэмплами
            sample_distances = torch.zeros((num_samples, num_samples), device=samples.device)
            
            for i in range(num_samples):
                for j in range(i + 1, num_samples):
                    # Вычисляем L2 расстояние между сэмплами, взвешенное неопределенностью
                    weighted_diff = (samples[b, i] - samples[b, j]) * uncertainty_map[b].expand_as(samples[b, i])
                    distance = torch.sum(weighted_diff ** 2)
                    sample_distances[i, j] = distance
                    sample_distances[j, i] = distance
                    
            # Выбираем первый сэмпл случайно
            selected_indices = [torch.randint(0, num_samples, (1,)).item()]
            
            # Выбираем остальные сэмплы на основе максимального расстояния
            for _ in range(1, num_variants):
                # Вычисляем минимальное расстояние от каждого сэмпла до уже выбранных
                min_distances = torch.full((num_samples,), float('inf'), device=samples.device)
                
                for idx in selected_indices:
                    min_distances = torch.min(min_distances, sample_distances[:, idx])
                    
                # Маскируем уже выбранные индексы
                for idx in selected_indices:
                    min_distances[idx] = -1
                    
                # Выбираем сэмпл с максимальным минимальным расстоянием
                next_idx = torch.argmax(min_distances).item()
                selected_indices.append(next_idx)
                
            # Собираем выбранные варианты
            batch_variants = [samples[b, idx].unsqueeze(0) for idx in selected_indices]
            variants.append(torch.cat(batch_variants, dim=0))
            
        # Транспонируем результат для соответствия формату [B, num_variants, C, H, W]
        return torch.stack(variants, dim=0)


class UncertaintyEstimationModule:
    """
    Модуль для интеграции оценки неопределенности в колоризатор.
    """
    @staticmethod
    def create_mc_dropout_estimator(colorizer, num_samples=10, dropout_rate=0.1, dropout_layers=None):
        """
        Создает оценщик неопределенности на основе Monte Carlo Dropout.
        
        Args:
            colorizer (nn.Module): Модель колоризации
            num_samples (int): Количество сэмплов для оценки неопределенности
            dropout_rate (float): Вероятность выключения нейронов при dropout
            dropout_layers (list): Список имен слоев, к которым применяется dropout
            
        Returns:
            MCDropoutUncertainty: Оценщик неопределенности
        """
        return MCDropoutUncertainty(
            colorizer=colorizer,
            num_samples=num_samples,
            dropout_rate=dropout_rate,
            dropout_layers=dropout_layers
        )
        
    @staticmethod
    def create_ensemble_estimator(colorizers):
        """
        Создает оценщик неопределенности на основе ансамбля моделей.
        
        Args:
            colorizers (list): Список моделей колоризации
            
        Returns:
            EnsembleUncertainty: Оценщик неопределенности
        """
        return EnsembleUncertainty(colorizers=colorizers)
        
    @staticmethod
    def create_probabilistic_colorizer(colorizer, input_channels=512, base_channels=64):
        """
        Создает вероятностный колоризатор.
        
        Args:
            colorizer (nn.Module): Базовая модель колоризации
            input_channels (int): Количество входных каналов для оценки неопределенности
            base_channels (int): Базовое количество каналов для оценки неопределенности
            
        Returns:
            ProbabilisticColorizer: Вероятностный колоризатор
        """
        return ProbabilisticColorizer(
            colorizer=colorizer,
            input_channels=input_channels,
            base_channels=base_channels
        )
        
    @staticmethod
    def create_uncertainty_guided_colorizer(colorizer, uncertainty_threshold=0.5,
                                           use_memory_bank=True, use_guide_net=True):
        """
        Создает колоризатор с учетом неопределенности.
        
        Args:
            colorizer (nn.Module): Базовая модель колоризации
            uncertainty_threshold (float): Порог неопределенности
            use_memory_bank (bool): Использовать ли банк памяти
            use_guide_net (bool): Использовать ли советника по цветам
            
        Returns:
            UncertaintyGuidedColorizer: Колоризатор с учетом неопределенности
        """
        return UncertaintyGuidedColorizer(
            colorizer=colorizer,
            uncertainty_threshold=uncertainty_threshold,
            use_memory_bank=use_memory_bank,
            use_guide_net=use_guide_net
        )


# Создаем функцию для создания оценщика неопределенности
def create_uncertainty_estimator(colorizer=None, config=None):
    """
    Создает оценщик неопределенности на основе конфигурации.
    
    Args:
        colorizer (nn.Module, optional): Базовая модель колоризации
        config (dict, optional): Конфигурация оценщика неопределенности
        
    Returns:
        nn.Module: Оценщик неопределенности
    """
    if config is None:
        config = {}
        
    # Параметры по умолчанию
    method = config.get('method', 'mc_dropout')
    num_samples = config.get('num_samples', 10)
    dropout_rate = config.get('dropout_rate', 0.1)
    dropout_layers = config.get('dropout_layers', None)
    uncertainty_threshold = config.get('uncertainty_threshold', 0.5)
    use_memory_bank = config.get('use_memory_bank', True)
    use_guide_net = config.get('use_guide_net', True)
    
    if method == 'mc_dropout':
        return UncertaintyEstimationModule.create_mc_dropout_estimator(
            colorizer=colorizer,
            num_samples=num_samples,
            dropout_rate=dropout_rate,
            dropout_layers=dropout_layers
        )
    elif method == 'probabilistic':
        return UncertaintyEstimationModule.create_probabilistic_colorizer(
            colorizer=colorizer
        )
    elif method == 'guided':
        return UncertaintyEstimationModule.create_uncertainty_guided_colorizer(
            colorizer=colorizer,
            uncertainty_threshold=uncertainty_threshold,
            use_memory_bank=use_memory_bank,
            use_guide_net=use_guide_net
        )
    else:
        raise ValueError(f"Неизвестный метод оценки неопределенности: {method}")


if __name__ == "__main__":
    # Пример использования оценщика неопределенности
    
    # Создаем простую модель колоризации для демонстрации
    class SimpleColorizer(nn.Module):
        def __init__(self):
            super(SimpleColorizer, self).__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            self.decoder = nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 2, kernel_size=3, padding=1)
            )
            
        def forward(self, x):
            features = self.encoder(x)
            output = self.decoder(features)
            return {'colorized': torch.cat([x, output], dim=1), 'features': features}
    
    # Создаем модель колоризации
    colorizer = SimpleColorizer()
    
    # Создаем оценщик неопределенности
    uncertainty_estimator = create_uncertainty_estimator(
        colorizer=colorizer,
        config={'method': 'mc_dropout', 'num_samples': 5}
    )
    
    # Создаем тестовое изображение
    batch_size = 2
    x = torch.randn(batch_size, 1, 64, 64)
    
    # Оцениваем неопределенность
    result = uncertainty_estimator(x)
    
    # Выводим информацию о результатах
    print("Результаты оценки неопределенности:")
    for key, value in result.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
            print(f"  {key}: список из {len(value)} тензоров с формой {value[0].shape}")
        else:
            print(f"  {key}: {type(value)}")