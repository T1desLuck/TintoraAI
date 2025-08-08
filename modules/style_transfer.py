"""
Style Transfer Component: Компонент для переноса стиля колоризации между изображениями.

Данный модуль реализует механизмы переноса цветовых стилей между изображениями,
позволяя адаптировать колоризацию под определенные художественные, исторические
или жанровые стили. Компонент позволяет извлекать цветовые характеристики из 
референсных изображений и применять их к целевым изображениям с сохранением 
их структуры и содержания.

Ключевые особенности:
- AdaIN (Adaptive Instance Normalization) для переноса цветовых статистик
- Color Transformation Network для генерации цветовых преобразований
- Механизмы сохранения локальной цветовой когерентности
- Контрольные точки для управления степенью стилизации
- Экстракция и применение цветовых палитр различных эпох и жанров

Преимущества для колоризации:
- Возможность создания исторически-достоверных цветов (1920-е, 1950-е и т.д.)
- Поддержка художественных стилей (импрессионизм, кино-нуар и т.п.)
- Согласованная цветовая гамма по всему изображению
- Гибкое управление интенсивностью стилизации
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Union, Optional


class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization для переноса статистик стиля.
    
    Этот метод выравнивает статистики активаций контента со статистиками стиля,
    эффективно перенося стилистические характеристики при сохранении содержания.
    """
    def __init__(self, epsilon=1e-5):
        super(AdaIN, self).__init__()
        self.epsilon = epsilon
        
    def calc_mean_std(self, features):
        """
        Вычисляет среднее значение и стандартное отклонение по пространственным измерениям.
        
        Args:
            features (torch.Tensor): Входной тензор признаков [B, C, H, W]
            
        Returns:
            tuple: (среднее, стандартное отклонение)
        """
        # Размерность тензора [B, C, H, W]
        batch_size, channels = features.shape[:2]
        
        # Изменяем форму для вычисления статистик по H и W измерениям
        features_view = features.view(batch_size, channels, -1)
        
        # Вычисляем среднее для каждого канала
        mean = torch.mean(features_view, dim=2).view(batch_size, channels, 1, 1)
        
        # Вычисляем стандартное отклонение
        var = torch.var(features_view, dim=2, unbiased=False).view(batch_size, channels, 1, 1)
        std = torch.sqrt(var + self.epsilon)
        
        return mean, std
    
    def forward(self, content_features, style_features):
        """
        Применяет адаптивную нормализацию экземпляров.
        
        Args:
            content_features (torch.Tensor): Признаки контентного изображения [B, C, H, W]
            style_features (torch.Tensor): Признаки стилевого изображения [B, C, H, W]
            
        Returns:
            torch.Tensor: Нормализованные признаки с перенесенным стилем [B, C, H, W]
        """
        # Вычисляем статистики контента
        content_mean, content_std = self.calc_mean_std(content_features)
        
        # Вычисляем статистики стиля
        style_mean, style_std = self.calc_mean_std(style_features)
        
        # Нормализуем контентные признаки и масштабируем их по статистикам стиля
        normalized_features = (content_features - content_mean) / content_std
        output_features = normalized_features * style_std + style_mean
        
        return output_features


class ColorTransformNetwork(nn.Module):
    """
    Сеть для генерации цветовых преобразований на основе стиля.
    
    Извлекает стилевые характеристики из референсного изображения и
    генерирует преобразования для применения к контентному изображению.
    
    Args:
        input_channels (int): Количество входных каналов
        hidden_dim (int): Размерность скрытого пространства
        output_channels (int): Количество выходных каналов (обычно 2 для ab каналов в Lab)
    """
    def __init__(self, input_channels=3, hidden_dim=64, output_channels=2):
        super(ColorTransformNetwork, self).__init__()
        
        # Экстрактор стилевых признаков
        self.style_encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Генератор параметров преобразования
        self.transform_generator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, output_channels * 6)  # 6 параметров на канал (a, b, c, d, e, f)
        )
        
        # Энкодер контентного изображения
        self.content_encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True)
        )
        
        # Декодер для генерации выходного изображения
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, output_channels, kernel_size=1)
        )
        
        # AdaIN слой для переноса стиля
        self.adain = AdaIN()
    
    def forward(self, content_image, style_image):
        """
        Генерирует цветовую трансформацию и применяет её.
        
        Args:
            content_image (torch.Tensor): Контентное изображение [B, C, H, W]
            style_image (torch.Tensor): Стилевое изображение [B, C, H, W]
            
        Returns:
            dict: {
                'transformed_image': torch.Tensor,  # Преобразованное изображение [B, output_channels, H, W]
                'transform_params': torch.Tensor,  # Параметры преобразования
                'content_features': torch.Tensor,  # Признаки контента
                'style_features': torch.Tensor     # Признаки стиля
            }
        """
        # Извлекаем стилевые признаки
        style_features = self.style_encoder(style_image)
        
        # Генерируем параметры преобразования
        transform_params = self.transform_generator(style_features)
        
        # Кодируем контентное изображение
        content_features = self.content_encoder(content_image)
        
        # Применяем перенос стиля с помощью AdaIN
        # Для этого нам нужно преобразовать признаки стиля к правильной размерности
        style_features_expanded = style_features.expand(-1, -1, content_features.size(2), content_features.size(3))
        stylized_features = self.adain(content_features, style_features_expanded)
        
        # Декодируем признаки для получения выходного изображения
        output_image = self.decoder(stylized_features)
        
        return {
            'transformed_image': output_image,
            'transform_params': transform_params,
            'content_features': content_features,
            'style_features': style_features
        }
        
    def apply_color_transform(self, image, transform_params):
        """
        Применяет параметризованное цветовое преобразование к изображению.
        
        Args:
            image (torch.Tensor): Входное изображение [B, C, H, W]
            transform_params (torch.Tensor): Параметры преобразования [B, output_channels * 6]
            
        Returns:
            torch.Tensor: Преобразованное изображение [B, output_channels, H, W]
        """
        batch_size, channels, height, width = image.shape
        output_channels = transform_params.shape[1] // 6
        
        # Преобразуем параметры в нужную форму
        params = transform_params.view(batch_size, output_channels, 6)
        
        # Извлекаем отдельные параметры преобразования
        a = params[:, :, 0].view(batch_size, output_channels, 1, 1)
        b = params[:, :, 1].view(batch_size, output_channels, 1, 1)
        c = params[:, :, 2].view(batch_size, output_channels, 1, 1)
        d = params[:, :, 3].view(batch_size, output_channels, 1, 1)
        e = params[:, :, 4].view(batch_size, output_channels, 1, 1)
        f = params[:, :, 5].view(batch_size, output_channels, 1, 1)
        
        # Кодируем входное изображение
        content_features = self.content_encoder(image)
        
        # Применяем аффинное преобразование: f(x) = ax^2 + bx + c + d*sin(ex + f)
        # Для нелинейного цветового преобразования
        input_expanded = content_features.mean(dim=1, keepdim=True).expand(-1, output_channels, -1, -1)
        
        transformed = a * input_expanded**2 + b * input_expanded + c + d * torch.sin(e * input_expanded + f)
        
        # Декодируем с учетом преобразования
        output_image = self.decoder(transformed)
        
        return output_image


class StyleEncoder(nn.Module):
    """
    Энкодер для извлечения цветовых характеристик стиля из изображения.
    
    Args:
        input_channels (int): Количество входных каналов
        style_dim (int): Размерность стилевого вектора
        use_attention (bool): Использовать ли механизм внимания
    """
    def __init__(self, input_channels=3, style_dim=512, use_attention=True):
        super(StyleEncoder, self).__init__()
        
        # Базовый энкодер
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Внимание для выделения важных областей с цветом (если используется)
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.Sequential(
                nn.Conv2d(512, 1, kernel_size=1),
                nn.Sigmoid()
            )
            
        # Проекция в пространство стилевых векторов
        self.style_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, style_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Экстрактор цветовой палитры
        self.palette_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),  # Сжимаем до 4x4 = 16 цветов
            nn.Conv2d(512, 3, kernel_size=1),  # Проекция в RGB пространство
        )
        
    def forward(self, x):
        """
        Извлекает стилевые характеристики из изображения.
        
        Args:
            x (torch.Tensor): Входное изображение [B, C, H, W]
            
        Returns:
            dict: {
                'style_vector': torch.Tensor,  # Стилевой вектор [B, style_dim]
                'attention_map': torch.Tensor,  # Карта внимания [B, 1, H/8, W/8] или None
                'color_palette': torch.Tensor,  # Цветовая палитра [B, 3, 4, 4]
                'features': torch.Tensor       # Промежуточные признаки [B, 512, H/8, W/8]
            }
        """
        # Извлекаем признаки
        features = self.encoder(x)
        
        # Вычисляем карту внимания, если нужно
        attention_map = None
        if self.use_attention:
            attention_map = self.attention(features)
            # Применяем внимание к признакам
            attended_features = features * attention_map
        else:
            attended_features = features
            
        # Проецируем в стилевой вектор
        style_vector = self.style_projector(attended_features)
        
        # Извлекаем цветовую палитру
        color_palette = self.palette_extractor(attended_features)
        
        return {
            'style_vector': style_vector,
            'attention_map': attention_map,
            'color_palette': color_palette,
            'features': features
        }


class StyleModulator(nn.Module):
    """
    Модулятор для внедрения стилевой информации в процесс колоризации.
    
    Args:
        feature_dim (int): Размерность признаков
        style_dim (int): Размерность стилевого вектора
        use_spatial_modulation (bool): Применять ли пространственную модуляцию
    """
    def __init__(self, feature_dim=256, style_dim=512, use_spatial_modulation=True):
        super(StyleModulator, self).__init__()
        
        # Проекция стиля в параметры модуляции
        self.style_to_params = nn.Sequential(
            nn.Linear(style_dim, feature_dim * 2),  # scale и bias
            nn.LeakyReLU(0.2),
            nn.Linear(feature_dim * 2, feature_dim * 2)
        )
        
        # Пространственная модуляция (если используется)
        self.use_spatial_modulation = use_spatial_modulation
        if use_spatial_modulation:
            self.spatial_modulator = nn.Sequential(
                nn.Linear(style_dim, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 16*16),  # Разрешение для пространственной модуляции
                nn.Sigmoid()
            )
            
    def forward(self, features, style_vector):
        """
        Модулирует признаки на основе стилевого вектора.
        
        Args:
            features (torch.Tensor): Входные признаки [B, C, H, W]
            style_vector (torch.Tensor): Стилевой вектор [B, style_dim]
            
        Returns:
            torch.Tensor: Модулированные признаки [B, C, H, W]
        """
        batch_size, channels, height, width = features.shape
        
        # Генерируем параметры модуляции
        modulation_params = self.style_to_params(style_vector)
        scale, bias = modulation_params.chunk(2, dim=1)
        
        # Изменяем форму для удобства применения
        scale = scale.view(batch_size, channels, 1, 1)
        bias = bias.view(batch_size, channels, 1, 1)
        
        # Применяем базовую модуляцию
        modulated_features = features * scale + bias
        
        # Применяем пространственную модуляцию, если нужно
        if self.use_spatial_modulation:
            spatial_weights = self.spatial_modulator(style_vector)
            spatial_weights = spatial_weights.view(batch_size, 1, 16, 16)
            
            # Изменяем размер для соответствия входным признакам
            spatial_weights = F.interpolate(spatial_weights, size=(height, width), mode='bilinear')
            
            # Применяем пространственную модуляцию
            modulated_features = modulated_features * spatial_weights
            
        return modulated_features


class ColorHistogramLoss(nn.Module):
    """
    Функция потери для сопоставления цветовых гистограмм.
    
    Args:
        num_bins (int): Количество бинов в гистограмме
        sigma (float): Сигма для мягкой квантизации
    """
    def __init__(self, num_bins=64, sigma=0.02):
        super(ColorHistogramLoss, self).__init__()
        self.num_bins = num_bins
        self.sigma = sigma
        
        # Центры бинов для мягкой квантизации
        bin_centers = torch.linspace(0, 1, steps=num_bins)
        self.register_buffer('bin_centers', bin_centers.view(1, 1, -1))
        
    def soft_histogram(self, x):
        """
        Вычисляет мягкую гистограмму для тензора.
        
        Args:
            x (torch.Tensor): Входной тензор [B, C, H, W] или [B, H*W, C]
            
        Returns:
            torch.Tensor: Гистограмма [B, C, num_bins]
        """
        if x.ndim == 4:
            # [B, C, H, W] -> [B, C, H*W]
            x = x.flatten(2)
            
        # Нормализуем значения в диапазон [0, 1]
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
        
        # Изменяем размерность для вычисления расстояний до центров бинов
        # [B, C, H*W] -> [B, C, H*W, 1]
        x_norm = x_norm.unsqueeze(-1)
        
        # Вычисляем веса для мягкой квантизации
        weights = torch.exp(-0.5 * ((x_norm - self.bin_centers) / self.sigma)**2)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Суммируем веса для получения гистограммы
        histogram = weights.sum(dim=2)
        
        # Нормализуем гистограмму
        histogram = histogram / (histogram.sum(dim=-1, keepdim=True) + 1e-8)
        
        return histogram
    
    def forward(self, x, target):
        """
        Вычисляет потерю между гистограммами входного и целевого тензоров.
        
        Args:
            x (torch.Tensor): Входной тензор [B, C, H, W]
            target (torch.Tensor): Целевой тензор [B, C, H, W]
            
        Returns:
            torch.Tensor: Потеря между гистограммами
        """
        # Вычисляем гистограммы
        x_hist = self.soft_histogram(x)
        target_hist = self.soft_histogram(target)
        
        # Вычисляем Earth Mover's Distance между гистограммами
        # Для простоты используем L1 расстояние как аппроксимацию EMD
        # В реальной реализации можно использовать более точный алгоритм
        loss = F.l1_loss(x_hist, target_hist)
        
        return loss


class StyleTransferModule(nn.Module):
    """
    Главный модуль для переноса стиля в задаче колоризации.
    
    Args:
        input_channels (int): Количество входных каналов
        output_channels (int): Количество выходных каналов (обычно 2 для ab каналов в Lab)
        style_dim (int): Размерность стилевого вектора
        use_attention (bool): Использовать ли механизм внимания
        use_histogram_loss (bool): Использовать ли потерю на гистограммах
    """
    def __init__(self, input_channels=3, output_channels=2, style_dim=512, 
                 use_attention=True, use_histogram_loss=True):
        super(StyleTransferModule, self).__init__()
        
        # Энкодер для извлечения стилевых характеристик
        self.style_encoder = StyleEncoder(
            input_channels=input_channels, 
            style_dim=style_dim,
            use_attention=use_attention
        )
        
        # Сеть для генерации цветовых преобразований
        self.color_transform_net = ColorTransformNetwork(
            input_channels=input_channels,
            hidden_dim=64,
            output_channels=output_channels
        )
        
        # Модулятор стиля для внедрения в процесс колоризации
        self.style_modulator = StyleModulator(
            feature_dim=256,
            style_dim=style_dim,
            use_spatial_modulation=True
        )
        
        # Потеря на гистограммах (если используется)
        self.use_histogram_loss = use_histogram_loss
        if use_histogram_loss:
            self.histogram_loss = ColorHistogramLoss(num_bins=64, sigma=0.02)
        
        # Для хранения словаря стилей
        self.style_dictionary = {}
        
        # Для отслеживания качества переноса стиля
        self.register_buffer('style_transfer_quality', torch.zeros(1))
        self.register_buffer('total_transfers', torch.zeros(1, dtype=torch.long))
        
    def encode_style(self, style_image):
        """
        Извлекает стилевые характеристики из изображения.
        
        Args:
            style_image (torch.Tensor): Стилевое изображение [B, C, H, W]
            
        Returns:
            dict: Результат работы style_encoder
        """
        return self.style_encoder(style_image)
    
    def add_style_to_dictionary(self, name, style_image):
        """
        Добавляет стиль в словарь стилей.
        
        Args:
            name (str): Название стиля
            style_image (torch.Tensor): Стилевое изображение [B, C, H, W]
            
        Returns:
            dict: Закодированный стиль
        """
        with torch.no_grad():
            style_data = self.encode_style(style_image)
            self.style_dictionary[name] = {
                'style_vector': style_data['style_vector'].cpu(),
                'color_palette': style_data['color_palette'].cpu()
            }
        return style_data
        
    def get_style_from_dictionary(self, name):
        """
        Получает стиль из словаря стилей.
        
        Args:
            name (str): Название стиля
            
        Returns:
            dict: Стилевые данные или None, если стиль не найден
        """
        if name in self.style_dictionary:
            style_data = self.style_dictionary[name]
            return {
                'style_vector': style_data['style_vector'].to(self.device),
                'color_palette': style_data['color_palette'].to(self.device)
            }
        return None
    
    def apply_style_transfer(self, content_image, style_reference=None, style_name=None, alpha=1.0):
        """
        Применяет перенос стиля к контентному изображению.
        
        Args:
            content_image (torch.Tensor): Контентное изображение [B, C, H, W]
            style_reference (torch.Tensor, optional): Референсное стилевое изображение [B, C, H, W]
            style_name (str, optional): Название стиля из словаря
            alpha (float): Интенсивность переноса стиля (0.0 - 1.0)
            
        Returns:
            dict: {
                'stylized_image': torch.Tensor,  # Стилизованное изображение [B, output_channels, H, W]
                'style_vector': torch.Tensor,  # Стилевой вектор [B, style_dim]
                'color_palette': torch.Tensor,  # Цветовая палитра [B, 3, 4, 4]
                'attention_map': torch.Tensor,  # Карта внимания [B, 1, H/8, W/8] или None
            }
        """
        # Получаем стилевые данные
        style_data = None
        
        if style_reference is not None:
            # Извлекаем стиль из референсного изображения
            style_data = self.encode_style(style_reference)
        elif style_name is not None and style_name in self.style_dictionary:
            # Получаем стиль из словаря
            style_data = self.get_style_from_dictionary(style_name)
        else:
            # Если нет ни референса, ни названия стиля, возвращаем контентное изображение
            return {
                'stylized_image': content_image,
                'style_vector': None,
                'color_palette': None,
                'attention_map': None
            }
            
        # Применяем цветовое преобразование
        transform_result = self.color_transform_net(content_image, style_reference)
        transformed_image = transform_result['transformed_image']
        
        # Смешиваем оригинальное и трансформированное изображения в зависимости от alpha
        stylized_image = alpha * transformed_image + (1 - alpha) * content_image
        
        # Увеличиваем счетчик переносов стиля
        self.total_transfers += content_image.size(0)
        
        # Возвращаем результат
        return {
            'stylized_image': stylized_image,
            'style_vector': style_data['style_vector'],
            'color_palette': style_data['color_palette'],
            'attention_map': style_data.get('attention_map', None)
        }
    
    def modulate_features(self, features, style_vector):
        """
        Модулирует признаки на основе стилевого вектора.
        
        Args:
            features (torch.Tensor): Признаки [B, C, H, W]
            style_vector (torch.Tensor): Стилевой вектор [B, style_dim]
            
        Returns:
            torch.Tensor: Модулированные признаки [B, C, H, W]
        """
        return self.style_modulator(features, style_vector)
    
    def compute_histogram_loss(self, output_image, target_style):
        """
        Вычисляет потерю на гистограммах, если включено.
        
        Args:
            output_image (torch.Tensor): Выходное изображение [B, C, H, W]
            target_style (torch.Tensor): Целевое стилевое изображение [B, C, H, W]
            
        Returns:
            torch.Tensor: Потеря на гистограммах или 0, если не используется
        """
        if self.use_histogram_loss:
            return self.histogram_loss(output_image, target_style)
        return torch.tensor(0.0, device=output_image.device)
    
    def update_transfer_quality(self, quality_score):
        """
        Обновляет метрику качества переноса стиля.
        
        Args:
            quality_score (float): Оценка качества переноса стиля (0.0 - 1.0)
        """
        # Скользящее среднее для метрики качества
        old_quality = self.style_transfer_quality.item()
        new_quality = old_quality * 0.9 + quality_score * 0.1
        self.style_transfer_quality.fill_(new_quality)
        
    def get_stats(self):
        """
        Возвращает статистики модуля.
        
        Returns:
            dict: Статистики модуля
        """
        return {
            'style_transfer_quality': self.style_transfer_quality.item(),
            'total_transfers': self.total_transfers.item(),
            'num_styles_in_dictionary': len(self.style_dictionary)
        }


# Функция для создания модуля переноса стиля
def create_style_transfer_module(config=None):
    """
    Создает модуль переноса стиля на основе конфигурации.
    
    Args:
        config (dict, optional): Словарь с параметрами конфигурации
        
    Returns:
        StyleTransferModule: Модуль переноса стиля
    """
    # Параметры по умолчанию
    default_config = {
        'input_channels': 3,
        'output_channels': 2,
        'style_dim': 512,
        'use_attention': True,
        'use_histogram_loss': True
    }
    
    # Объединяем с пользовательской конфигурацией
    if config is not None:
        for key, value in config.items():
            if key in default_config:
                default_config[key] = value
    
    # Создаем модель с указанными параметрами
    model = StyleTransferModule(
        input_channels=default_config['input_channels'],
        output_channels=default_config['output_channels'],
        style_dim=default_config['style_dim'],
        use_attention=default_config['use_attention'],
        use_histogram_loss=default_config['use_histogram_loss']
    )
    
    return model


if __name__ == "__main__":
    # Пример использования модуля переноса стиля
    
    # Создаем модуль
    style_transfer = create_style_transfer_module()
    
    # Создаем тестовые данные
    batch_size = 2
    content_image = torch.randn(batch_size, 3, 256, 256)
    style_reference = torch.randn(batch_size, 3, 256, 256)
    
    # Применяем перенос стиля
    result = style_transfer.apply_style_transfer(content_image, style_reference, alpha=0.8)
    
    # Выводим информацию о результатах
    for key, value in result.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {type(value)}")
    
    # Пример добавления стиля в словарь
    style_transfer.add_style_to_dictionary("vintage", style_reference)
    print(f"Styles in dictionary: {len(style_transfer.style_dictionary)}")
    
    # Пример применения стиля из словаря
    result = style_transfer.apply_style_transfer(content_image, style_name="vintage", alpha=1.0)


# Alias for backward compatibility
StyleTransfer = StyleTransferModule