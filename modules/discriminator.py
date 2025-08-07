"""
GAN Discriminator: Критик для оценки реалистичности колоризации с системой наград.

Данный модуль реализует мощный дискриминатор для оценки реалистичности и качества
колоризации изображений. Дискриминатор не только выполняет классическую GAN-роль 
отличия реальных изображений от генерированных, но и содержит систему "наград и наказаний"
для обратной связи с генератором, помогая ему улучшать качество колоризации.

Ключевые особенности:
- Многоуровневая архитектура для оценки глобальных и локальных особенностей
- PatchGAN для детального анализа текстур и цветовых переходов
- Семантическая сегментация для проверки цветовой корректности объектов
- Система "наград и наказаний" для обучения генератора
- Интеграция с механизмом внимания для фокусировки на важных областях

Преимущества для колоризации:
- Обеспечивает реалистичные и естественные цвета
- Предотвращает появление артефактов и цветовых искажений
- "Мотивирует" генератор выходить за рамки безопасных, но серых решений
- Улучшает детализацию в сложных для колоризации областях
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import math


class ConvBlock(nn.Module):
    """
    Базовый сверточный блок с нормализацией и активацией.
    
    Args:
        in_channels (int): Количество входных каналов
        out_channels (int): Количество выходных каналов
        kernel_size (int): Размер ядра свертки
        stride (int): Шаг свертки
        padding (int): Отступы
        use_bias (bool): Использовать ли смещение
        norm_type (str): Тип нормализации ('batch', 'instance', 'none')
        activation (str): Тип активации ('relu', 'lrelu', 'none')
        lrelu_slope (float): Наклон для LeakyReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1,
                 use_bias=False, norm_type='batch', activation='lrelu', lrelu_slope=0.2):
        super(ConvBlock, self).__init__()
        
        # Слой свертки
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=use_bias
        )
        
        # Слой нормализации
        if norm_type == 'batch':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels)
        else:  # 'none'
            self.norm = nn.Identity()
            
        # Активационная функция
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(lrelu_slope, inplace=True)
        else:  # 'none'
            self.activation = nn.Identity()
            
    def forward(self, x):
        """
        Прямое распространение через блок.
        
        Args:
            x (torch.Tensor): Входной тензор [B, C, H, W]
            
        Returns:
            torch.Tensor: Выходной тензор [B, out_channels, H/stride, W/stride]
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class SelfAttention(nn.Module):
    """
    Модуль самовнимания для карт признаков.
    
    Args:
        channels (int): Количество каналов во входных картах признаков
        reduction (int): Коэффициент сжатия для уменьшения вычислительной сложности
    """
    def __init__(self, channels, reduction=8):
        super(SelfAttention, self).__init__()
        
        # Проекции для запроса, ключа и значения
        self.query_conv = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Гамма-параметр для взвешивания входа и внимания
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        """
        Прямое распространение через модуль самовнимания.
        
        Args:
            x (torch.Tensor): Входной тензор [B, C, H, W]
            
        Returns:
            tuple: (Выходной тензор с примененным вниманием [B, C, H, W], 
                   Карта внимания [B, H*W, H*W])
        """
        batch_size, channels, height, width = x.size()
        
        # Проецируем входные данные
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # [B, H*W, C//r]
        key = self.key_conv(x).view(batch_size, -1, height * width)  # [B, C//r, H*W]
        value = self.value_conv(x).view(batch_size, -1, height * width)  # [B, C, H*W]
        
        # Вычисляем матрицу внимания
        attention = self.softmax(torch.bmm(query, key))  # [B, H*W, H*W]
        
        # Применяем внимание к значениям
        out = torch.bmm(value, attention.permute(0, 2, 1))  # [B, C, H*W]
        out = out.view(batch_size, channels, height, width)  # [B, C, H, W]
        
        # Взвешиваем выход с помощью гамма-параметра
        out = self.gamma * out + x
        
        return out, attention


class SpectralNormConv2d(nn.Module):
    """
    Свёрточный слой со спектральной нормализацией для стабильности обучения GAN.
    
    Args:
        in_channels (int): Количество входных каналов
        out_channels (int): Количество выходных каналов
        kernel_size (int): Размер ядра свертки
        stride (int): Шаг свертки
        padding (int): Отступы
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SpectralNormConv2d, self).__init__()
        
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
        )
        
    def forward(self, x):
        """
        Прямое распространение через нормализованный слой.
        
        Args:
            x (torch.Tensor): Входной тензор [B, in_channels, H, W]
            
        Returns:
            torch.Tensor: Выходной тензор [B, out_channels, H/stride, W/stride]
        """
        return self.conv(x)


class PatchDiscriminator(nn.Module):
    """
    Дискриминатор типа PatchGAN, который классифицирует каждый N×N патч изображения
    как реальный или поддельный.
    
    Args:
        input_channels (int): Количество входных каналов
        ndf (int): Базовое количество признаков в дискриминаторе
        n_layers (int): Количество понижающих разрешение слоев
        use_attention (bool): Использовать ли слои внимания
        use_spectral_norm (bool): Использовать ли спектральную нормализацию
    """
    def __init__(self, input_channels=3, ndf=64, n_layers=3, 
                 use_attention=True, use_spectral_norm=True):
        super(PatchDiscriminator, self).__init__()
        
        # Выбираем функцию свертки на основе использования спектральной нормализации
        conv_func = SpectralNormConv2d if use_spectral_norm else nn.Conv2d
        
        # Начальный слой свертки
        sequence = [
            conv_func(input_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        
        # Промежуточные слои
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)  # Максимум 8 * ndf
            
            sequence += [
                conv_func(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
            
            # Добавляем слой внимания в середине сети
            if use_attention and n == n_layers // 2:
                sequence.append(SelfAttention(ndf * nf_mult))
        
        # Увеличиваем количество признаков для последнего слоя
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        
        sequence += [
            conv_func(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        # Финальный слой для классификации патчей
        sequence += [
            conv_func(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]
        
        # Создаем последовательность из всех слоев
        self.model = nn.Sequential(*sequence)
        
    def forward(self, x):
        """
        Прямое распространение через дискриминатор.
        
        Args:
            x (torch.Tensor): Входное изображение [B, input_channels, H, W]
            
        Returns:
            torch.Tensor: Выходные скоры [B, 1, H', W']
        """
        return self.model(x)


class MultiscaleDiscriminator(nn.Module):
    """
    Мультимасштабный дискриминатор, состоящий из нескольких подсетей,
    которые работают на разных масштабах изображения.
    
    Args:
        input_channels (int): Количество входных каналов
        num_discriminators (int): Количество дискриминаторов на разных масштабах
        ndf (int): Базовое количество признаков в дискриминаторе
        n_layers (int): Количество понижающих разрешение слоев в каждом дискриминаторе
        use_attention (bool): Использовать ли слои внимания
        use_spectral_norm (bool): Использовать ли спектральную нормализацию
    """
    def __init__(self, input_channels=3, num_discriminators=3, ndf=64, n_layers=3,
                 use_attention=True, use_spectral_norm=True):
        super(MultiscaleDiscriminator, self).__init__()
        
        self.num_discriminators = num_discriminators
        self.n_layers = n_layers
        
        # Создаем дискриминаторы для разных масштабов
        self.discriminators = nn.ModuleList()
        for i in range(num_discriminators):
            # Каждый дискриминатор - это отдельный PatchDiscriminator
            self.discriminators.append(
                PatchDiscriminator(
                    input_channels=input_channels,
                    ndf=ndf,
                    n_layers=n_layers,
                    use_attention=use_attention,
                    use_spectral_norm=use_spectral_norm
                )
            )
            
        # Слои для понижения размерности изображения между дискриминаторами
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        """
        Прямое распространение через мультимасштабный дискриминатор.
        
        Args:
            x (torch.Tensor): Входное изображение [B, input_channels, H, W]
            
        Returns:
            list: Список выходных тензоров от каждого дискриминатора
        """
        results = []
        
        # Проходим через каждый дискриминатор
        input_image = x
        for i, discriminator in enumerate(self.discriminators):
            results.append(discriminator(input_image))
            
            # Уменьшаем размер изображения для следующего дискриминатора
            if i < self.num_discriminators - 1:
                input_image = self.downsample(input_image)
                
        return results


class SemanticDiscriminatorBlock(nn.Module):
    """
    Блок дискриминатора с семантической информацией для оценки реалистичности
    цветов объектов на основе их типа.
    
    Args:
        input_channels (int): Количество входных каналов
        semantic_channels (int): Количество семантических классов
        ndf (int): Базовое количество признаков
        use_spectral_norm (bool): Использовать ли спектральную нормализацию
    """
    def __init__(self, input_channels=3, semantic_channels=150, ndf=64, use_spectral_norm=True):
        super(SemanticDiscriminatorBlock, self).__init__()
        
        # Функция свертки в зависимости от использования спектральной нормализации
        conv_func = SpectralNormConv2d if use_spectral_norm else nn.Conv2d
        
        # Слой для обработки входного изображения
        self.img_encoder = nn.Sequential(
            conv_func(input_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            conv_func(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True)
        )
        
        # Слой для обработки семантической информации
        self.sem_encoder = nn.Sequential(
            conv_func(semantic_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            conv_func(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True)
        )
        
        # Слой для комбинирования изображения и семантики
        self.combined_encoder = nn.Sequential(
            conv_func(ndf * 4, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),
            conv_func(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True),
            conv_func(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
        )
        
    def forward(self, x, semantics):
        """
        Прямое распространение через блок.
        
        Args:
            x (torch.Tensor): Входное изображение [B, input_channels, H, W]
            semantics (torch.Tensor): Семантическая информация [B, semantic_channels, H, W]
            
        Returns:
            torch.Tensor: Выходные скоры [B, 1, H', W']
        """
        # Кодируем изображение
        img_feats = self.img_encoder(x)
        
        # Размер семантической информации должен соответствовать размеру изображения
        if semantics.size()[2:] != x.size()[2:]:
            semantics = F.interpolate(semantics, size=x.size()[2:], mode='bilinear', align_corners=False)
            
        # Кодируем семантику
        sem_feats = self.sem_encoder(semantics)
        
        # Объединяем признаки
        combined = torch.cat([img_feats, sem_feats], dim=1)
        
        # Применяем финальный энкодер
        output = self.combined_encoder(combined)
        
        return output


class RewardGenerator(nn.Module):
    """
    Генератор наград для системы обучения с подкреплением.
    
    Args:
        input_channels (int): Количество входных каналов
        ndf (int): Базовое количество признаков
        use_spectral_norm (bool): Использовать ли спектральную нормализацию
    """
    def __init__(self, input_channels=3, ndf=64, use_spectral_norm=True):
        super(RewardGenerator, self).__init__()
        
        # Функция свертки в зависимости от использования спектральной нормализации
        conv_func = SpectralNormConv2d if use_spectral_norm else nn.Conv2d
        
        # Энкодер для извлечения признаков из изображения
        self.encoder = nn.Sequential(
            conv_func(input_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            conv_func(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            conv_func(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),
            conv_func(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True)
        )
        
        # Генератор наград на основе извлеченных признаков
        self.reward_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ndf * 8, ndf * 4),
            nn.ReLU(True),
            nn.Linear(ndf * 4, 1),
            nn.Sigmoid()
        )
        
        # Сложность колоризации на основе содержания изображения
        self.complexity_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ndf * 8, ndf * 4),
            nn.ReLU(True),
            nn.Linear(ndf * 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, real_img, fake_img):
        """
        Прямое распространение для генерации наград.
        
        Args:
            real_img (torch.Tensor): Реальное изображение [B, input_channels, H, W]
            fake_img (torch.Tensor): Сгенерированное изображение [B, input_channels, H, W]
            
        Returns:
            dict: {
                'reward': torch.Tensor,  # Награда [B, 1]
                'complexity': torch.Tensor,  # Сложность колоризации [B, 1]
                'quality': torch.Tensor  # Качество относительно сложности [B, 1]
            }
        """
        # Извлекаем признаки из реальных и поддельных изображений
        real_features = self.encoder(real_img)
        fake_features = self.encoder(fake_img)
        
        # Вычисляем сложность колоризации на основе реального изображения
        complexity = self.complexity_head(real_features)
        
        # Вычисляем базовую награду на основе сгенерированного изображения
        base_reward = self.reward_head(fake_features)
        
        # Регулируем награду в зависимости от сложности
        # Более высокая награда за хорошую колоризацию сложных изображений
        adjusted_reward = base_reward * (1.0 + complexity)
        
        # Качество относительно сложности
        quality = base_reward / (complexity + 0.1)  # Избегаем деления на очень маленькие значения
        
        return {
            'reward': adjusted_reward,
            'complexity': complexity,
            'quality': quality
        }


class ColorDiscriminator(nn.Module):
    """
    Специализированный дискриминатор для оценки реалистичности цветов.
    
    Args:
        input_channels (int): Количество входных каналов
        ndf (int): Базовое количество признаков
        use_spectral_norm (bool): Использовать ли спектральную нормализацию
    """
    def __init__(self, input_channels=3, ndf=64, use_spectral_norm=True):
        super(ColorDiscriminator, self).__init__()
        
        # Функция свертки в зависимости от использования спектральной нормализации
        conv_func = SpectralNormConv2d if use_spectral_norm else nn.Conv2d
        
        # Энкодер для изображений с двумя потоками: основным и цветовым
        self.main_encoder = nn.Sequential(
            conv_func(input_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            conv_func(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True)
        )
        
        # Отдельный поток для цветовой информации
        self.color_encoder = nn.Sequential(
            # Выделяем только цветовые каналы (a и b в Lab пространстве или цветовые каналы в RGB)
            nn.Conv2d(2, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            conv_func(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True)
        )
        
        # Объединенный энкодер для совместной обработки
        self.combined_encoder = nn.Sequential(
            conv_func(ndf * 4, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),
            conv_func(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True)
        )
        
        # Финальный классификатор
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ndf * 8, 1)
        )
        
        # Дополнительная голова для оценки цветовой реалистичности
        self.color_quality = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ndf * 8, ndf * 4),
            nn.ReLU(True),
            nn.Linear(ndf * 4, 1),
            nn.Sigmoid()
        )
        
    def extract_color_channels(self, x):
        """
        Извлекает только цветовые каналы из изображения.
        
        Args:
            x (torch.Tensor): Входное изображение [B, C, H, W]
            
        Returns:
            torch.Tensor: Цветовые каналы [B, 2, H, W]
        """
        # Предполагаем, что x имеет формат Lab или RGB
        if x.size(1) >= 3:
            # Для Lab берем только каналы a и b
            # Для RGB преобразуем в более цветовую информацию (упрощенная версия)
            color_channels = x[:, 1:3, :, :]
        else:
            # Если меньше 3 каналов, возвращаем без изменений
            color_channels = x
            
        return color_channels
        
    def forward(self, x):
        """
        Прямое распространение через дискриминатор.
        
        Args:
            x (torch.Tensor): Входное изображение [B, input_channels, H, W]
            
        Returns:
            dict: {
                'realness': torch.Tensor,  # Скор реалистичности [B, 1]
                'color_quality': torch.Tensor,  # Оценка качества цветов [B, 1]
                'features': torch.Tensor  # Извлеченные признаки для дальнейшего использования
            }
        """
        # Извлекаем цветовые каналы
        color_channels = self.extract_color_channels(x)
        
        # Кодируем входное изображение целиком
        main_features = self.main_encoder(x)
        
        # Кодируем только цветовые каналы
        color_features = self.color_encoder(color_channels)
        
        # Объединяем признаки
        combined = torch.cat([main_features, color_features], dim=1)
        
        # Применяем объединенный энкодер
        features = self.combined_encoder(combined)
        
        # Получаем скор реалистичности
        realness = self.classifier(features)
        
        # Оцениваем качество цветов
        quality = self.color_quality(features)
        
        return {
            'realness': realness,
            'color_quality': quality,
            'features': features
        }


class ColorQualityMetric(nn.Module):
    """
    Метрика для оценки качества колоризации на основе сравнения с реальным изображением.
    
    Args:
        input_channels (int): Количество входных каналов
        ndf (int): Базовое количество признаков
    """
    def __init__(self, input_channels=3, ndf=64):
        super(ColorQualityMetric, self).__init__()
        
        # Энкодер для извлечения признаков
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True)
        )
        
        # Оценка качества
        self.quality_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ndf * 4, ndf * 2),
            nn.ReLU(True),
            nn.Linear(ndf * 2, 1),
            nn.Sigmoid()
        )
        
        # Оценка сходства
        self.similarity_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ndf * 4 * 2, ndf * 2),
            nn.ReLU(True),
            nn.Linear(ndf * 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, real_img, fake_img):
        """
        Оценивает качество колоризации.
        
        Args:
            real_img (torch.Tensor): Реальное изображение [B, input_channels, H, W]
            fake_img (torch.Tensor): Сгенерированное изображение [B, input_channels, H, W]
            
        Returns:
            dict: {
                'quality': torch.Tensor,  # Оценка качества колоризации [B, 1]
                'similarity': torch.Tensor,  # Сходство с реальным изображением [B, 1]
                'real_features': torch.Tensor,  # Признаки реального изображения
                'fake_features': torch.Tensor  # Признаки сгенерированного изображения
            }
        """
        # Извлекаем признаки
        real_features = self.encoder(real_img)
        fake_features = self.encoder(fake_img)
        
        # Оцениваем качество сгенерированного изображения
        quality = self.quality_head(fake_features)
        
        # Объединяем признаки для оценки сходства
        combined_features = torch.cat([
            real_features.view(real_features.size(0), -1), 
            fake_features.view(fake_features.size(0), -1)
        ], dim=1)
        
        # Оцениваем сходство
        similarity = self.similarity_head(combined_features)
        
        return {
            'quality': quality,
            'similarity': similarity,
            'real_features': real_features,
            'fake_features': fake_features
        }


class MotivationalDiscriminator(nn.Module):
    """
    Основной класс дискриминатора с системой наград для колоризатора.
    
    Args:
        input_channels (int): Количество входных каналов
        ndf (int): Базовое количество признаков
        num_discriminators (int): Количество дискриминаторов в мультимасштабной архитектуре
        n_layers (int): Количество слоев в каждом дискриминаторе
        use_attention (bool): Использовать ли механизм внимания
        use_spectral_norm (bool): Использовать ли спектральную нормализацию
        use_semantic (bool): Использовать ли семантическую информацию
        use_rewards (bool): Использовать ли систему наград
    """
    def __init__(self, input_channels=3, ndf=64, num_discriminators=3, n_layers=3,
                 use_attention=True, use_spectral_norm=True, 
                 use_semantic=True, use_rewards=True):
        super(MotivationalDiscriminator, self).__init__()
        
        self.use_semantic = use_semantic
        self.use_rewards = use_rewards
        
        # Мультимасштабный дискриминатор
        self.multiscale_disc = MultiscaleDiscriminator(
            input_channels=input_channels,
            num_discriminators=num_discriminators,
            ndf=ndf,
            n_layers=n_layers,
            use_attention=use_attention,
            use_spectral_norm=use_spectral_norm
        )
        
        # Специализированный цветовой дискриминатор
        self.color_disc = ColorDiscriminator(
            input_channels=input_channels,
            ndf=ndf,
            use_spectral_norm=use_spectral_norm
        )
        
        # Семантический дискриминатор (если используется)
        if use_semantic:
            self.semantic_disc = SemanticDiscriminatorBlock(
                input_channels=input_channels,
                semantic_channels=150,  # Количество семантических классов
                ndf=ndf,
                use_spectral_norm=use_spectral_norm
            )
            
        # Генератор наград (если используется)
        if use_rewards:
            self.reward_generator = RewardGenerator(
                input_channels=input_channels,
                ndf=ndf,
                use_spectral_norm=use_spectral_norm
            )
            
        # Метрика качества колоризации
        self.quality_metric = ColorQualityMetric(
            input_channels=input_channels,
            ndf=ndf
        )
        
        # Для отслеживания статистик
        self.register_buffer('total_rewards', torch.zeros(1))
        self.register_buffer('positive_rewards', torch.zeros(1))
        self.register_buffer('negative_rewards', torch.zeros(1))
        self.register_buffer('reward_count', torch.zeros(1, dtype=torch.long))
        
    def forward(self, fake_img, real_img=None, semantics=None):
        """
        Прямое распространение через дискриминатор.
        
        Args:
            fake_img (torch.Tensor): Сгенерированное изображение [B, input_channels, H, W]
            real_img (torch.Tensor, optional): Реальное изображение [B, input_channels, H, W]
            semantics (torch.Tensor, optional): Семантическая информация [B, semantic_channels, H, W]
            
        Returns:
            dict: Результаты работы дискриминатора, включающие скоры реалистичности,
                  качество цветов и потенциальные награды
        """
        results = {}
        
        # Мультимасштабный дискриминатор для общей оценки реалистичности
        multiscale_results = self.multiscale_disc(fake_img)
        results['multiscale_scores'] = multiscale_results
        
        # Цветовой дискриминатор для оценки качества цветов
        color_results = self.color_disc(fake_img)
        results['color_scores'] = color_results
        
        # Если предоставлена семантическая информация и используется семантический дискриминатор
        if self.use_semantic and semantics is not None:
            semantic_score = self.semantic_disc(fake_img, semantics)
            results['semantic_score'] = semantic_score
            
        # Если предоставлено реальное изображение
        if real_img is not None:
            # Оценка качества колоризации
            quality_results = self.quality_metric(real_img, fake_img)
            results['quality_metrics'] = quality_results
            
            # Если используется система наград
            if self.use_rewards:
                reward_results = self.reward_generator(real_img, fake_img)
                results['rewards'] = reward_results
                
                # Обновляем статистики наград
                with torch.no_grad():
                    reward_value = reward_results['reward']
                    self.total_rewards += torch.sum(reward_value).item()
                    self.positive_rewards += torch.sum(torch.clamp(reward_value, min=0)).item()
                    self.negative_rewards += torch.sum(torch.clamp(reward_value, max=0)).item()
                    self.reward_count += reward_value.numel()
                    
        return results
    
    def get_reward_stats(self):
        """
        Возвращает статистики наград.
        
        Returns:
            dict: Статистики наград
        """
        avg_reward = self.total_rewards / max(self.reward_count, 1)
        avg_positive = self.positive_rewards / max(self.reward_count, 1)
        avg_negative = self.negative_rewards / max(self.reward_count, 1)
        
        stats = {
            'avg_reward': avg_reward.item(),
            'avg_positive': avg_positive.item(),
            'avg_negative': avg_negative.item(),
            'reward_count': self.reward_count.item()
        }
        
        return stats
        
    def reset_stats(self):
        """
        Сбрасывает статистики наград.
        """
        self.total_rewards.zero_()
        self.positive_rewards.zero_()
        self.negative_rewards.zero_()
        self.reward_count.zero_()


def create_motivational_discriminator(config=None):
    """
    Создает экземпляр MotivationalDiscriminator с заданной конфигурацией.
    
    Args:
        config (dict, optional): Словарь с параметрами конфигурации
        
    Returns:
        MotivationalDiscriminator: Экземпляр дискриминатора
    """
    # Параметры по умолчанию
    default_config = {
        'input_channels': 3,
        'ndf': 64,
        'num_discriminators': 3,
        'n_layers': 3,
        'use_attention': True,
        'use_spectral_norm': True,
        'use_semantic': True,
        'use_rewards': True
    }
    
    # Объединяем с пользовательской конфигурацией
    if config is not None:
        for key, value in config.items():
            if key in default_config:
                default_config[key] = value
    
    # Создаем модель с указанными параметрами
    model = MotivationalDiscriminator(
        input_channels=default_config['input_channels'],
        ndf=default_config['ndf'],
        num_discriminators=default_config['num_discriminators'],
        n_layers=default_config['n_layers'],
        use_attention=default_config['use_attention'],
        use_spectral_norm=default_config['use_spectral_norm'],
        use_semantic=default_config['use_semantic'],
        use_rewards=default_config['use_rewards']
    )
    
    return model


if __name__ == "__main__":
    # Пример использования дискриминатора
    
    # Создаем модель
    disc = create_motivational_discriminator()
    
    # Создаем тестовые данные
    batch_size = 2
    height, width = 256, 256
    
    # Сгенерированное цветное изображение
    fake_img = torch.randn(batch_size, 3, height, width)
    
    # Реальное цветное изображение
    real_img = torch.randn(batch_size, 3, height, width)
    
    # Семантическая информация (one-hot кодирование классов)
    semantics = torch.randn(batch_size, 150, height, width)
    
    # Прямое распространение
    results = disc(fake_img, real_img, semantics)
    
    # Выводим информацию о результатах
    print("Results from MotivationalDiscriminator:")
    for key, value in results.items():
        if isinstance(value, list):
            print(f"{key}: list of {len(value)} tensors")
        elif isinstance(value, dict):
            print(f"{key}: {', '.join(value.keys())}")
        elif isinstance(value, torch.Tensor):
            print(f"{key}: tensor of shape {value.shape}")
    
    # Выводим статистики наград
    reward_stats = disc.get_reward_stats()
    print("\nReward Statistics:")
    for key, value in reward_stats.items():
        print(f"{key}: {value}")