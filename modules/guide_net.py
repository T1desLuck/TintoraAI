"""
GuideNet: Интеллектуальный советник по цветам на основе семантики.

Данный модуль реализует нейронную сеть GuideNet, которая выступает в роли "советника" 
по цветам для основного генератора. GuideNet анализирует семантику изображения и 
предлагает наиболее подходящие цветовые решения, основываясь на контексте, объектах 
и их взаимосвязях.

Ключевые особенности:
- Анализ семантического контекста изображения для предложения релевантных цветов
- Интеграция с базой знаний цветовых схем для различных объектов и сцен
- Система обратной связи для улучшения рекомендаций на основе успешности колоризации
- Механизм "наград" для усиления правильных цветовых решений
- Генерация цветовых подсказок разной степени детализации и уверенности

Преимущества для колоризации:
- Помогает основному генератору принимать более обоснованные решения о цветах
- Обеспечивает семантически корректные цветовые схемы для объектов
- Улучшает реалистичность и естественность колоризации
- Снижает вероятность выбора "безопасных", но неестественных цветов
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple, Union, Optional


class SemanticEncoder(nn.Module):
    """
    Энкодер для извлечения семантических признаков из изображения.
    
    Args:
        in_channels (int): Количество входных каналов
        base_channels (int): Базовое количество каналов
        num_stages (int): Количество стадий даунсемплинга
        use_attention (bool): Использовать ли механизм внимания
    """
    def __init__(self, in_channels=1, base_channels=64, num_stages=4, use_attention=True):
        super(SemanticEncoder, self).__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.use_attention = use_attention
        
        # Начальный свёрточный блок
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Создаем энкодер с DownBlocks
        self.down_blocks = nn.ModuleList()
        in_ch = base_channels
        
        for i in range(num_stages):
            out_ch = in_ch * 2
            self.down_blocks.append(DownBlock(in_ch, out_ch, use_attention=(i >= num_stages // 2 and use_attention)))
            in_ch = out_ch
            
        # Слой пространственного внимания для финальных признаков
        if use_attention:
            self.spatial_attention = SpatialAttention(in_ch)
        
    def forward(self, x):
        """
        Прямое распространение через энкодер.
        
        Args:
            x (torch.Tensor): Входное изображение [B, C, H, W]
            
        Returns:
            tuple: (список промежуточных признаков, итоговые признаки)
        """
        # Начальная свертка
        x = self.initial_conv(x)
        
        # Промежуточные признаки для skip connections и дальнейшего анализа
        features = []
        features.append(x)
        
        # Проходим через DownBlocks
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            features.append(x)
            
        # Применяем пространственное внимание к финальным признакам, если используется
        if self.use_attention:
            x = self.spatial_attention(x)
            
        return features, x


class DownBlock(nn.Module):
    """
    Блок понижающего разрешения с остаточными соединениями.
    
    Args:
        in_channels (int): Количество входных каналов
        out_channels (int): Количество выходных каналов
        use_attention (bool): Использовать ли механизм внимания
    """
    def __init__(self, in_channels, out_channels, use_attention=False):
        super(DownBlock, self).__init__()
        
        # Downsampling с использованием свёртки с шагом 2
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        
        # Два остаточных блока
        self.res_block1 = ResidualBlock(out_channels, out_channels)
        self.res_block2 = ResidualBlock(out_channels, out_channels)
        
        # Блок внимания (опционально)
        self.use_attention = use_attention
        if use_attention:
            self.attention = ChannelAttention(out_channels)
            
    def forward(self, x):
        """
        Прямое распространение через блок.
        
        Args:
            x (torch.Tensor): Входные данные
            
        Returns:
            torch.Tensor: Выходные данные
        """
        # Понижаем разрешение
        x = self.downsample(x)
        
        # Проходим через остаточные блоки
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        # Применяем внимание, если включено
        if self.use_attention:
            x = self.attention(x)
            
        return x


class ResidualBlock(nn.Module):
    """
    Стандартный остаточный блок.
    
    Args:
        in_channels (int): Количество входных каналов
        out_channels (int): Количество выходных каналов
        stride (int): Шаг свёртки
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Основной путь
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut соединение
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        """
        Прямое распространение через остаточный блок.
        
        Args:
            x (torch.Tensor): Входные данные
            
        Returns:
            torch.Tensor: Выходные данные
        """
        # Основной путь
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Добавляем shortcut соединение
        out += self.shortcut(x)
        out = self.relu(out)
        
        return out


class ChannelAttention(nn.Module):
    """
    Механизм внимания по каналам.
    
    Args:
        channels (int): Количество каналов
        reduction (int): Коэффициент уменьшения размерности для MLP
    """
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # MLP для обработки глобальной информации
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Прямое распространение через механизм внимания.
        
        Args:
            x (torch.Tensor): Входные данные [B, C, H, W]
            
        Returns:
            torch.Tensor: Взвешенные признаки [B, C, H, W]
        """
        # Обрабатываем усредненные признаки
        avg_out = self.fc(self.avg_pool(x))
        
        # Обрабатываем максимальные признаки
        max_out = self.fc(self.max_pool(x))
        
        # Комбинируем их и применяем сигмоиду
        out = self.sigmoid(avg_out + max_out)
        
        # Взвешиваем входные данные
        return x * out


class SpatialAttention(nn.Module):
    """
    Механизм пространственного внимания.
    
    Args:
        kernel_size (int): Размер ядра свёртки
    """
    def __init__(self, channels, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        # Слой свертки для обработки карт внимания
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        
        # Проекция для каналов
        self.channel_proj = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, kernel_size=1)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Прямое распространение через механизм внимания.
        
        Args:
            x (torch.Tensor): Входные данные [B, C, H, W]
            
        Returns:
            torch.Tensor: Выходные данные с примененным вниманием [B, C, H, W]
        """
        # Вычисляем средние и максимальные значения по каналам
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Объединяем их
        attention_map = torch.cat([avg_out, max_out], dim=1)
        
        # Применяем свертку и сигмоиду
        attention_map = self.conv(attention_map)
        attention_map = self.sigmoid(attention_map)
        
        # Проекция для каналов
        channel_refined = self.channel_proj(x)
        
        # Взвешиваем признаки
        return channel_refined * attention_map


class ColorSemanticProcessor(nn.Module):
    """
    Процессор для извлечения и анализа семантических связей между объектами и цветами.
    
    Args:
        in_channels (int): Количество входных каналов
        hidden_dim (int): Размерность скрытого пространства
        num_heads (int): Количество голов внимания
        num_classes (int): Количество семантических классов (объектов)
        color_dim (int): Размерность цветового пространства
    """
    def __init__(self, in_channels=512, hidden_dim=256, num_heads=8, num_classes=150, color_dim=64):
        super(ColorSemanticProcessor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.color_dim = color_dim
        
        # Проекция входных признаков в скрытое пространство
        self.feature_projection = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        
        # Трансформер для обработки семантической информации
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Позиционное кодирование
        self.pos_embedding = PositionalEncoding(hidden_dim, dropout=0.1, max_len=1000)
        
        # Семантический классификатор
        self.semantic_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Цветовой эмбеддинг (для каждого класса объектов)
        self.color_embeddings = nn.Embedding(num_classes, color_dim)
        
        # Проекция для генерации цветовых советов
        self.color_projector = nn.Sequential(
            nn.Linear(hidden_dim + color_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # 2 канала для ab в пространстве Lab
        )
        
        # Для оценки уверенности в цветовых советах
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim + color_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        """
        Прямое распространение через процессор.
        
        Args:
            features (torch.Tensor): Входные семантические признаки [B, C, H, W]
            
        Returns:
            tuple: (цветовые советы [B, 2, H, W], уверенность [B, 1, H, W], семантические логиты [B, num_classes, H, W])
        """
        batch_size, _, height, width = features.shape
        
        # Проекция признаков
        projected_features = self.feature_projection(features)
        
        # Изменяем размерность для трансформера
        # [B, C, H, W] -> [B, H*W, C]
        projected_features = projected_features.flatten(2).permute(0, 2, 1)
        
        # Добавляем позиционное кодирование
        projected_features = self.pos_embedding(projected_features)
        
        # Пропускаем через трансформер
        transformed_features = self.transformer_encoder(projected_features)
        
        # Семантическая классификация
        semantic_logits = self.semantic_classifier(transformed_features)
        
        # Получаем вероятности классов
        semantic_probs = F.softmax(semantic_logits, dim=-1)
        
        # Получаем цветовые эмбеддинги для каждого класса и взвешиваем их вероятностями
        # [B, H*W, num_classes] x [num_classes, color_dim] -> [B, H*W, color_dim]
        color_embeddings = torch.matmul(semantic_probs, self.color_embeddings.weight)
        
        # Объединяем семантические признаки и цветовые эмбеддинги
        combined_features = torch.cat([transformed_features, color_embeddings], dim=-1)
        
        # Генерируем цветовые советы
        color_advice = self.color_projector(combined_features)
        
        # Оцениваем уверенность в советах
        confidence = self.confidence_estimator(combined_features)
        
        # Преобразуем обратно в пространственную форму
        # [B, H*W, 2] -> [B, 2, H, W]
        color_advice = color_advice.permute(0, 2, 1).reshape(batch_size, 2, height, width)
        confidence = confidence.permute(0, 2, 1).reshape(batch_size, 1, height, width)
        semantic_logits = semantic_logits.permute(0, 2, 1).reshape(batch_size, self.num_classes, height, width)
        
        return color_advice, confidence, semantic_logits


class PositionalEncoding(nn.Module):
    """
    Позиционное кодирование для трансформера.
    
    Args:
        d_model (int): Размерность модели
        dropout (float): Вероятность dropout
        max_len (int): Максимальная длина последовательности
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        Добавляет позиционное кодирование к входным данным.
        
        Args:
            x (torch.Tensor): Входные данные [B, L, D]
            
        Returns:
            torch.Tensor: Выходные данные с позиционным кодированием [B, L, D]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ColorAdvisor(nn.Module):
    """
    Модуль для генерации цветовых советов на основе семантических признаков и
    референсных изображений.
    
    Args:
        in_channels (int): Количество входных каналов
        hidden_dim (int): Размерность скрытого пространства
        advice_channels (int): Количество каналов для цветовых советов
        use_reference (bool): Использовать ли референсные изображения
    """
    def __init__(self, in_channels=512, hidden_dim=256, advice_channels=2, use_reference=True):
        super(ColorAdvisor, self).__init__()
        
        self.use_reference = use_reference
        
        # Для обработки основных признаков
        self.main_processor = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Для обработки референсных признаков (если используются)
        if use_reference:
            self.reference_processor = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            )
            
            # Для слияния основных и референсных признаков
            self.fusion = nn.Sequential(
                nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            )
        
        # Для генерации цветовых советов
        self.color_advice_generator = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, advice_channels, kernel_size=1)
        )
        
        # Для оценки уверенности в советах
        self.confidence_estimator = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, features, reference_features=None):
        """
        Прямое распространение через модуль.
        
        Args:
            features (torch.Tensor): Основные признаки [B, C, H, W]
            reference_features (torch.Tensor, optional): Референсные признаки [B, C, H, W]
            
        Returns:
            tuple: (цветовые советы [B, advice_channels, H, W], уверенность [B, 1, H, W])
        """
        # Обработка основных признаков
        main_processed = self.main_processor(features)
        
        # Обработка референсных признаков и слияние, если они используются
        if self.use_reference and reference_features is not None:
            ref_processed = self.reference_processor(reference_features)
            
            # Изменяем размер референсных признаков, если нужно
            if ref_processed.shape[2:] != main_processed.shape[2:]:
                ref_processed = F.interpolate(
                    ref_processed, 
                    size=main_processed.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
                
            # Слияние признаков
            processed = self.fusion(torch.cat([main_processed, ref_processed], dim=1))
        else:
            processed = main_processed
            
        # Генерация цветовых советов
        color_advice = self.color_advice_generator(processed)
        
        # Оценка уверенности
        confidence = self.confidence_estimator(processed)
        
        return color_advice, confidence


class RewardModule(nn.Module):
    """
    Модуль для обучения с подкреплением на основе успешности цветовых советов.
    
    Args:
        hidden_dim (int): Размерность скрытого пространства
        reward_scale (float): Масштаб наград
    """
    def __init__(self, hidden_dim=256, reward_scale=1.0):
        super(RewardModule, self).__init__()
        
        self.reward_scale = reward_scale
        
        # Сеть для оценки качества совета
        self.quality_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Для отслеживания статистик наград
        self.register_buffer('total_rewards', torch.zeros(1))
        self.register_buffer('total_samples', torch.zeros(1, dtype=torch.long))
        self.register_buffer('positive_rewards', torch.zeros(1))
        self.register_buffer('negative_rewards', torch.zeros(1))
        
    def compute_reward(self, advice_features, color_advice, true_colors):
        """
        Вычисляет награду на основе сравнения цветового совета и истинных цветов.
        
        Args:
            advice_features (torch.Tensor): Признаки, использованные для генерации совета [B, L, hidden_dim]
            color_advice (torch.Tensor): Цветовой совет [B, L, 2]
            true_colors (torch.Tensor): Истинные цвета [B, L, 2]
            
        Returns:
            torch.Tensor: Вычисленная награда [B, L, 1]
        """
        # Вычисляем ошибку совета (L1 расстояние)
        color_error = torch.abs(color_advice - true_colors).sum(dim=-1, keepdim=True)
        
        # Нормализуем ошибку
        max_error = 2.0  # Максимальное L1 расстояние в ab-пространстве ~ 2.0
        normalized_error = torch.clamp(color_error / max_error, 0.0, 1.0)
        
        # Объединяем признаки, совет и ошибку
        combined = torch.cat([advice_features, color_advice, normalized_error], dim=-1)
        
        # Оцениваем качество совета
        quality = self.quality_estimator(combined)
        
        # Преобразуем качество в награду [0, 1] -> [-0.5, 0.5]
        reward = (quality - 0.5) * 2 * self.reward_scale
        
        # Обновляем статистики
        with torch.no_grad():
            self.total_rewards += torch.sum(reward).item()
            self.total_samples += reward.numel()
            self.positive_rewards += torch.sum(reward[reward > 0]).item()
            self.negative_rewards += torch.sum(reward[reward < 0]).item()
        
        return reward
        
    def get_statistics(self):
        """
        Возвращает статистики наград.
        
        Returns:
            dict: Статистики наград
        """
        avg_reward = self.total_rewards / max(self.total_samples, 1)
        avg_positive = self.positive_rewards / max(self.total_samples, 1)
        avg_negative = self.negative_rewards / max(self.total_samples, 1)
        
        stats = {
            'avg_reward': avg_reward.item(),
            'avg_positive': avg_positive.item(),
            'avg_negative': avg_negative.item(),
            'samples': self.total_samples.item()
        }
        
        return stats
        
    def reset_statistics(self):
        """
        Сбрасывает статистики наград.
        """
        self.total_rewards.zero_()
        self.total_samples.zero_()
        self.positive_rewards.zero_()
        self.negative_rewards.zero_()


class GuideNet(nn.Module):
    """
    GuideNet: Советник по цветам на основе семантики изображения.
    
    Основной класс, объединяющий все компоненты для генерации цветовых советов.
    
    Args:
        input_channels (int): Количество входных каналов
        base_channels (int): Базовое количество каналов
        num_stages (int): Количество стадий даунсемплинга
        hidden_dim (int): Размерность скрытого пространства
        use_attention (bool): Использовать ли механизм внимания
        use_semantic (bool): Использовать ли семантический анализ
        use_reference (bool): Использовать ли референсные изображения
        use_rewards (bool): Использовать ли систему наград
    """
    def __init__(self, input_channels=1, base_channels=64, num_stages=4, hidden_dim=256,
                 use_attention=True, use_semantic=True, use_reference=True, use_rewards=True):
        super(GuideNet, self).__init__()
        
        self.use_semantic = use_semantic
        self.use_reference = use_reference
        self.use_rewards = use_rewards
        
        # Энкодер для извлечения признаков
        self.encoder = SemanticEncoder(
            in_channels=input_channels, 
            base_channels=base_channels,
            num_stages=num_stages,
            use_attention=use_attention
        )
        
        # Определяем количество каналов в выходном слое энкодера
        encoder_out_channels = base_channels * (2 ** num_stages)
        
        # Семантический процессор (если используется)
        if use_semantic:
            self.semantic_processor = ColorSemanticProcessor(
                in_channels=encoder_out_channels,
                hidden_dim=hidden_dim,
                num_heads=8,
                num_classes=150,  # Типичное количество классов для семантической сегментации
                color_dim=64
            )
        
        # Советник по цветам
        self.color_advisor = ColorAdvisor(
            in_channels=encoder_out_channels,
            hidden_dim=hidden_dim,
            advice_channels=2,  # Для a и b каналов в Lab пространстве
            use_reference=use_reference
        )
        
        # Модуль наград (если используется)
        if use_rewards:
            self.reward_module = RewardModule(
                hidden_dim=hidden_dim,
                reward_scale=1.0
            )
            
        # Для отслеживания производительности
        self.register_buffer('successful_advices', torch.zeros(1))
        self.register_buffer('total_advices', torch.zeros(1))
        
    def forward(self, image, reference_image=None, target_colors=None):
        """
        Прямое распространение через GuideNet.
        
        Args:
            image (torch.Tensor): Входное черно-белое изображение [B, 1, H, W]
            reference_image (torch.Tensor, optional): Референсное изображение [B, 3, H, W]
            target_colors (torch.Tensor, optional): Целевые цвета [B, 2, H, W]
            
        Returns:
            dict: Результаты {
                'color_advice': torch.Tensor,  # Цветовые советы [B, 2, H, W]
                'confidence': torch.Tensor,    # Уверенность в советах [B, 1, H, W]
                'semantic_logits': torch.Tensor (optional),  # Семантические логиты
                'rewards': torch.Tensor (optional)  # Награды за советы
            }
        """
        # Кодируем входное изображение
        features, encoded = self.encoder(image)
        
        # Кодируем референсное изображение, если оно предоставлено
        reference_encoded = None
        if self.use_reference and reference_image is not None:
            # Для простоты используем тот же энкодер
            # В реальной реализации может быть отдельный энкодер для референсов
            _, reference_encoded = self.encoder(reference_image)
            
        results = {}
        
        # Применяем семантический процессор, если используется
        if self.use_semantic:
            semantic_advice, semantic_confidence, semantic_logits = self.semantic_processor(encoded)
            results['semantic_logits'] = semantic_logits
        
        # Генерируем цветовые советы
        color_advice, confidence = self.color_advisor(encoded, reference_encoded)
        
        # Сохраняем основные результаты
        results['color_advice'] = color_advice
        results['confidence'] = confidence
        
        # Если есть целевые цвета и используется система наград, вычисляем награды
        if self.use_rewards and target_colors is not None and self.training:
            # Подготавливаем данные для вычисления наград
            batch_size, _, height, width = encoded.shape
            
            # Преобразуем пространственные данные в последовательности
            flat_encoded = encoded.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
            flat_advice = color_advice.flatten(2).permute(0, 2, 1)  # [B, H*W, 2]
            flat_target = target_colors.flatten(2).permute(0, 2, 1)  # [B, H*W, 2]
            
            # Вычисляем награды
            rewards = self.reward_module.compute_reward(flat_encoded, flat_advice, flat_target)
            
            # Преобразуем обратно в пространственную форму
            rewards = rewards.permute(0, 2, 1).reshape(batch_size, 1, height, width)
            
            # Сохраняем награды в результаты
            results['rewards'] = rewards
            
            # Обновляем статистику успешных советов (положительные награды)
            with torch.no_grad():
                self.successful_advices += torch.sum(rewards > 0).item()
                self.total_advices += rewards.numel()
            
        return results
    
    def get_performance_stats(self):
        """
        Возвращает статистики производительности советника.
        
        Returns:
            dict: Статистики производительности
        """
        success_rate = (self.successful_advices / max(self.total_advices, 1)).item()
        
        stats = {
            'success_rate': success_rate,
            'total_advices': self.total_advices.item()
        }
        
        if self.use_rewards:
            reward_stats = self.reward_module.get_statistics()
            stats.update(reward_stats)
            
        return stats
        
    def reset_stats(self):
        """
        Сбрасывает статистики производительности.
        """
        self.successful_advices.zero_()
        self.total_advices.zero_()
        
        if self.use_rewards:
            self.reward_module.reset_statistics()


def create_guide_net(config=None):
    """
    Создает экземпляр GuideNet с заданной конфигурацией.
    
    Args:
        config (dict, optional): Словарь с параметрами конфигурации
        
    Returns:
        GuideNet: Экземпляр GuideNet
    """
    # Параметры по умолчанию
    default_config = {
        'input_channels': 1,
        'base_channels': 64,
        'num_stages': 4,
        'hidden_dim': 256,
        'use_attention': True,
        'use_semantic': True,
        'use_reference': True,
        'use_rewards': True
    }
    
    # Объединяем с пользовательской конфигурацией
    if config is not None:
        for key, value in config.items():
            if key in default_config:
                default_config[key] = value
    
    # Создаем модель с указанными параметрами
    model = GuideNet(
        input_channels=default_config['input_channels'],
        base_channels=default_config['base_channels'],
        num_stages=default_config['num_stages'],
        hidden_dim=default_config['hidden_dim'],
        use_attention=default_config['use_attention'],
        use_semantic=default_config['use_semantic'],
        use_reference=default_config['use_reference'],
        use_rewards=default_config['use_rewards']
    )
    
    return model


if __name__ == "__main__":
    # Пример использования GuideNet
    
    # Создаем модель
    guide_net = create_guide_net()
    
    # Создаем тестовые данные
    batch_size = 2
    height, width = 256, 256
    
    # Входное черно-белое изображение
    grayscale_image = torch.randn(batch_size, 1, height, width)
    
    # Референсное цветное изображение (опционально)
    reference_image = torch.randn(batch_size, 3, height, width)
    
    # Целевые цвета (для обучения)
    target_colors = torch.randn(batch_size, 2, height, width)
    
    # Прямое распространение
    results = guide_net(grayscale_image, reference_image, target_colors)
    
    # Выводим информацию о результатах
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
