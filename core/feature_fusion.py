"""
Multi-Head Feature Fusion Module: Интеллектуальное слияние признаков с весами внимания.

Данный модуль обеспечивает эффективное слияние признаков из разных источников (Swin-UNet, ViT, FPN),
учитывая важность каждого источника с помощью механизма внимания. Это позволяет модели
адаптивно выбирать наиболее релевантную информацию для колоризации изображения,
что особенно важно для сложных объектов и текстур.

Основные компоненты:
- Multi-Head Attention: Слияние признаков через механизм многоголовочного внимания
- Feature Refinement: Уточнение признаков с учетом контекста
- Adaptive Weighting: Адаптивное взвешивание признаков из разных источников
- Channel-Spatial Attention: Канальное и пространственное внимание для фокусировки
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, reduce


class ChannelAttention(nn.Module):
    """
    Механизм канального внимания для взвешивания каналов признаков.
    
    Args:
        in_channels (int): Количество входных каналов
        reduction_ratio (int): Коэффициент снижения размерности
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Общий MLP для обоих типов пулинга
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Прямое распространение через канальное внимание.
        
        Args:
            x (torch.Tensor): Входные признаки [B, C, H, W]
            
        Returns:
            torch.Tensor: Канальные веса внимания [B, C, 1, 1]
        """
        # Применяем среднее и максимальное пулинг
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        
        # Объединяем результаты
        out = avg_out + max_out
        
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Механизм пространственного внимания для фокусировки на важных областях.
    
    Args:
        kernel_size (int): Размер ядра свертки
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        
        # Сверточный слой для обработки карт признаков
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Прямое распространение через пространственное внимание.
        
        Args:
            x (torch.Tensor): Входные признаки [B, C, H, W]
            
        Returns:
            torch.Tensor: Пространственные веса внимания [B, 1, H, W]
        """
        # Объединяем среднее и максимальное значения по каналам
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Конкатенация и обработка
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        
        return self.sigmoid(x)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM) для канального и пространственного внимания.
    
    Args:
        in_channels (int): Количество входных каналов
        reduction_ratio (int): Коэффициент снижения размерности
        spatial_kernel_size (int): Размер ядра для пространственного внимания
    """
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        super().__init__()
        
        # Канальное внимание
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        
        # Пространственное внимание
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel_size)
        
    def forward(self, x):
        """
        Прямое распространение через CBAM.
        
        Args:
            x (torch.Tensor): Входные признаки [B, C, H, W]
            
        Returns:
            torch.Tensor: Уточненные признаки [B, C, H, W]
        """
        # Применяем канальное внимание
        x = x * self.channel_attention(x)
        
        # Применяем пространственное внимание
        x = x * self.spatial_attention(x)
        
        return x


class FeatureRefinementBlock(nn.Module):
    """
    Блок уточнения признаков с использованием внимания и residual connections.
    
    Args:
        in_channels (int): Количество входных каналов
        out_channels (int): Количество выходных каналов
        use_attention (bool): Использовать ли CBAM
    """
    def __init__(self, in_channels, out_channels, use_attention=True):
        super().__init__()
        
        # Основной путь
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Residual connection
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()
        
        # Внимание (опционально)
        self.use_attention = use_attention
        if use_attention:
            self.attention = CBAM(out_channels)
        
    def forward(self, x):
        """
        Прямое распространение через блок уточнения признаков.
        
        Args:
            x (torch.Tensor): Входные признаки [B, C, H, W]
            
        Returns:
            torch.Tensor: Уточненные признаки [B, out_channels, H, W]
        """
        residual = self.residual(x)
        
        # Основной путь
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Residual connection
        out = out + residual
        
        # Применяем внимание (если используется)
        if self.use_attention:
            out = self.attention(out)
            
        out = self.relu2(out)
        
        return out


class MultiHeadCrossAttention(nn.Module):
    """
    Многоголовочное перекрестное внимание для слияния признаков из разных источников.
    
    Args:
        embed_dim (int): Размерность вектора признаков
        num_heads (int): Количество голов внимания
        dropout (float): Вероятность dropout
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim должен быть кратен num_heads"
        
        # Проекции для Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Выходная проекция
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Масштабирующий фактор
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query, key, value):
        """
        Прямое распространение через многоголовочное внимание.
        
        Args:
            query (torch.Tensor): Тензор запросов [B, L_q, C]
            key (torch.Tensor): Тензор ключей [B, L_k, C]
            value (torch.Tensor): Тензор значений [B, L_v, C]
            
        Returns:
            torch.Tensor: Взвешенная сумма значений [B, L_q, C]
        """
        batch_size = query.size(0)
        
        # Проекции
        q = self.q_proj(query)  # [B, L_q, C]
        k = self.k_proj(key)    # [B, L_k, C]
        v = self.v_proj(value)  # [B, L_v, C]
        
        # Разделение на головы
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_q, D]
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_k, D]
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_v, D]
        
        # Вычисление внимания
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L_q, L_k]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Взвешенная сумма значений
        attn_output = torch.matmul(attn_weights, v)  # [B, H, L_q, D]
        
        # Объединение голов и проекция на выход
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        
        return attn_output


class AdaptiveFeatureWeighting(nn.Module):
    """
    Адаптивное взвешивание признаков на основе их важности.
    
    Args:
        in_channels (int): Количество входных каналов
        num_features (int): Количество входных потоков признаков
        reduction_ratio (int): Коэффициент снижения размерности
    """
    def __init__(self, in_channels, num_features, reduction_ratio=8):
        super().__init__()
        
        self.num_features = num_features
        
        # Глобальное пулинг
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # MLP для оценки важности каждого потока признаков
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, num_features),
            nn.Softmax(dim=1)
        )
        
    def forward(self, feature_list):
        """
        Прямое распространение для адаптивного взвешивания признаков.
        
        Args:
            feature_list (list): Список тензоров признаков [B, C, H, W]
            
        Returns:
            torch.Tensor: Взвешенные признаки [B, C, H, W]
        """
        assert len(feature_list) == self.num_features, f"Ожидается {self.num_features} потоков признаков"
        
        # Выбираем первый поток признаков для оценки весов
        x = feature_list[0]
        batch_size, channels = x.shape[0], x.shape[1]
        
        # Глобальное пулинг
        pooled = self.gap(x).view(batch_size, channels)
        
        # Вычисляем веса для каждого потока
        weights = self.mlp(pooled).view(batch_size, self.num_features, 1, 1, 1)
        
        # Адаптивно взвешиваем и суммируем признаки
        stacked_features = torch.stack(feature_list, dim=1)  # [B, N, C, H, W]
        weighted_features = stacked_features * weights
        output = torch.sum(weighted_features, dim=1)  # [B, C, H, W]
        
        return output


class DynamicRouting(nn.Module):
    """
    Динамическая маршрутизация признаков для слияния информации из разных источников.
    Использует идеи из Capsule Networks для определения важности каждого источника признаков.
    
    Args:
        in_channels (int): Количество входных каналов
        num_features (int): Количество входных потоков признаков
        num_iterations (int): Количество итераций маршрутизации
    """
    def __init__(self, in_channels, num_features, num_iterations=3):
        super().__init__()
        
        self.num_features = num_features
        self.num_iterations = num_iterations
        
        # Преобразования для каждого потока признаков
        self.transforms = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
            for _ in range(num_features)
        ])
        
        # Финальная обработка после слияния
        self.output_transform = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, feature_list):
        """
        Прямое распространение через динамическую маршрутизацию.
        
        Args:
            feature_list (list): Список тензоров признаков [B, C, H, W]
            
        Returns:
            torch.Tensor: Взвешенные признаки после маршрутизации [B, C, H, W]
        """
        assert len(feature_list) == self.num_features, f"Ожидается {self.num_features} потоков признаков"
        
        # Преобразуем признаки
        transformed_features = [transform(feature) for transform, feature in zip(self.transforms, feature_list)]
        
        # Создаем маршрутные веса
        batch_size = feature_list[0].shape[0]
        h, w = feature_list[0].shape[2], feature_list[0].shape[3]
        
        # Инициализируем логиты маршрутизации (b_ij)
        routing_logits = torch.zeros(batch_size, self.num_features, 1, h, w, device=feature_list[0].device)
        
        # Итерации маршрутизации
        for i in range(self.num_iterations):
            # Веса маршрутизации (c_ij)
            routing_weights = F.softmax(routing_logits, dim=1)
            
            # Взвешенные признаки
            stacked_features = torch.stack(transformed_features, dim=1)  # [B, N, C, H, W]
            weighted_sum = torch.sum(stacked_features * routing_weights, dim=1)  # [B, C, H, W]
            
            if i < self.num_iterations - 1:
                # Обновляем логиты маршрутизации
                activation = torch.norm(weighted_sum, dim=1, keepdim=True)  # [B, 1, H, W]
                
                for j in range(self.num_features):
                    similarity = F.cosine_similarity(
                        transformed_features[j], weighted_sum, dim=1, keepdim=True
                    )  # [B, 1, H, W]
                    routing_logits[:, j, :, :, :] += similarity.unsqueeze(1)
        
        # Финальная обработка
        output = self.output_transform(weighted_sum)
        
        return output


class CrossModalAttention(nn.Module):
    """
    Кросс-модальное внимание для взаимодействия между разными типами признаков.
    
    Args:
        dim (int): Размерность вектора признаков
        num_heads (int): Количество голов внимания
        dropout (float): Вероятность dropout
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Проекции
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # Нормализация
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, context=None):
        """
        Прямое распространение через кросс-модальное внимание.
        
        Args:
            x (torch.Tensor): Основной тензор признаков [B, L, C]
            context (torch.Tensor, optional): Контекстный тензор признаков [B, L_ctx, C]
            
        Returns:
            torch.Tensor: Обогащенные признаки [B, L, C]
        """
        # Если контекст не предоставлен, используем самовнимание
        if context is None:
            context = x
            
        # Сохраняем исходный тензор для residual connection
        shortcut = x
        
        # Нормализация
        x = self.norm1(x)
        
        # Multi-head attention
        q = self.q_proj(x).view(-1, x.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(-1, context.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(-1, context.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        
        # Вычисление внимания
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Применение внимания
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(-1, x.size(1), self.dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.proj_dropout(attn_output)
        
        # Residual connection
        x = shortcut + attn_output
        
        # Feed forward
        x = x + self.mlp(self.norm2(x))
        
        return x


class MultiHeadFeatureFusion(nn.Module):
    """
    Основной модуль для многоголовочного слияния признаков из разных источников.
    
    Args:
        input_dims (dict): Словарь размерностей входных признаков {"source_name": dim}
        fusion_dim (int): Размерность объединенных признаков
        num_heads (int): Количество голов внимания
        dropout (float): Вероятность dropout
        fusion_method (str): Метод слияния признаков ("attention", "routing", "adaptive", "concat")
    """
    def __init__(self, input_dims, fusion_dim=256, num_heads=8, dropout=0.1, fusion_method="attention"):
        super().__init__()
        
        self.input_dims = input_dims
        self.fusion_dim = fusion_dim
        self.sources = list(input_dims.keys())
        self.fusion_method = fusion_method
        
        # Проекции для приведения всех признаков к одной размерности
        self.projections = nn.ModuleDict({
            source: nn.Sequential(
                nn.Linear(dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.Dropout(dropout)
            ) for source, dim in input_dims.items()
        })
        
        # Кросс-модальное внимание между разными источниками
        self.cross_attentions = nn.ModuleDict({
            f"{source1}_to_{source2}": CrossModalAttention(fusion_dim, num_heads, dropout)
            for i, source1 in enumerate(self.sources)
            for j, source2 in enumerate(self.sources)
            if i < j  # Только уникальные пары
        })
        
        # Методы слияния признаков
        if fusion_method == "attention":
            # Многоголовочное внимание для слияния
            self.fusion_module = MultiHeadCrossAttention(fusion_dim, num_heads, dropout)
        elif fusion_method == "routing":
            # Динамическая маршрутизация
            self.fusion_module = DynamicRouting(fusion_dim, len(input_dims), num_iterations=3)
        elif fusion_method == "adaptive":
            # Адаптивное взвешивание
            self.fusion_module = AdaptiveFeatureWeighting(fusion_dim, len(input_dims), reduction_ratio=8)
        elif fusion_method == "concat":
            # Конкатенация с последующей проекцией
            self.fusion_module = nn.Sequential(
                nn.Linear(fusion_dim * len(input_dims), fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
            
        # Финальное уточнение признаков
        self.feature_refinement = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # Выходная проекция в зависимости от задачи
        self.output_projection = nn.Linear(fusion_dim, fusion_dim)
            
    def forward(self, feature_dict):
        """
        Прямое распространение через модуль слияния признаков.
        
        Args:
            feature_dict (dict): Словарь тензоров признаков {"source_name": tensor}
            
        Returns:
            dict: {
                "fused_features": torch.Tensor,  # Объединенные признаки
                "per_source_features": dict,     # Признаки после cross-attention для каждого источника
                "attention_weights": dict        # Веса внимания (при использовании attention)
            }
        """
        # Проверка входных данных
        assert set(feature_dict.keys()) == set(self.sources), "Источники признаков не совпадают с ожидаемыми"
        
        # Проекция входных признаков
        projected_features = {
            source: self.projections[source](feature)
            for source, feature in feature_dict.items()
        }
        
        # Кросс-модальное внимание между источниками
        cross_modal_features = {}
        
        # Для каждого источника применяем внимание от других источников
        for target_source in self.sources:
            # Начинаем с собственных признаков
            enhanced_feature = projected_features[target_source]
            
            # Применяем внимание от каждого другого источника
            for source in self.sources:
                if source == target_source:
                    continue
                    
                # Определяем правильный ключ для модуля внимания
                if f"{source}_to_{target_source}" in self.cross_attentions:
                    attention_key = f"{source}_to_{target_source}"
                else:
                    attention_key = f"{target_source}_to_{source}"
                    
                # Применяем кросс-модальное внимание
                enhanced_feature = self.cross_attentions[attention_key](
                    enhanced_feature, projected_features[source]
                )
                
            # Сохраняем улучшенные признаки
            cross_modal_features[target_source] = enhanced_feature
            
        # Слияние признаков разными методами
        if self.fusion_method == "attention":
            # Выбираем первый источник как запрос, остальные как ключи/значения
            query_source = self.sources[0]
            key_value_features = torch.cat([
                cross_modal_features[source] for source in self.sources if source != query_source
            ], dim=1)
            
            # Применяем многоголовочное внимание
            fused_features = self.fusion_module(
                cross_modal_features[query_source],  # query
                key_value_features,                  # key
                key_value_features                   # value
            )
            
        elif self.fusion_method in ["routing", "adaptive"]:
            # Преобразуем признаки из формата [B, L, C] в [B, C, H, W]
            spatial_features = []
            for source in self.sources:
                feature = cross_modal_features[source]
                batch_size, seq_len, channels = feature.shape
                
                # Предполагаем квадратное разрешение
                hw = int(math.sqrt(seq_len))
                if hw * hw == seq_len:  # Только если seq_len = h*w
                    spatial_feature = feature.view(batch_size, hw, hw, channels).permute(0, 3, 1, 2)
                    spatial_features.append(spatial_feature)
                else:
                    # Если не квадратное, делаем проекцию
                    spatial_feature = feature.permute(0, 2, 1).unsqueeze(-1)  # [B, C, L, 1]
                    spatial_feature = F.adaptive_avg_pool2d(spatial_feature, (hw, hw))
                    spatial_features.append(spatial_feature)
            
            # Применяем модуль слияния
            fused_spatial = self.fusion_module(spatial_features)
            
            # Возвращаем в формат [B, L, C]
            fused_features = fused_spatial.flatten(2).transpose(1, 2)
            
        elif self.fusion_method == "concat":
            # Конкатенация всех признаков
            concat_features = torch.cat([
                cross_modal_features[source] for source in self.sources
            ], dim=-1)
            
            # Применяем проекцию
            fused_features = self.fusion_module(concat_features)
            
        # Финальное уточнение признаков
        refined_features = self.feature_refinement(fused_features)
        
        # Выходная проекция
        output_features = self.output_projection(refined_features)
        
        return {
            "fused_features": output_features,
            "per_source_features": cross_modal_features,
        }


class SpatialFeatureFusion(nn.Module):
    """
    Модуль для слияния пространственных признаков с разрешениями разных масштабов.
    
    Args:
        in_channels_list (list): Список количества каналов для каждого масштаба
        out_channels (int): Количество выходных каналов
        target_size (tuple, optional): Целевой размер (H, W), если None, используется размер признаков верхнего уровня
    """
    def __init__(self, in_channels_list, out_channels, target_size=None):
        super().__init__()
        
        self.target_size = target_size
        self.num_levels = len(in_channels_list)
        
        # Выравнивание каналов для каждого уровня
        self.channel_alignments = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        
        # Уточнение признаков каждого уровня
        self.refinements = nn.ModuleList([
            FeatureRefinementBlock(out_channels, out_channels, use_attention=True)
            for _ in range(self.num_levels)
        ])
        
        # Веса внимания для каждого уровня
        self.attention_weights = nn.Parameter(torch.ones(self.num_levels) / self.num_levels)
        
        # Финальная свертка
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, feature_list):
        """
        Прямое распространение через модуль пространственного слияния.
        
        Args:
            feature_list (list): Список тензоров признаков [B, C_i, H_i, W_i] для каждого масштаба
            
        Returns:
            torch.Tensor: Объединенные пространственные признаки [B, out_channels, H, W]
        """
        assert len(feature_list) == self.num_levels, f"Ожидается {self.num_levels} уровней признаков"
        
        # Определяем целевой размер
        if self.target_size:
            target_size = self.target_size
        else:
            # Используем размер верхнего уровня (с самым высоким разрешением)
            target_size = feature_list[0].shape[2:]
            
        # Выравнивание каналов и масштабирование
        aligned_features = []
        for i, feature in enumerate(feature_list):
            # Выравнивание каналов
            feature = self.channel_alignments[i](feature)
            
            # Масштабирование до целевого размера (если нужно)
            if feature.shape[2:] != target_size:
                feature = F.interpolate(feature, size=target_size, mode='bilinear', align_corners=True)
                
            # Уточнение признаков
            feature = self.refinements[i](feature)
            
            aligned_features.append(feature)
            
        # Нормализуем веса
        normalized_weights = F.softmax(self.attention_weights, dim=0)
        
        # Взвешенное суммирование
        output = torch.zeros_like(aligned_features[0])
        for i, feature in enumerate(aligned_features):
            output += feature * normalized_weights[i].view(1, 1, 1, 1)
            
        # Финальная обработка
        output = self.final_conv(output)
        
        return output


class FeatureTransformer(nn.Module):
    """
    Трансформер для обработки признаков с пространственным вниманием.
    
    Args:
        dim (int): Размерность вектора признаков
        num_heads (int): Количество голов внимания
        mlp_ratio (float): Соотношение скрытой размерности MLP к dim
        dropout (float): Вероятность dropout
        num_layers (int): Количество слоев трансформера
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1, num_layers=2):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=int(dim * mlp_ratio),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        """
        Прямое распространение через трансформер.
        
        Args:
            x (torch.Tensor): Входные признаки [B, L, C]
            mask (torch.Tensor, optional): Маска внимания
            
        Returns:
            torch.Tensor: Обработанные признаки [B, L, C]
        """
        for layer in self.layers:
            x = layer(x, src_mask=mask)
            
        return x


class MultiModalFeatureFusion(nn.Module):
    """
    Полный модуль для мультимодального слияния признаков из разных источников,
    с поддержкой как пространственных, так и последовательных данных.
    
    Args:
        spatial_dims (dict): Словарь размерностей пространственных признаков {"source_name": (C, H, W)}
        sequential_dims (dict): Словарь размерностей последовательных признаков {"source_name": (L, C)}
        fusion_dim (int): Размерность объединенных признаков
        spatial_target_size (tuple): Целевой размер для пространственных признаков (H, W)
        num_heads (int): Количество голов внимания
        dropout (float): Вероятность dropout
    """
    def __init__(self, spatial_dims, sequential_dims, fusion_dim=256, 
                 spatial_target_size=(64, 64), num_heads=8, dropout=0.1):
        super().__init__()
        
        self.spatial_sources = list(spatial_dims.keys())
        self.sequential_sources = list(sequential_dims.keys())
        self.fusion_dim = fusion_dim
        self.spatial_target_size = spatial_target_size
        
        # Обработка пространственных признаков
        if spatial_dims:
            self.spatial_projections = nn.ModuleDict({
                source: nn.Sequential(
                    nn.Conv2d(channels, fusion_dim, kernel_size=1),
                    nn.BatchNorm2d(fusion_dim),
                    nn.ReLU(inplace=True)
                )
                for source, (channels, _, _) in spatial_dims.items()
            })
            
            # Слияние пространственных признаков
            self.spatial_fusion = SpatialFeatureFusion(
                in_channels_list=[fusion_dim] * len(spatial_dims),
                out_channels=fusion_dim,
                target_size=spatial_target_size
            )
        
        # Обработка последовательных признаков
        if sequential_dims:
            self.sequential_projections = nn.ModuleDict({
                source: nn.Sequential(
                    nn.Linear(dim, fusion_dim),
                    nn.LayerNorm(fusion_dim),
                    nn.Dropout(dropout)
                )
                for source, (_, dim) in sequential_dims.items()
            })
            
            # Трансформер для последовательных признаков
            self.sequential_transformer = FeatureTransformer(
                dim=fusion_dim,
                num_heads=num_heads,
                dropout=dropout,
                num_layers=2
            )
        
        # Объединение пространственных и последовательных признаков
        if spatial_dims and sequential_dims:
            self.spatial_to_sequential = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1)
            )
            
            # Кросс-модальное внимание
            self.cross_modal_attention = CrossModalAttention(
                dim=fusion_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            
        # Финальное слияние
        input_dim = fusion_dim * (1 if not (spatial_dims and sequential_dims) else 2)
        self.final_fusion = nn.Sequential(
            nn.Linear(input_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
            
    def forward(self, inputs):
        """
        Прямое распространение через модуль мультимодального слияния.
        
        Args:
            inputs (dict): Словарь входных тензоров {"source_name": tensor}
            
        Returns:
            dict: {
                "fused_features": torch.Tensor,  # Финальные объединенные признаки
                "spatial_features": torch.Tensor, # Пространственные признаки (если есть)
                "sequential_features": torch.Tensor, # Последовательные признаки (если есть)
            }
        """
        results = {}
        
        # Обработка пространственных признаков
        if hasattr(self, 'spatial_projections'):
            spatial_features = []
            
            for source in self.spatial_sources:
                if source in inputs:
                    # Проекция и обработка
                    projected = self.spatial_projections[source](inputs[source])
                    spatial_features.append(projected)
                    
            if spatial_features:
                # Слияние пространственных признаков
                spatial_fused = self.spatial_fusion(spatial_features)
                results["spatial_features"] = spatial_fused
        
        # Обработка последовательных признаков
        if hasattr(self, 'sequential_projections'):
            sequential_features = []
            
            for source in self.sequential_sources:
                if source in inputs:
                    # Проекция
                    projected = self.sequential_projections[source](inputs[source])
                    sequential_features.append(projected)
                    
            if sequential_features:
                # Конкатенация всех последовательных признаков
                sequential_concat = torch.cat(sequential_features, dim=1)
                
                # Обработка трансформером
                sequential_fused = self.sequential_transformer(sequential_concat)
                results["sequential_features"] = sequential_fused
        
        # Объединение пространственных и последовательных признаков
        if "spatial_features" in results and "sequential_features" in results:
            # Преобразование пространственных признаков в последовательные
            spatial_seq = self.spatial_to_sequential(results["spatial_features"])
            
            # Применяем кросс-модальное внимание
            enhanced_seq = self.cross_modal_attention(
                results["sequential_features"],
                spatial_seq.unsqueeze(1)  # Добавляем размерность последовательности
            )
            
            # Объединяем признаки
            fused_features = torch.cat([spatial_seq, enhanced_seq.mean(dim=1)], dim=1)
            fused_features = self.final_fusion(fused_features)
            
        elif "spatial_features" in results:
            # Только пространственные признаки
            fused_features = self.spatial_to_sequential(results["spatial_features"])
            
        elif "sequential_features" in results:
            # Только последовательные признаки
            fused_features = torch.mean(results["sequential_features"], dim=1)
            
        else:
            raise ValueError("Не предоставлены входные данные для обработки")
            
        results["fused_features"] = fused_features
        
        return results


# Создаем модуль для экспорта
def create_feature_fusion(
    spatial_dims=None,
    sequential_dims=None,
    fusion_dim=256,
    num_heads=8,
    dropout=0.1
):
    """
    Создает модуль Multi-Head Feature Fusion для слияния признаков из разных источников.
    
    Args:
        spatial_dims (dict, optional): Словарь размерностей пространственных признаков {"source_name": (C, H, W)}
        sequential_dims (dict, optional): Словарь размерностей последовательных признаков {"source_name": (L, C)}
        fusion_dim (int): Размерность объединенных признаков
        num_heads (int): Количество голов внимания
        dropout (float): Вероятность dropout
        
    Returns:
        nn.Module: Модуль для слияния признаков
    """
    # Если не указаны размерности, используем значения по умолчанию
    if spatial_dims is None:
        spatial_dims = {
            "swin_unet": (384, 64, 64),
            "fpn": (128, 64, 64)
        }
        
    if sequential_dims is None:
        sequential_dims = {
            "vit": (196, 768)
        }
        
    return MultiModalFeatureFusion(
        spatial_dims=spatial_dims,
        sequential_dims=sequential_dims,
        fusion_dim=fusion_dim,
        spatial_target_size=(64, 64),
        num_heads=num_heads,
        dropout=dropout
    )


# Пример использования
if __name__ == "__main__":
    # Создание модуля слияния признаков
    fusion_module = create_feature_fusion()
    
    # Создание тестовых входных данных
    batch_size = 2
    
    # Пространственные признаки
    swin_features = torch.randn(batch_size, 384, 64, 64)
    fpn_features = torch.randn(batch_size, 128, 64, 64)
    
    # Последовательные признаки
    vit_features = torch.randn(batch_size, 196, 768)
    
    # Формируем словарь входных данных
    inputs = {
        "swin_unet": swin_features,
        "fpn": fpn_features,
        "vit": vit_features
    }
    
    # Прямое распространение
    outputs = fusion_module(inputs)
    
    # Вывод формы выходных тензоров
    for key, value in outputs.items():
        print(f"{key}: {value.shape}")