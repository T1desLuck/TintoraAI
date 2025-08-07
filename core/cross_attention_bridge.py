"""
Cross-Attention Bridge: Мост взаимодействия между Swin-UNet и ViT компонентами.

Данный модуль реализует механизм cross-attention, позволяющий эффективно
связать локальные особенности из Swin-UNet с глобальным семантическим пониманием
из ViT. Это критический компонент для генерации правильных цветов с учетом
контекста и семантики изображения.

Основные компоненты:
- Cross-Attention Layer: Слой для взаимодействия между разными типами признаков
- Feature Alignment: Выравнивание размерностей признаков из разных источников
- Multi-Level Feature Fusion: Слияние признаков на разных уровнях детализации
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math


class FeatureAlignment(nn.Module):
    """
    Модуль для выравнивания размерностей признаков из разных источников.
    
    Args:
        swin_dim (int): Размерность признаков из Swin-UNet
        vit_dim (int): Размерность признаков из ViT
        fusion_dim (int): Целевая размерность слияния
        use_conv (bool): Использовать ли свертки для пространственного выравнивания
    """
    def __init__(self, swin_dim, vit_dim, fusion_dim, use_conv=True):
        super().__init__()
        self.swin_dim = swin_dim
        self.vit_dim = vit_dim
        self.fusion_dim = fusion_dim
        self.use_conv = use_conv
        
        # Преобразование признаков Swin-UNet
        self.swin_projection = nn.Linear(swin_dim, fusion_dim)
        
        # Преобразование признаков ViT
        self.vit_projection = nn.Linear(vit_dim, fusion_dim)
        
        # Слой для пространственного выравнивания (опционально)
        if use_conv:
            self.spatial_align = nn.Conv2d(fusion_dim, fusion_dim, kernel_size=3, padding=1)
        
        # Нормализация
        self.norm_swin = nn.LayerNorm(fusion_dim)
        self.norm_vit = nn.LayerNorm(fusion_dim)
        
    def forward(self, swin_features, vit_features, swin_resolution, vit_resolution):
        """
        Выравнивание признаков из разных источников.
        
        Args:
            swin_features (tensor): Признаки из Swin-UNet [B, N_swin, C_swin]
            vit_features (tensor): Признаки из ViT [B, N_vit, C_vit]
            swin_resolution (tuple): Разрешение признаков Swin (H, W)
            vit_resolution (tuple): Разрешение признаков ViT (H, W)
            
        Returns:
            tuple: Выровненные признаки (aligned_swin, aligned_vit)
        """
        B, N_swin, _ = swin_features.shape
        _, N_vit, _ = vit_features.shape
        
        # Проекция признаков в общее пространство
        swin_proj = self.swin_projection(swin_features)  # [B, N_swin, fusion_dim]
        vit_proj = self.vit_projection(vit_features)     # [B, N_vit, fusion_dim]
        
        # Нормализация
        swin_proj = self.norm_swin(swin_proj)
        vit_proj = self.norm_vit(vit_proj)
        
        # Пространственное выравнивание (если используется)
        if self.use_conv:
            H_swin, W_swin = swin_resolution
            swin_proj_2d = rearrange(swin_proj, 'b (h w) c -> b c h w', h=H_swin, w=W_swin)
            swin_proj_2d = self.spatial_align(swin_proj_2d)
            swin_proj = rearrange(swin_proj_2d, 'b c h w -> b (h w) c')
            
            # Для ViT признаков может потребоваться интерполяция, если разрешения не совпадают
            H_vit, W_vit = vit_resolution
            if H_vit != H_swin or W_vit != W_swin:
                vit_proj_2d = rearrange(vit_proj, 'b (h w) c -> b c h w', h=H_vit, w=W_vit)
                vit_proj_2d = F.interpolate(vit_proj_2d, size=(H_swin, W_swin), mode='bilinear', align_corners=False)
                vit_proj = rearrange(vit_proj_2d, 'b c h w -> b (h w) c')
        
        return swin_proj, vit_proj


class CrossAttentionLayer(nn.Module):
    """
    Слой Cross-Attention для взаимодействия между признаками из разных источников.
    
    Args:
        dim (int): Размерность признаков
        num_heads (int): Количество голов внимания
        qkv_bias (bool): Использовать ли смещение в QKV проекциях
        qk_scale (float): Масштабирование для QK произведения (если None, используется по умолчанию)
        attn_drop (float): Dropout для весов внимания
        proj_drop (float): Dropout для выходных проекций
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        # Проекции для Q, K, V
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        # Dropout
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, query_features, key_features, value_features=None):
        """
        Прямое распространение через слой Cross-Attention.
        
        Args:
            query_features (tensor): Признаки-запросы [B, N_q, C]
            key_features (tensor): Признаки-ключи [B, N_kv, C]
            value_features (tensor): Признаки-значения [B, N_kv, C], если None, то равны key_features
            
        Returns:
            tensor: Выходные признаки после применения внимания [B, N_q, C]
        """
        B, N_q, C = query_features.shape
        N_k = key_features.shape[1]
        
        if value_features is None:
            value_features = key_features
            
        # Проекция Query, Key, Value
        q = self.q_proj(query_features).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B, h, N_q, d
        k = self.k_proj(key_features).reshape(B, N_k, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)    # B, h, N_k, d
        v = self.v_proj(value_features).reshape(B, N_k, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B, h, N_k, d
        
        # Вычисление attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, h, N_q, N_k
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Взвешенная сумма значений
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)  # B, N_q, C
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MultiLevelFeatureFusion(nn.Module):
    """
    Модуль для слияния признаков из разных уровней иерархии.
    
    Args:
        dims (list): Список размерностей признаков на разных уровнях
        fusion_dim (int): Размерность слияния
        num_levels (int): Количество уровней для слияния
        use_weights (bool): Использовать ли веса для слияния
    """
    def __init__(self, dims, fusion_dim, num_levels=3, use_weights=True):
        super().__init__()
        self.num_levels = num_levels
        self.use_weights = use_weights
        
        # Проекции для каждого уровня
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dims[i], fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU()
            ) for i in range(num_levels)
        ])
        
        # Веса для взвешенного слияния (если используются)
        if use_weights:
            self.weights = nn.Parameter(torch.ones(num_levels) / num_levels)
            self.weight_norm = nn.Softmax(dim=0)
        
        # Финальная проекция после слияния
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU()
        )
        
    def forward(self, features_list):
        """
        Слияние признаков с разных уровней.
        
        Args:
            features_list (list): Список тензоров признаков [B, N_i, C_i]
            
        Returns:
            tensor: Слитые признаки [B, N_0, fusion_dim]
        """
        assert len(features_list) >= self.num_levels, "Недостаточно уровней признаков"
        
        # Используем только указанное количество уровней
        features_to_fuse = features_list[:self.num_levels]
        
        # Проекция всех признаков в общую размерность
        projected_features = []
        base_resolution = None
        
        for i, features in enumerate(features_to_fuse):
            B, N, _ = features.shape
            
            # Сохраняем разрешение базового (первого) уровня
            if i == 0:
                base_resolution = N
                base_features_shape = features.shape
            
            # Проекция в общую размерность
            proj_features = self.projections[i](features)
            
            # Изменение разрешения для соответствия базовому уровню
            if N != base_resolution:
                # Предполагаем, что N = H*W и можно определить H, W
                side_len = int(math.sqrt(N))
                if side_len * side_len == N:  # Проверка на квадратное разрешение
                    proj_features = rearrange(proj_features, 'b (h w) c -> b c h w', h=side_len, w=side_len)
                    base_side_len = int(math.sqrt(base_resolution))
                    proj_features = F.interpolate(proj_features, size=(base_side_len, base_side_len), 
                                                mode='bilinear', align_corners=False)
                    proj_features = rearrange(proj_features, 'b c h w -> b (h w) c')
                else:
                    # Если не квадратное, используем линейную интерполяцию
                    proj_features = F.interpolate(
                        proj_features.unsqueeze(1), size=(base_resolution, proj_features.shape[-1]), 
                        mode='bilinear', align_corners=False
                    ).squeeze(1)
            
            projected_features.append(proj_features)
        
        # Слияние признаков (взвешенное или простое среднее)
        if self.use_weights:
            normalized_weights = self.weight_norm(self.weights)
            fused_features = sum(w * f for w, f in zip(normalized_weights, projected_features))
        else:
            fused_features = sum(projected_features) / len(projected_features)
        
        # Финальная проекция
        fused_features = self.fusion_proj(fused_features)
        
        return fused_features


class CrossAttentionBridge(nn.Module):
    """
    Полный мост между Swin-UNet и ViT с использованием cross-attention.
    
    Args:
        swin_dims (list): Список размерностей признаков из Swin-UNet на разных уровнях
        vit_dim (int): Размерность признаков из ViT
        fusion_dim (int): Размерность для слияния признаков
        num_heads (int): Количество голов внимания
        mlp_ratio (float): Соотношение скрытой размерности MLP к fusion_dim
        drop_path (float): Вероятность DropPath
        norm_layer (nn.Module): Слой нормализации
        use_multi_level (bool): Использовать ли слияние нескольких уровней признаков
    """
    def __init__(
        self, 
        swin_dims, 
        vit_dim, 
        fusion_dim=512, 
        num_heads=8, 
        mlp_ratio=4., 
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        use_multi_level=True
    ):
        super().__init__()
        self.use_multi_level = use_multi_level
        
        # Модуль выравнивания признаков для основного уровня
        self.feature_alignment = FeatureAlignment(
            swin_dim=swin_dims[0],  # Используем размерность самого высокого уровня Swin
            vit_dim=vit_dim,
            fusion_dim=fusion_dim,
            use_conv=True
        )
        
        # Слияние признаков с разных уровней (если используется)
        if use_multi_level:
            self.multi_level_fusion = MultiLevelFeatureFusion(
                dims=swin_dims,
                fusion_dim=fusion_dim,
                num_levels=len(swin_dims),
                use_weights=True
            )
        
        # Cross-attention от Swin к ViT (Swin queries ViT)
        self.cross_attn_swin_to_vit = CrossAttentionLayer(
            dim=fusion_dim,
            num_heads=num_heads,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.,
            proj_drop=0.
        )
        
        # Cross-attention от ViT к Swin (ViT queries Swin)
        self.cross_attn_vit_to_swin = CrossAttentionLayer(
            dim=fusion_dim,
            num_heads=num_heads,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.,
            proj_drop=0.
        )
        
        # MLP после внимания для Swin
        self.mlp_swin = nn.Sequential(
            norm_layer(fusion_dim),
            nn.Linear(fusion_dim, int(fusion_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(int(fusion_dim * mlp_ratio), fusion_dim),
            nn.Dropout(0.1)
        )
        
        # MLP после внимания для ViT
        self.mlp_vit = nn.Sequential(
            norm_layer(fusion_dim),
            nn.Linear(fusion_dim, int(fusion_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(int(fusion_dim * mlp_ratio), fusion_dim),
            nn.Dropout(0.1)
        )
        
        # Нормализация
        self.norm1_swin = norm_layer(fusion_dim)
        self.norm2_swin = norm_layer(fusion_dim)
        self.norm1_vit = norm_layer(fusion_dim)
        self.norm2_vit = norm_layer(fusion_dim)
        
        # DropPath
        self.drop_path = nn.Identity() if drop_path == 0. else DropPath(drop_path)
        
        # Финальные проекции обратно в оригинальные размерности
        self.proj_back_swin = nn.Linear(fusion_dim, swin_dims[0])
        self.proj_back_vit = nn.Linear(fusion_dim, vit_dim)
        
        # Оценка важности информации (attention gates)
        self.swin_attention_gate = nn.Sequential(
            nn.Linear(fusion_dim, 1),
            nn.Sigmoid()
        )
        
        self.vit_attention_gate = nn.Sequential(
            nn.Linear(fusion_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, swin_features_list, vit_features, swin_resolution, vit_resolution):
        """
        Прямое распространение через CrossAttentionBridge.
        
        Args:
            swin_features_list (list): Список тензоров признаков из Swin-UNet [B, N_swin, C_swin]
            vit_features (tensor): Признаки из ViT [B, N_vit, C_vit]
            swin_resolution (tuple): Разрешение признаков Swin (H, W)
            vit_resolution (tuple): Разрешение признаков ViT (H, W)
            
        Returns:
            tuple: Обновленные признаки (updated_swin_features, updated_vit_features)
        """
        # Подготовка признаков Swin (слияние нескольких уровней или использование верхнего уровня)
        if self.use_multi_level and len(swin_features_list) > 1:
            # Слияние признаков с разных уровней
            fused_swin_features = self.multi_level_fusion(swin_features_list)
        else:
            # Используем только верхний уровень
            fused_swin_features = swin_features_list[0]
        
        # Выравнивание размерностей
        aligned_swin, aligned_vit = self.feature_alignment(
            fused_swin_features, vit_features, swin_resolution, vit_resolution
        )
        
        # Cross-attention от Swin к ViT (Swin запрашивает контекст от ViT)
        # Сохраняем исходные признаки для residual connection
        swin_shortcut = aligned_swin
        aligned_swin = self.norm1_swin(aligned_swin)
        swin_with_vit_context = self.cross_attn_swin_to_vit(
            query_features=aligned_swin,  # Swin как запросы
            key_features=aligned_vit,     # ViT как ключи
            value_features=aligned_vit     # ViT как значения
        )
        
        # Применяем attention gate для взвешивания важности информации от ViT
        swin_attn_weights = self.swin_attention_gate(swin_with_vit_context)
        swin_with_vit_context = swin_attn_weights * swin_with_vit_context
        
        # Residual connection и dropout
        aligned_swin = swin_shortcut + self.drop_path(swin_with_vit_context)
        
        # FFN (MLP) для Swin признаков
        aligned_swin = aligned_swin + self.drop_path(self.mlp_swin(self.norm2_swin(aligned_swin)))
        
        # Cross-attention от ViT к Swin (ViT запрашивает детали от Swin)
        # Сохраняем исходные признаки для residual connection
        vit_shortcut = aligned_vit
        aligned_vit = self.norm1_vit(aligned_vit)
        vit_with_swin_context = self.cross_attn_vit_to_swin(
            query_features=aligned_vit,  # ViT как запросы
            key_features=aligned_swin,   # Swin как ключи
            value_features=aligned_swin   # Swin как значения
        )
        
        # Применяем attention gate для взвешивания важности информации от Swin
        vit_attn_weights = self.vit_attention_gate(vit_with_swin_context)
        vit_with_swin_context = vit_attn_weights * vit_with_swin_context
        
        # Residual connection и dropout
        aligned_vit = vit_shortcut + self.drop_path(vit_with_swin_context)
        
        # FFN (MLP) для ViT признаков
        aligned_vit = aligned_vit + self.drop_path(self.mlp_vit(self.norm2_vit(aligned_vit)))
        
        # Проекция обратно в оригинальные размерности
        updated_swin_features = self.proj_back_swin(aligned_swin)
        updated_vit_features = self.proj_back_vit(aligned_vit)
        
        return updated_swin_features, updated_vit_features


class DynamicFeatureAggregation(nn.Module):
    """
    Динамическое агрегирование признаков с адаптивными весами.
    
    Args:
        dim (int): Размерность входных признаков
        reduction_ratio (int): Коэффициент снижения размерности для внутреннего представления
        num_sources (int): Количество источников признаков для агрегации
    """
    def __init__(self, dim, reduction_ratio=4, num_sources=2):
        super().__init__()
        self.dim = dim
        self.num_sources = num_sources
        
        # Модуль внимания для расчета весов каждого источника признаков
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction_ratio, num_sources),
            nn.Softmax(dim=-1)
        )
        
        # Нормализация и проекция для выхода
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, features_list):
        """
        Динамическое агрегирование признаков.
        
        Args:
            features_list (list): Список тензоров признаков [B, N, C] для агрегации
            
        Returns:
            tensor: Агрегированные признаки [B, N, C]
        """
        assert len(features_list) == self.num_sources, f"Ожидается {self.num_sources} источников признаков, получено {len(features_list)}"
        
        # Предполагаем, что все тензоры имеют одинаковую форму
        B, N, C = features_list[0].shape
        
        # Создаем усредненное представление для расчета весов
        avg_features = sum(features_list) / self.num_sources
        
        # Используем среднее для расчета весов с помощью механизма внимания
        # Сначала усредняем по пространственному измерению
        avg_spatial = torch.mean(avg_features, dim=1)  # [B, C]
        
        # Вычисляем веса для каждого источника
        weights = self.attention(avg_spatial)  # [B, num_sources]
        
        # Применяем веса к каждому источнику и агрегируем
        weighted_sum = torch.zeros_like(features_list[0])
        for i, features in enumerate(features_list):
            # Для каждого пакета и каждого источника применяем соответствующий вес
            weight = weights[:, i].view(B, 1, 1)  # [B, 1, 1]
            weighted_sum += weight * features
        
        # Нормализация и финальная проекция
        weighted_sum = self.norm(weighted_sum)
        weighted_sum = self.proj(weighted_sum)
        
        return weighted_sum


class BridgeController(nn.Module):
    """
    Контроллер для управления потоком информации через CrossAttentionBridge.
    Решает, какой контекст должен преобладать: локальный или глобальный.
    
    Args:
        dim (int): Размерность признаков
        hidden_dim (int): Размерность скрытого представления
    """
    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        
        # Анализатор контекста
        self.context_analyzer = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),  # 2 канала: локальный и глобальный контекст
            nn.Softmax(dim=-1)
        )
        
        # Нормализация
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, local_features, global_features, content_type=None):
        """
        Управление потоком информации.
        
        Args:
            local_features (tensor): Локальные признаки (Swin) [B, N, C]
            global_features (tensor): Глобальные признаки (ViT) [B, N, C]
            content_type (tensor, optional): Тип контента для условного контроля [B, num_types]
            
        Returns:
            tensor: Взвешенная комбинация признаков [B, N, C]
        """
        # Объединяем признаки для анализа контекста
        combined_features = local_features + global_features
        combined_features = self.norm(combined_features)
        
        # Усреднение по пространственному измерению
        avg_features = torch.mean(combined_features, dim=1)  # [B, C]
        
        # Анализ контекста
        context_weights = self.context_analyzer(avg_features)  # [B, 2]
        
        # Если предоставлен тип контента, модифицируем веса
        if content_type is not None:
            # Например, для определенных типов контента можем усилить глобальный контекст
            # content_type влияет на context_weights
            # Это можно реализовать по-разному в зависимости от задачи
            pass
        
        # Применяем веса для комбинирования признаков
        local_weight = context_weights[:, 0].view(-1, 1, 1)  # [B, 1, 1]
        global_weight = context_weights[:, 1].view(-1, 1, 1)  # [B, 1, 1]
        
        weighted_features = local_weight * local_features + global_weight * global_features
        
        return weighted_features


# Вспомогательная функция для создания CrossAttentionBridge с предопределенными параметрами
def create_cross_attention_bridge(swin_dims=[128, 256, 512, 1024], vit_dim=768, fusion_dim=512, num_heads=8):
    """
    Создает модуль CrossAttentionBridge для связи между Swin-UNet и ViT.
    
    Args:
        swin_dims (list): Список размерностей признаков из Swin-UNet на разных уровнях
        vit_dim (int): Размерность признаков из ViT
        fusion_dim (int): Размерность для слияния признаков
        num_heads (int): Количество голов внимания
        
    Returns:
        nn.Module: Модуль CrossAttentionBridge
    """
    return CrossAttentionBridge(
        swin_dims=swin_dims,
        vit_dim=vit_dim,
        fusion_dim=fusion_dim,
        num_heads=num_heads,
        mlp_ratio=4.0,
        drop_path=0.1,
        norm_layer=nn.LayerNorm,
        use_multi_level=True
    )


# Пример использования
if __name__ == "__main__":
    # Пример создания CrossAttentionBridge
    bridge = create_cross_attention_bridge()
    
    # Пример входных данных
    batch_size = 2
    swin_resolution = (16, 16)
    vit_resolution = (16, 16)
    
    # Пример списка признаков Swin-UNet с разных уровней
    swin_features_1 = torch.randn(batch_size, swin_resolution[0]*swin_resolution[1], 128)  # Уровень 1
    swin_features_2 = torch.randn(batch_size, swin_resolution[0]*swin_resolution[1]//4, 256)  # Уровень 2
    swin_features_3 = torch.randn(batch_size, swin_resolution[0]*swin_resolution[1]//16, 512)  # Уровень 3
    swin_features_4 = torch.randn(batch_size, swin_resolution[0]*swin_resolution[1]//64, 1024)  # Уровень 4
    
    swin_features_list = [swin_features_1, swin_features_2, swin_features_3, swin_features_4]
    
    # Пример признаков ViT
    vit_features = torch.randn(batch_size, vit_resolution[0]*vit_resolution[1], 768)
    
    # Прямое распространение
    updated_swin, updated_vit = bridge(swin_features_list, vit_features, swin_resolution, vit_resolution)
    
    print(f"Входные признаки Swin уровня 1: {swin_features_1.shape}")
    print(f"Входные признаки ViT: {vit_features.shape}")
    print(f"Обновленные признаки Swin: {updated_swin.shape}")
    print(f"Обновленные признаки ViT: {updated_vit.shape}")