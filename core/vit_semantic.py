"""
ViT Semantic: Семантический блок на основе Vision Transformer.

Данный модуль реализует понимание контекста и семантики изображения
с помощью архитектуры Vision Transformer (ViT). В отличие от CNN,
ViT обрабатывает изображение как последовательность патчей,
что позволяет эффективнее улавливать глобальные зависимости и
семантический контекст для более точной колоризации.

Основные компоненты:
- Patch Embedding: Преобразование изображения в последовательность патчей
- Transformer Encoder: Обработка патчей с помощью self-attention
- Semantic Projection: Проекция признаков в семантическое пространство
- Classification Token: Специальный токен для обобщения информации об изображении
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math


class PatchEmbedding(nn.Module):
    """
    Преобразование изображения в последовательность патчей с embedding.
    
    Args:
        img_size (int): Размер входного изображения
        patch_size (int): Размер патча
        in_chans (int): Число каналов входного изображения
        embed_dim (int): Размерность embedding вектора
        norm_layer (nn.Module, optional): Слой нормализации
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768, norm_layer=None):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.num_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])
        
        # Проекция для преобразования патчей в embedding
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        
        # Нормализация (если задана)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
    def forward(self, x):
        """
        Прямое распространение.
        
        Args:
            x (torch.Tensor): Входное изображение формы [B, C, H, W]
            
        Returns:
            torch.Tensor: Последовательность embedded патчей формы [B, num_patches, embed_dim]
        """
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Размер входа {H}x{W} не соответствует ожидаемому размеру {self.img_size[0]}x{self.img_size[1]}"
            
        # Проекция патчей и преобразование формы
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = rearrange(x, 'b c h w -> b (h w) c')  # [B, num_patches, embed_dim]
        
        # Нормализация
        x = self.norm(x)
        
        return x


class Attention(nn.Module):
    """
    Механизм внимания для ViT.
    
    Args:
        dim (int): Размерность входного вектора
        num_heads (int): Количество голов внимания
        qkv_bias (bool): Использовать ли смещение в QKV проекциях
        qk_scale (float): Масштабирование для QK произведения
        attn_drop (float): Вероятность dropout для весов внимания
        proj_drop (float): Вероятность dropout для выходной проекции
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        # Проекция для Query, Key, Value
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # Dropout для весов внимания
        self.attn_drop = nn.Dropout(attn_drop)
        
        # Выходная проекция и dropout
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        """
        Прямое распространение.
        
        Args:
            x (torch.Tensor): Входной тензор формы [B, N, C]
            
        Returns:
            torch.Tensor: Выходной тензор с той же формой
        """
        B, N, C = x.shape
        
        # Проекция QKV и разделение на головы
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]
        
        # Вычисление весов внимания
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Взвешенная сумма значений
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        
        # Выходная проекция и dropout
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """
    Многослойный персептрон для блоков Transformer.
    
    Args:
        dim (int): Размерность входного вектора
        hidden_dim (int): Размерность скрытого слоя
        dropout (float): Вероятность dropout
    """
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Прямое распространение.
        
        Args:
            x (torch.Tensor): Входной тензор
            
        Returns:
            torch.Tensor: Выходной тензор
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Блок Transformer для ViT.
    
    Args:
        dim (int): Размерность входного вектора
        num_heads (int): Количество голов внимания
        mlp_ratio (float): Соотношение размерности MLP к размерности входа
        qkv_bias (bool): Использовать ли смещение в QKV проекциях
        qk_scale (float): Масштабирование для QK произведения
        drop (float): Вероятность dropout
        attn_drop (float): Вероятность dropout для весов внимания
        drop_path (float): Вероятность DropPath
        norm_layer (nn.Module): Тип слоя нормализации
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        
        # Блок внимания
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop
        )
        
        # DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        
        # MLP блок
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim=dim, hidden_dim=mlp_hidden_dim, dropout=drop)
        
    def forward(self, x):
        """
        Прямое распространение через блок Transformer.
        
        Args:
            x (torch.Tensor): Входной тензор
            
        Returns:
            torch.Tensor: Выходной тензор
        """
        # Внимание с residual connection
        x = x + self.drop_path(self.attn(self.norm1(x)))
        
        # MLP с residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


# Определение DropPath для стохастической глубины
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Стохастическое падение путей (DropPath) для регуляризации.
    
    Args:
        x (torch.Tensor): Входной тензор
        drop_prob (float): Вероятность падения пути
        training (bool): Находится ли модель в режиме обучения
        
    Returns:
        torch.Tensor: Выходной тензор
    """
    if drop_prob == 0. or not training:
        return x
    
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # работает для любой размерности
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # бинаризация
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Модуль DropPath (стохастическая глубина).
    
    Args:
        drop_prob (float): Вероятность падения пути
    """
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class SemanticHead(nn.Module):
    """
    Семантическая голова для извлечения контекстной информации.
    
    Args:
        dim (int): Размерность входного вектора
        semantic_dim (int): Размерность семантических признаков
        hidden_dim (int): Размерность скрытого слоя
        dropout (float): Вероятность dropout
    """
    def __init__(self, dim, semantic_dim=256, hidden_dim=1024, dropout=0.1):
        super().__init__()
        
        self.semantic_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, semantic_dim),
            nn.LayerNorm(semantic_dim)
        )
        
    def forward(self, x, cls_token=None):
        """
        Прямое распространение.
        
        Args:
            x (torch.Tensor): Входные признаки [B, N, C]
            cls_token (torch.Tensor, optional): CLS токен [B, 1, C]
            
        Returns:
            torch.Tensor: Семантические признаки
        """
        if cls_token is not None:
            # Используем только CLS токен для семантического представления
            semantic_features = self.semantic_proj(cls_token)
        else:
            # Усредняем все токены
            semantic_features = self.semantic_proj(torch.mean(x, dim=1, keepdim=True))
            
        return semantic_features


class ContextualAttention(nn.Module):
    """
    Контекстуальное внимание для улучшения семантического представления.
    
    Args:
        dim (int): Размерность входного вектора
        context_dim (int): Размерность контекстного вектора
        num_heads (int): Количество голов внимания
        dropout (float): Вероятность dropout
    """
    def __init__(self, dim, context_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Проекции для contextual attention
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(context_dim, dim)
        self.v_proj = nn.Linear(context_dim, dim)
        
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x, context):
        """
        Прямое распространение.
        
        Args:
            x (torch.Tensor): Входные признаки [B, N, C]
            context (torch.Tensor): Контекстные признаки [B, M, C_context]
            
        Returns:
            torch.Tensor: Обогащенные контекстом признаки [B, N, C]
        """
        B, N, C = x.shape
        M = context.shape[1]
        
        # Проекции Q из x, K и V из context
        q = self.q_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # [B, h, N, d]
        k = self.k_proj(context).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # [B, h, M, d]
        v = self.v_proj(context).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # [B, h, M, d]
        
        # Вычисление attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, h, N, M]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Взвешенная сумма значений
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class ViTSemantic(nn.Module):
    """
    Семантический блок на основе Vision Transformer для колоризации.
    
    Args:
        img_size (int): Размер входного изображения
        patch_size (int): Размер патча
        in_chans (int): Число каналов входного изображения
        embed_dim (int): Размерность embedding вектора
        depth (int): Глубина (число блоков Transformer)
        num_heads (int): Количество голов внимания
        mlp_ratio (float): Соотношение размерности MLP к размерности входа
        qkv_bias (bool): Использовать ли смещение в QKV проекциях
        qk_scale (float): Масштабирование для QK произведения
        drop_rate (float): Вероятность dropout
        attn_drop_rate (float): Вероятность dropout для весов внимания
        drop_path_rate (float): Вероятность DropPath
        norm_layer (nn.Module): Тип слоя нормализации
        semantic_dim (int): Размерность семантических признаков
        use_cls_token (bool): Использовать ли CLS токен
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 semantic_dim=256, use_cls_token=True):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, norm_layer=norm_layer
        )
        num_patches = self.patch_embed.num_patches
        
        # CLS token и позиционное кодирование
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            num_tokens = num_patches + 1
        else:
            num_tokens = num_patches
            
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Transformer блоки
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        
        # Нормализация
        self.norm = norm_layer(embed_dim)
        
        # Семантическая голова
        self.semantic_head = SemanticHead(dim=embed_dim, semantic_dim=semantic_dim)
        
        # Контекстуальное внимание для обогащения признаков
        self.contextual_attn = ContextualAttention(
            dim=embed_dim,
            context_dim=semantic_dim,
            num_heads=num_heads
        )
        
        # Инициализация весов
        self._init_weights()
        
    def _init_weights(self):
        """
        Инициализация весов модели.
        """
        # Инициализация позиционных эмбеддингов
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Инициализация CLS токена (если используется)
        if self.use_cls_token:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            
        # Применяем функцию инициализации к модулям
        self.apply(self._init_weights_recursive)
            
    def _init_weights_recursive(self, m):
        """
        Рекурсивная инициализация весов для всех модулей.
        
        Args:
            m (nn.Module): Модуль для инициализации
        """
        if isinstance(m, nn.Linear):
            # Инициализация линейных слоев
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # Инициализация слоев нормализации
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def interpolate_pos_encoding(self, x, h, w):
        """
        Интерполяция позиционного кодирования для изображений с произвольным разрешением.
        
        Args:
            x (torch.Tensor): Входной тензор
            h (int): Высота в патчах
            w (int): Ширина в патчах
            
        Returns:
            torch.Tensor: Интерполированное позиционное кодирование
        """
        npatch = x.shape[1] - 1 if self.use_cls_token else x.shape[1]
        N = self.pos_embed.shape[1] - 1 if self.use_cls_token else self.pos_embed.shape[1]
        
        if npatch == N:
            return self.pos_embed
            
        class_pos_embed = self.pos_embed[:, 0:1] if self.use_cls_token else None
        patch_pos_embed = self.pos_embed[:, 1:] if self.use_cls_token else self.pos_embed
        
        dim = x.shape[-1]
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(patch_pos_embed, size=(h, w), mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
        
        if self.use_cls_token:
            return torch.cat((class_pos_embed, patch_pos_embed), dim=1)
        else:
            return patch_pos_embed
    
    def forward_features(self, x):
        """
        Прямое распространение для извлечения признаков.
        
        Args:
            x (torch.Tensor): Входное изображение [B, C, H, W]
            
        Returns:
            tuple: (all_tokens, cls_token или mean_token)
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        
        # Добавление CLS токена (если используется)
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        
        # Добавление позиционного кодирования
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Прохождение через Transformer блоки
        for blk in self.blocks:
            x = blk(x)
            
        # Нормализация
        x = self.norm(x)
        
        # Выделяем CLS токен или усредняем все токены
        if self.use_cls_token:
            cls_token = x[:, 0:1]  # [B, 1, embed_dim]
            patch_tokens = x[:, 1:]  # [B, num_patches, embed_dim]
        else:
            cls_token = torch.mean(x, dim=1, keepdim=True)  # [B, 1, embed_dim]
            patch_tokens = x  # [B, num_patches, embed_dim]
            
        return patch_tokens, cls_token
    
    def forward_semantic(self, patch_tokens, cls_token):
        """
        Создание семантических признаков и контекстуально обогащенных признаков.
        
        Args:
            patch_tokens (torch.Tensor): Токены патчей [B, num_patches, embed_dim]
            cls_token (torch.Tensor): CLS токен или усредненный токен [B, 1, embed_dim]
            
        Returns:
            tuple: (enriched_features, semantic_features)
        """
        # Получаем семантическое представление
        semantic_features = self.semantic_head(patch_tokens, cls_token)  # [B, 1, semantic_dim]
        
        # Обогащаем признаки семантическим контекстом
        enriched_features = self.contextual_attn(patch_tokens, semantic_features)
        
        return enriched_features, semantic_features
    
    def forward(self, x):
        """
        Полное прямое распространение через ViT Semantic.
        
        Args:
            x (torch.Tensor): Входное изображение [B, C, H, W]
            
        Returns:
            dict: {
                'enriched_features': torch.Tensor,  # Обогащенные контекстом признаки
                'semantic_features': torch.Tensor,  # Семантические признаки
                'patch_tokens': torch.Tensor,       # Исходные токены патчей
                'cls_token': torch.Tensor          # CLS токен или усредненный токен
            }
        """
        # Извлечение признаков
        patch_tokens, cls_token = self.forward_features(x)
        
        # Создание семантических признаков и обогащение контекстом
        enriched_features, semantic_features = self.forward_semantic(patch_tokens, cls_token)
        
        return {
            'enriched_features': enriched_features,
            'semantic_features': semantic_features,
            'patch_tokens': patch_tokens,
            'cls_token': cls_token
        }


def create_vit_semantic(img_size=256, patch_size=16, in_chans=1, embed_dim=768,
                        depth=12, num_heads=12, semantic_dim=256, use_cls_token=True):
    """
    Создает модель ViT Semantic для колоризации.
    
    Args:
        img_size (int): Размер входного изображения
        patch_size (int): Размер патча
        in_chans (int): Число каналов входного изображения
        embed_dim (int): Размерность embedding вектора
        depth (int): Глубина (число блоков Transformer)
        num_heads (int): Количество голов внимания
        semantic_dim (int): Размерность семантических признаков
        use_cls_token (bool): Использовать ли CLS токен
        
    Returns:
        ViTSemantic: Модель ViT Semantic
    """
    model = ViTSemantic(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.1,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        semantic_dim=semantic_dim,
        use_cls_token=use_cls_token
    )
    
    return model


# Вспомогательная функция для тестирования модели
def test_vit_semantic():
    """
    Тестирование модели ViT Semantic.
    """
    # Создаем модель
    model = create_vit_semantic(img_size=256, patch_size=16, in_chans=1)
    
    # Создаем входные данные
    x = torch.randn(2, 1, 256, 256)
    
    # Прямое распространение
    output = model(x)
    
    # Вывод информации о формах тензоров
    for k, v in output.items():
        print(f"{k}: {v.shape}")
        
    return output


if __name__ == "__main__":
    # При запуске файла как скрипта, тестируем модель
    test_vit_semantic()