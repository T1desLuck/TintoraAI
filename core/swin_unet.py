"""
Swin-UNet: Основной backbone колоризатора с встроенным attention механизмом.

Данный модуль объединяет архитектуру Swin Transformer с U-Net для эффективной
колоризации изображений. Swin-UNet использует окна с локальным вниманием (shifted windows)
для обработки изображений с высоким разрешением и имеет кодер-декодер структуру.

Основные компоненты:
- Swin Transformer Blocks: Блоки с механизмом внимания со сдвинутыми окнами
- Patch Embedding: Преобразование входного изображения в embedding
- Patch Merging/Expanding: Изменение разрешения feature maps
- Skip Connections: Соединения между энкодером и декодером
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from einops import rearrange
from timm.layers.drop import DropPath
from timm.layers.initialization import trunc_normal_


class MLP(nn.Module):
    """
    Многослойный персептрон с двумя линейными слоями и активацией GELU.
    
    Args:
        dim (int): Размерность входного вектора
        mlp_ratio (float): Множитель для скрытой размерности
        dropout (float): Вероятность dropout
    """
    def __init__(self, dim, mlp_ratio=4., dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(int(dim * mlp_ratio), dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """Прямое распространение через MLP."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Разбивает изображение на окна фиксированного размера.
    
    Args:
        x: (B, H, W, C)
        window_size (int): Размер окна
    
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Обратное преобразование окон в изображение.
    
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Размер окна
        H (int): Высота изображения
        W (int): Ширина изображения
    
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    Многоголовое внимание на основе окон с относительным позиционным кодированием.
    
    Args:
        dim (int): Размерность входного вектора
        window_size (tuple[int]): Размер окна
        num_heads (int): Число голов внимания
        qkv_bias (bool): Использовать ли смещение в QKV проекции
        qk_scale (float): Масштабирование для QK произведения
        attn_drop (float): Вероятность dropout для внимания
        proj_drop (float): Вероятность dropout для проекции
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Параметры для относительного позиционного кодирования
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  
        
        # Инициализация индексов для относительного позиционного кодирования
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        
        # Вычисление относительных позиционных индексов
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # Смещение для положительных индексов
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # Слои внимания
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Прямое распространение через блок внимания.
        
        Args:
            x (tensor): Входной тензор формы [num_windows*B, N, C]
            mask (tensor): Маска внимания формы [nW, Mh*Mw, Mh*Mw] или None
        """
        B_, N, C = x.shape
        
        # Проекция QKV
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C/nH
        
        # Масштабирование dot-product внимания
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B_, nH, N, N
        
        # Добавление относительного позиционного смещения
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        
        # Применение маски внимания, если она предоставлена
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
            
        attn = self.attn_drop(attn)
        
        # Взвешенная сумма
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Блок Swin Transformer с окнами для многоголового самовнимания.
    
    Args:
        dim (int): Размерность входного вектора
        input_resolution (tuple[int]): Входное разрешение
        num_heads (int): Число голов внимания
        window_size (int): Размер окна
        shift_size (int): Размер сдвига окна
        mlp_ratio (float): Соотношение размерности MLP к dim
        qkv_bias (bool): Использовать смещение в QKV проекции
        qk_scale (float): Масштабирование QK произведения
        drop (float): Вероятность dropout
        attn_drop (float): Вероятность dropout для внимания
        drop_path (float): Вероятность DropPath
        act_layer (nn.Module): Слой активации
        norm_layer (nn.Module): Слой нормализации
        use_checkpoint (bool): Использовать ли checkpointing
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        
        # Проверка на корректность размеров сдвига и окна
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must be between 0 and window_size"
        
        # Слои нормализации и блок внимания
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim=dim, mlp_ratio=mlp_ratio, dropout=drop)
        
        # Создание маски внимания для SW-MSA
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
                    
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
            
        self.register_buffer("attn_mask", attn_mask)
        
    def forward(self, x):
        """Прямое распространение через Swin Transformer блок."""
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"Input feature size ({L}) doesn't match resolution ({H}*{W})"
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            
        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # W-MSA/SW-MSA
        if self.use_checkpoint:
            attn_windows = checkpoint.checkpoint(self.attn, x_windows, self.attn_mask)
        else:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
            
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            
        x = x.view(B, H * W, C)
        
        # FFN (Feed-Forward Network)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class PatchEmbed(nn.Module):
    """
    Преобразование изображения в патчи и их embedding.
    
    Args:
        img_size (int): Размер входного изображения
        patch_size (int): Размер патча
        in_chans (int): Число входных каналов
        embed_dim (int): Размерность embedding вектора
        norm_layer (nn.Module): Слой нормализации
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
    def forward(self, x):
        """Прямое распространение: преобразование изображения в embedding."""
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            
        # Проекция патчей
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    """
    Слой объединения патчей для уменьшения разрешения feature map.
    
    Args:
        input_resolution (tuple[int]): Разрешение входа
        dim (int): Размерность входного вектора
        norm_layer (nn.Module): Слой нормализации
    """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
        
    def forward(self, x):
        """Прямое распространение через слой объединения патчей."""
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"Input feature size ({L}) doesn't match resolution ({H}*{W})"
        
        x = x.view(B, H, W, C)
        
        # Объединение соседних патчей
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        
        x = self.norm(x)
        x = self.reduction(x)  # B H/2*W/2 2*C
        
        return x


class PatchExpanding(nn.Module):
    """
    Слой расширения патчей для увеличения разрешения feature map.
    
    Args:
        input_resolution (tuple[int]): Разрешение входа
        dim (int): Размерность входного вектора
        dim_scale (int): Масштаб расширения размерности
        norm_layer (nn.Module): Слой нормализации
    """
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim)
        
    def forward(self, x):
        """Прямое распространение через слой расширения патчей."""
        H, W = self.input_resolution
        x = self.norm(x)
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, f"Input feature size ({L}) doesn't match resolution ({H}*{W})"
        
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B, -1, C//4)
        return x


class FinalPatchExpand_X4(nn.Module):
    """
    Финальное расширение патчей для увеличения разрешения в 4 раза.
    
    Args:
        input_resolution (tuple[int]): Разрешение входа
        dim (int): Размерность входного вектора
        dim_scale (int): Масштаб расширения размерности
        norm_layer (nn.Module): Слой нормализации
    """
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, dim_scale * dim_scale * dim, bias=False)
        self.norm = norm_layer(dim)
        
    def forward(self, x):
        """Прямое распространение через финальный слой расширения патчей."""
        H, W = self.input_resolution
        x = self.norm(x)
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, f"Input feature size ({L}) doesn't match resolution ({H}*{W})"
        
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', 
                     p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B, -1, C//(self.dim_scale**2))
        return x


class BasicLayer(nn.Module):
    """
    Базовый слой Swin Transformer для одного этапа.
    
    Args:
        dim (int): Размерность входного вектора
        input_resolution (tuple[int]): Входное разрешение
        depth (int): Число блоков Swin Transformer
        num_heads (int): Число голов внимания
        window_size (int): Размер окна
        mlp_ratio (float): Соотношение размерности MLP к dim
        qkv_bias (bool): Использовать смещение в QKV проекции
        qk_scale (float): Масштабирование QK произведения
        drop (float): Вероятность dropout
        attn_drop (float): Вероятность dropout для внимания
        drop_path (float/list): Вероятность DropPath
        norm_layer (nn.Module): Слой нормализации
        downsample (nn.Module): Слой даунсемплинга в конце слоя
        use_checkpoint (bool): Использовать ли checkpointing
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        
        # Строим блоки Swin Transformer
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads, 
                window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for i in range(depth)
        ])
        
        # Слой даунсемплинга (при наличии)
        self.downsample = downsample
        
    def forward(self, x):
        """Прямое распространение через базовый слой."""
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        
        if self.downsample is not None:
            x = self.downsample(x)
            
        return x


class BasicLayer_up(nn.Module):
    """
    Базовый слой для апсемплинга в декодере Swin-UNet.
    
    Args:
        dim (int): Размерность входного вектора
        input_resolution (tuple[int]): Входное разрешение
        depth (int): Число блоков Swin Transformer
        num_heads (int): Число голов внимания
        window_size (int): Размер окна
        mlp_ratio (float): Соотношение размерности MLP к dim
        qkv_bias (bool): Использовать смещение в QKV проекции
        qk_scale (float): Масштабирование QK произведения
        drop (float): Вероятность dropout
        attn_drop (float): Вероятность dropout для внимания
        drop_path (float/list): Вероятность DropPath
        norm_layer (nn.Module): Слой нормализации
        upsample (nn.Module): Слой апсемплинга в конце слоя
        use_checkpoint (bool): Использовать ли checkpointing
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        
        # Строим блоки Swin Transformer
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads, 
                window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for i in range(depth)
        ])
        
        # Слой апсемплинга (при наличии)
        self.upsample = upsample
        
    def forward(self, x):
        """Прямое распространение через слой апсемплинга."""
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        
        if self.upsample is not None:
            x = self.upsample(x)
            
        return x


class SwinUNet(nn.Module):
    """
    Swin-UNet: Архитектура, объединяющая Swin Transformer и U-Net для задачи колоризации.
    
    Args:
        img_size (int): Размер входного изображения
        patch_size (int): Размер патча
        in_chans (int): Число входных каналов (обычно 1 для ЧБ изображения)
        out_chans (int): Число выходных каналов (обычно 2 для ab в Lab цветовом пространстве)
        embed_dim (int): Базовая размерность embedding вектора
        depths (tuple[int]): Глубина каждого слоя Swin Transformer
        num_heads (tuple[int]): Число голов внимания для каждого слоя
        window_size (int): Размер окна для внимания
        mlp_ratio (float): Соотношение размерности MLP к dim
        qkv_bias (bool): Использовать смещение в QKV проекции
        qk_scale (float): Масштабирование QK произведения
        drop_rate (float): Вероятность dropout
        attn_drop_rate (float): Вероятность dropout для внимания
        drop_path_rate (float): Вероятность DropPath
        norm_layer (nn.Module): Слой нормализации
        patch_norm (bool): Применять ли нормализацию после проекции патча
        use_checkpoint (bool): Использовать ли checkpointing для экономии памяти
        final_upsample (str): Метод финального апсемплинга ('expand' или 'interpolate')
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=1, out_chans=2, 
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand"):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.final_upsample = final_upsample
        self.mlp_ratio = mlp_ratio
        
        # Преобразование изображения в патчи и embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, 
            embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        
        # Абсолютное позиционное кодирование
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # Stochastic depth decay rule
        
        # Строим энкодер
        self.layers_down = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                 patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers_down.append(layer)
        
        # Строим декодер
        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer_up(
                dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)),
                input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                                 patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))),
                depth=depths[(self.num_layers-1-i_layer)],
                num_heads=num_heads[(self.num_layers-1-i_layer)],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:(self.num_layers-1-i_layer)]):sum(depths[:(self.num_layers-1-i_layer) + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpanding if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers_up.append(layer)
        
        # Финальная проекция после декодера
        if self.final_upsample == "expand":
            self.up = FinalPatchExpand_X4(
                input_resolution=(img_size // patch_size, img_size // patch_size),
                dim=embed_dim,
                dim_scale=self.patch_size, 
                norm_layer=norm_layer
            )
            self.output = nn.Sequential(
                nn.Linear(embed_dim, out_chans),
            )
        
        # Инициализация весов
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Инициализация весов для модели."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        """
        Прямое распространение через Swin-UNet.
        
        Args:
            x (tensor): Входное изображение в формате (B, C, H, W)
            
        Returns:
            tensor: Выходное изображение в формате (B, out_chans, H, W)
        """
        # Embedding патчей
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        
        # Сохраняем промежуточные features для skip connections
        features = []
        
        # Энкодер (downsample)
        for layer in self.layers_down:
            features.append(x)
            x = layer(x)
        
        # Декодер (upsample) с skip connections
        for i, layer in enumerate(self.layers_up):
            if i > 0:  # Skip connection начиная со второго слоя декодера
                x = x + features[self.num_layers - 1 - i]
            x = layer(x)
        
        # Финальный слой upsample
        if self.final_upsample == "expand":
            x = self.up(x)
            x = self.output(x)
            # Преобразование формата (B, L, C) -> (B, C, H, W)
            B, L, C = x.shape
            H = W = int(L ** 0.5)
            x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        return x


# Функция для создания SwinUNet с предопределенными параметрами
def create_colorizer_swin_unet(img_size=256, in_chans=1, out_chans=2, embed_dim=96,
                              window_size=8, depths=[2, 2, 8, 2], num_heads=[3, 6, 12, 24],
                              use_checkpoint=False, final_upsample="expand"):
    """
    Создает Swin-UNet модель для колоризации изображений.
    
    Args:
        img_size (int): Размер входного изображения
        in_chans (int): Число входных каналов (обычно 1 для ЧБ изображения)
        out_chans (int): Число выходных каналов (обычно 2 для ab в Lab цветовом пространстве)
        embed_dim (int): Базовая размерность embedding вектора
        window_size (int): Размер окна для внимания
        depths (tuple[int]): Глубина каждого слоя Swin Transformer
        num_heads (tuple[int]): Число голов внимания для каждого слоя
        use_checkpoint (bool): Использовать ли checkpointing для экономии памяти
        final_upsample (str): Метод финального апсемплинга ('expand' или 'interpolate')
        
    Returns:
        nn.Module: Swin-UNet модель для колоризации
    """
    model = SwinUNet(
        img_size=img_size,
        patch_size=4,
        in_chans=in_chans,
        out_chans=out_chans,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=use_checkpoint,
        final_upsample=final_upsample
    )
    
    return model


if __name__ == "__main__":
    # Пример использования
    model = create_colorizer_swin_unet()
    print("Модель создана успешно!")
    
    # Тестовый вход
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    print(f"Вход: {x.shape}, Выход: {y.shape}")