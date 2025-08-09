"""
TintoraAI Core: Основные компоненты архитектуры колоризатора.

Данный модуль объединяет все ключевые компоненты архитектуры колоризатора TintoraAI:
- Swin-UNet: Основной backbone с встроенным механизмом внимания
- ViT Semantic: Семантический блок на основе Vision Transformer для понимания контекста
- FPN Pyramid: Feature Pyramid Network с Pyramid Pooling для мультимасштабного восприятия
- Cross-Attention Bridge: Модуль взаимодействия между Swin-UNet и ViT
- Feature Fusion: Интеллектуальное слияние признаков с весами внимания

Модули спроектированы для максимальной гибкости и производительности,
позволяя колоризатору эффективно обрабатывать изображения разных типов и стилей.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Импортируем все основные компоненты
from .swin_unet import SwinUNet, create_colorizer_swin_unet
from .vit_semantic import ViTSemantic, create_vit_semantic
from .fpn_pyramid import FPNPyramid, DynamicFPNPyramid, create_fpn_pyramid, create_dynamic_fpn_pyramid
from .cross_attention_bridge import CrossAttentionBridge, create_cross_attention_bridge
from .feature_fusion import MultiHeadFeatureFusion, MultiModalFeatureFusion, create_feature_fusion


class ColorizerBackbone(nn.Module):
    """
    Основной backbone колоризатора, объединяющий все компоненты в единую архитектуру.
    
    Args:
        img_size (int): Размер входного изображения
        in_channels (int): Количество каналов входного изображения (обычно 1 для ЧБ)
        out_channels (int): Количество каналов выходного изображения (обычно 2 для ab в Lab)
        swin_embed_dim (int): Размерность embedding для Swin-UNet
        swin_depths (list): Глубина каждого слоя Swin Transformer
        swin_num_heads (list): Количество голов внимания для каждого слоя
        vit_embed_dim (int): Размерность embedding для ViT
        vit_depth (int): Глубина ViT (количество блоков Transformer)
        vit_num_heads (int): Количество голов внимания для ViT
        fpn_channels (int): Количество каналов для FPN
        bridge_fusion_dim (int): Размерность для слияния в Cross-Attention Bridge
        fusion_dim (int): Размерность для финального слияния признаков
        use_dynamic_fpn (bool): Использовать ли динамический FPN
    """
    def __init__(
        self,
        img_size=256,
        in_channels=1,
        out_channels=2,
        swin_embed_dim=96,
        swin_depths=[2, 2, 6, 2],
        swin_num_heads=[3, 6, 12, 24],
        vit_embed_dim=768,
        vit_depth=12,
        vit_num_heads=12,
        fpn_channels=256,
        bridge_fusion_dim=512,
        fusion_dim=256,
        use_dynamic_fpn=True
    ):
        super(ColorizerBackbone, self).__init__()
        
        # Swin-UNet backbone
        self.swin_unet = create_colorizer_swin_unet(
            img_size=img_size,
            in_chans=in_channels,
            out_chans=out_channels,
            embed_dim=swin_embed_dim,
            depths=swin_depths,
            num_heads=swin_num_heads,
            return_intermediate=True
        )
        
        # ViT для семантического понимания
        self.vit_semantic = create_vit_semantic(
            img_size=img_size,
            patch_size=16,
            in_chans=in_channels,
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_num_heads
        )
        
        # FPN для мультимасштабного восприятия
        swin_channels = [swin_embed_dim * 2**i for i in range(len(swin_depths))]
        
        if use_dynamic_fpn:
            self.fpn_pyramid = create_dynamic_fpn_pyramid(
                backbone_channels=swin_channels,
                fpn_channels=fpn_channels,
                output_channels=fpn_channels // 2
            )
        else:
            self.fpn_pyramid = create_fpn_pyramid(
                backbone_channels=swin_channels,
                fpn_channels=fpn_channels,
                output_channels=fpn_channels // 2
            )
            
        # Cross-Attention Bridge между Swin-UNet и ViT
        self.cross_bridge = create_cross_attention_bridge(
            swin_dims=swin_channels,
            vit_dim=vit_embed_dim,
            fusion_dim=bridge_fusion_dim
        )
        
        # Feature Fusion для слияния признаков
        self.feature_fusion = create_feature_fusion(
            spatial_dims={
                "swin_unet": (swin_channels[0], img_size // 4, img_size // 4),
                "fpn": (fpn_channels // 2, img_size // 4, img_size // 4)
            },
            sequential_dims={
                "vit": (img_size // 16 * img_size // 16, vit_embed_dim)
            },
            fusion_dim=fusion_dim
        )
        
        # Финальная проекция для выходного изображения
        self.output_conv = nn.Conv2d(fusion_dim, out_channels, kernel_size=1)
        
    def forward(self, x):
        """
        Прямое распространение через backbone колоризатора.
        
        Args:
            x (torch.Tensor): Входное ЧБ изображение [B, 1, H, W]
            
        Returns:
            dict: {
                'output': torch.Tensor,  # Финальный выход (ab каналы) [B, 2, H, W]
                'swin_features': torch.Tensor,  # Признаки из Swin-UNet
                'vit_features': dict,  # Признаки из ViT
                'fpn_features': dict,  # Признаки из FPN
                'fused_features': torch.Tensor  # Слитые признаки
            }
        """
        batch_size, _, height, width = x.shape
        
        # Проход через Swin-UNet с извлечением промежуточных признаков и финального выхода
        swin_out = self.swin_unet(x)
        assert isinstance(swin_out, (list, tuple)) and len(swin_out) >= 5, "Ожидается 4 уровня признаков и финальный выход от SwinUNet"
        swin_features_2d = list(swin_out[:-1])  # список из 4 карт признаков [B,C_i,H_i,W_i]
        swin_final = swin_out[-1]               # [B,out_channels,H,W]
        
        # Проход через ViT
        vit_output = self.vit_semantic(x)
        vit_features = vit_output['enriched_features']  # [B, N_vit, C_vit]
        
        # Проход через FPN с промежуточными признаками из Swin-UNet
        fpn_output = self.fpn_pyramid(swin_features_2d)
        
        # Cross-Attention Bridge между Swin-UNet и ViT
        # Подготовим списки последовательностей для моста: [B, H_i*W_i, C_i]
        swin_features_seq = []
        swin_base_h = swin_features_2d[0].shape[2]
        swin_base_w = swin_features_2d[0].shape[3]
        for feat in swin_features_2d:
            B, C, H_i, W_i = feat.shape
            swin_features_seq.append(feat.permute(0, 2, 3, 1).contiguous().view(B, H_i * W_i, C))

        vit_patch_res = (height // 16, width // 16)
        enhanced_swin_seq, enhanced_vit_seq = self.cross_bridge(
            swin_features_list=swin_features_seq,
            vit_features=vit_features,
            swin_resolution=(swin_base_h, swin_base_w),
            vit_resolution=vit_patch_res
        )
        # Преобразуем улучшенные признаки Swin обратно в 2D карту базового разрешения
        B = x.shape[0]
        C_swin0 = swin_features_2d[0].shape[1]
        enhanced_swin_2d = enhanced_swin_seq.view(B, swin_base_h, swin_base_w, C_swin0).permute(0, 3, 1, 2).contiguous()
        
        # Объединяем все признаки с помощью Feature Fusion
        fusion_input = {
            "swin_unet": enhanced_swin_2d,
            "fpn": fpn_output['output'],
            "vit": enhanced_vit_seq
        }
        fusion_output = self.feature_fusion(fusion_input)
        # Используем пространственные слитые признаки как вход к финальной проекции
        spatial_features = fusion_output["spatial_features"]
        
        # Финальная проекция для получения ab каналов
        output = self.output_conv(spatial_features)
        
        # Апсемплинг до исходного разрешения
        output = F.interpolate(output, size=(height, width), mode='bilinear', align_corners=True)
        
        return {
            'output': output,
            'swin_features': enhanced_swin_2d,
            'vit_features': vit_output,
            'fpn_features': fpn_output,
            'fused_features': fusion_output.get("fused_features")
        }


# Функция для создания полного колоризатора
def create_colorizer(config=None):
    """
    Создает полную модель колоризатора на основе конфигурации.
    
    Args:
        config (dict, optional): Словарь с параметрами конфигурации
        
    Returns:
        nn.Module: Модель колоризатора
    """
    # Параметры по умолчанию
    default_config = {
        'img_size': 256,
        'in_channels': 1,
        'out_channels': 2,
        'swin_embed_dim': 96,
        'swin_depths': [2, 2, 6, 2],
        'swin_num_heads': [3, 6, 12, 24],
        'vit_embed_dim': 768,
        'vit_depth': 12,
        'vit_num_heads': 12,
        'fpn_channels': 256,
        'bridge_fusion_dim': 512,
        'fusion_dim': 256,
        'use_dynamic_fpn': True
    }
    
    # Объединяем с пользовательской конфигурацией
    if config is not None:
        for key, value in config.items():
            if key in default_config:
                default_config[key] = value
    
    # Создаем модель с указанными параметрами
    model = ColorizerBackbone(**default_config)
    
    return model


# Экспортируем все основные компоненты
__all__ = [
    'SwinUNet',
    'create_colorizer_swin_unet',
    'ViTSemantic',
    'create_vit_semantic',
    'FPNPyramid',
    'DynamicFPNPyramid',
    'create_fpn_pyramid',
    'create_dynamic_fpn_pyramid',
    'CrossAttentionBridge',
    'create_cross_attention_bridge',
    'MultiHeadFeatureFusion',
    'MultiModalFeatureFusion',
    'create_feature_fusion',
    'ColorizerBackbone',
    'create_colorizer'
]
