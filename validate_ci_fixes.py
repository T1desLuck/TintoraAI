#!/usr/bin/env python3
"""
Скрипт для валидации исправлений CI/CD ошибок TintoraAI.
Проверяет все ключевые сигнатуры и методы на совместимость с тестами.
"""

import torch
import torch.nn as nn
import sys
import traceback
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

def test_swin_unet():
    """Тест SwinUNet с исправленными сигнатурами."""
    print("Testing SwinUNet...")
    try:
        from core.swin_unet import SwinUNet
        
        # Тест с альтернативными параметрами
        model = SwinUNet(
            in_channels=1,
            out_channels=3,
            img_size=224,
            patch_size=4,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24]
        )
        
        # Тест forward
        x = torch.randn(1, 1, 224, 224)
        output = model(x)
        assert output.shape == (1, 3, 224, 224), f"Неверная форма выхода: {output.shape}"
        
        print("✓ SwinUNet OK")
        return True
    except Exception as e:
        print(f"✗ SwinUNet FAILED: {e}")
        traceback.print_exc()
        return False

def test_fpn_pyramid():
    """Тест FPNPyramid с исправленными сигнатурами."""
    print("Testing FPNPyramid...")
    try:
        from core.fpn_pyramid import FPNPyramid
        
        # Тест с альтернативными параметрами
        model = FPNPyramid(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256,
            use_pyramid_pooling=True
        )
        
        # Тест forward - должен возвращать tensor по умолчанию
        features = [
            torch.randn(1, 256, 56, 56),
            torch.randn(1, 512, 28, 28),
            torch.randn(1, 1024, 14, 14),
            torch.randn(1, 2048, 7, 7)
        ]
        
        output = model(features)
        assert isinstance(output, torch.Tensor), f"Ожидался tensor, получен {type(output)}"
        
        # Тест с return_dict=True
        output_dict = model(features, return_dict=True)
        assert isinstance(output_dict, dict), f"Ожидался dict, получен {type(output_dict)}"
        
        print("✓ FPNPyramid OK")
        return True
    except Exception as e:
        print(f"✗ FPNPyramid FAILED: {e}")
        traceback.print_exc()
        return False

def test_cross_attention_bridge():
    """Тест CrossAttentionBridge с исправленными сигнатурами."""
    print("Testing CrossAttentionBridge...")
    try:
        from core.cross_attention_bridge import CrossAttentionBridge
        
        model = CrossAttentionBridge(
            swin_dim=768,
            vit_dim=768,
            hidden_dim=512
        )
        
        # Тест forward с автоматическим определением разрешений
        swin_features = torch.randn(1, 768, 14, 14)
        vit_features = torch.randn(1, 768, 14, 14)
        
        output = model(swin_features, vit_features)
        assert len(output) == 2, f"Ожидалось 2 выхода, получено {len(output)}"
        
        print("✓ CrossAttentionBridge OK")
        return True
    except Exception as e:
        print(f"✗ CrossAttentionBridge FAILED: {e}")
        traceback.print_exc()
        return False

def test_multi_head_feature_fusion():
    """Тест MultiHeadFeatureFusion с исправленными сигнатурами."""
    print("Testing MultiHeadFeatureFusion...")
    try:
        from core.feature_fusion import MultiHeadFeatureFusion
        
        model = MultiHeadFeatureFusion(
            sources=['source_0', 'source_1'],
            in_channels_list=[256, 512],
            out_channels=256
        )
        
        # Тест с list (должен конвертироваться в dict)
        features_list = [
            torch.randn(1, 256, 14, 14),
            torch.randn(1, 512, 14, 14)
        ]
        
        output = model(features_list)
        assert isinstance(output, dict), f"Ожидался dict, получен {type(output)}"
        
        print("✓ MultiHeadFeatureFusion OK")
        return True
    except Exception as e:
        print(f"✗ MultiHeadFeatureFusion FAILED: {e}")
        traceback.print_exc()
        return False

def test_patch_nce_loss():
    """Тест PatchNCELoss с исправленными сигнатурами."""
    print("Testing PatchNCELoss...")
    try:
        from losses.patch_nce import PatchNCELoss
        
        loss_fn = PatchNCELoss(
            nce_weight=1.0,
            gradient_weight=0.1,
            temperature=0.07
        )
        
        # Тест forward с 3 параметрами
        query = torch.randn(1, 256, 32, 32)
        key = torch.randn(1, 256, 32, 32)
        reference = torch.randn(1, 256, 32, 32)
        
        loss = loss_fn(query, key, reference)
        assert isinstance(loss, torch.Tensor), f"Ожидался tensor, получен {type(loss)}"
        
        print("✓ PatchNCELoss OK")
        return True
    except Exception as e:
        print(f"✗ PatchNCELoss FAILED: {e}")
        traceback.print_exc()
        return False

def test_dynamic_loss_balancer():
    """Тест DynamicLossBalancer с исправленными сигнатурами."""
    print("Testing DynamicLossBalancer...")
    try:
        from losses.dynamic_balancer import DynamicLossBalancer
        
        # Тест с автоматическими loss_names
        balancer = DynamicLossBalancer(
            strategy='adaptive',
            num_losses=4
        )
        
        # Тест с явными loss_names
        balancer2 = DynamicLossBalancer(
            loss_names=['nce', 'perceptual', 'gan', 'consistency']
        )
        
        print("✓ DynamicLossBalancer OK")
        return True
    except Exception as e:
        print(f"✗ DynamicLossBalancer FAILED: {e}")
        traceback.print_exc()
        return False

def test_modules():
    """Тест всех модулей с исправленными сигнатурами."""
    print("Testing Modules...")
    
    results = []
    
    # GuideNet
    try:
        from modules.guide_net import GuideNet
        guide_net = GuideNet(
            input_dim=512,
            hidden_dim=256,
            num_layers=3,
            feature_dim=512
        )
        print("✓ GuideNet OK")
        results.append(True)
    except Exception as e:
        print(f"✗ GuideNet FAILED: {e}")
        results.append(False)
    
    # Discriminator
    try:
        from modules.discriminator import MotivationalDiscriminator
        discriminator = MotivationalDiscriminator(
            input_nc=3,
            ndf=64,
            reward_type='binary'
        )
        print("✓ MotivationalDiscriminator OK")
        results.append(True)
    except Exception as e:
        print(f"✗ MotivationalDiscriminator FAILED: {e}")
        results.append(False)
    
    # StyleTransferModule
    try:
        from modules.style_transfer import StyleTransferModule
        style_module = StyleTransferModule(
            input_channels=3,
            content_weight=1.0,
            style_weight=1.0,
            content_layers=['conv_4']
        )
        print("✓ StyleTransferModule OK")
        results.append(True)
    except Exception as e:
        print(f"✗ StyleTransferModule FAILED: {e}")
        results.append(False)
    
    # UncertaintyEstimationModule
    try:
        from modules.uncertainty_estimation import UncertaintyEstimationModule
        uncertainty_module = UncertaintyEstimationModule(
            num_samples=10,
            method='mc_dropout'
        )
        print("✓ UncertaintyEstimationModule OK")
        results.append(True)
    except Exception as e:
        print(f"✗ UncertaintyEstimationModule FAILED: {e}")
        results.append(False)
    
    # AdaptableColorizer
    try:
        from modules.few_shot_adapter import AdaptableColorizer
        # Создаем фиктивный базовый колоризатор
        base_model = nn.Sequential(nn.Conv2d(1, 3, 3, padding=1))
        adapter = AdaptableColorizer(
            base_colorizer=base_model,
            bottleneck_dim=64,
            base_model=base_model
        )
        print("✓ AdaptableColorizer OK")
        results.append(True)
    except Exception as e:
        print(f"✗ AdaptableColorizer FAILED: {e}")
        results.append(False)
    
    # MemoryBankModule
    try:
        from modules.memory_bank import MemoryBankModule
        memory_bank = MemoryBankModule(
            feature_dim=512,
            color_channels=3
        )
        
        # Тест add_item с 3 параметрами
        item = torch.randn(512)
        memory_bank.add_item(item, label='test', metadata={'quality': 0.8})
        print("✓ MemoryBankModule OK")
        results.append(True)
    except Exception as e:
        print(f"✗ MemoryBankModule FAILED: {e}")
        results.append(False)
    
    return all(results)

def main():
    """Основная функция тестирования."""
    print("=== Валидация исправлений CI/CD TintoraAI ===\n")
    
    tests = [
        test_swin_unet,
        test_fpn_pyramid,
        test_cross_attention_bridge,
        test_multi_head_feature_fusion,
        test_patch_nce_loss,
        test_dynamic_loss_balancer,
        test_modules
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} CRASHED: {e}")
            results.append(False)
        print()
    
    passed = sum(results)
    total = len(results)
    
    print(f"=== Результаты: {passed}/{total} тестов прошли ===")
    
    if passed == total:
        print("🎉 Все исправления CI/CD работают корректно!")
        return 0
    else:
        print("❌ Некоторые исправления требуют дополнительной работы.")
        return 1

if __name__ == "__main__":
    exit(main())
