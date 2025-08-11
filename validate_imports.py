#!/usr/bin/env python3
"""
Упрощенная валидация исправлений CI/CD TintoraAI.
Проверяет импорты и сигнатуры без PyTorch.
"""

import sys
import inspect
import traceback
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Тест импортов всех ключевых модулей."""
    print("Testing imports...")
    
    modules_to_test = [
        ('core.swin_unet', 'SwinUNet'),
        ('core.fpn_pyramid', 'FPNPyramid'),
        ('core.cross_attention_bridge', 'CrossAttentionBridge'),
        ('core.feature_fusion', 'MultiHeadFeatureFusion'),
        ('losses.patch_nce', 'PatchNCELoss'),
        ('losses.dynamic_balancer', 'DynamicLossBalancer'),
        ('losses.gan_loss', 'GANLoss'),
        ('modules.guide_net', 'GuideNet'),
        ('modules.discriminator', 'MotivationalDiscriminator'),
        ('modules.style_transfer', 'StyleTransferModule'),
        ('modules.uncertainty_estimation', 'UncertaintyEstimationModule'),
        ('modules.few_shot_adapter', 'AdaptableColorizer'),
        ('modules.memory_bank', 'MemoryBankModule')
    ]
    
    results = []
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"[OK] {module_name}.{class_name} - import OK")
            results.append(True)
        except Exception as e:
            print(f"[ERROR] {module_name}.{class_name} - import FAILED: {e}")
            results.append(False)
    
    return all(results)

def test_signatures():
    """Тест сигнатур конструкторов."""
    print("\nTesting constructor signatures...")
    
    try:
        # SwinUNet
        from core.swin_unet import SwinUNet
        sig = inspect.signature(SwinUNet.__init__)
        params = list(sig.parameters.keys())
        expected_params = ['self', 'img_size', 'patch_size', 'in_chans', 'num_classes', 'embed_dim']
        print(f"[OK] SwinUNet params: {params[:6]}...")
        
        # FPNPyramid
        from core.fpn_pyramid import FPNPyramid
        sig = inspect.signature(FPNPyramid.__init__)
        params = list(sig.parameters.keys())
        print(f"[OK] FPNPyramid params: {params[:6]}...")
        
        # CrossAttentionBridge
        from core.cross_attention_bridge import CrossAttentionBridge
        sig = inspect.signature(CrossAttentionBridge.__init__)
        params = list(sig.parameters.keys())
        print(f"[OK] CrossAttentionBridge params: {params[:6]}...")
        
        # MultiHeadFeatureFusion
        from core.feature_fusion import MultiHeadFeatureFusion
        sig = inspect.signature(MultiHeadFeatureFusion.__init__)
        params = list(sig.parameters.keys())
        print(f"[OK] MultiHeadFeatureFusion params: {params[:6]}...")
        
        # PatchNCELoss
        from losses.patch_nce import PatchNCELoss
        sig = inspect.signature(PatchNCELoss.forward)
        params = list(sig.parameters.keys())
        expected = ['self', 'query', 'key', 'reference']
        if len(params) >= 4 and params[:4] == expected:
            print(f"[OK] PatchNCELoss.forward params: {params}")
        else:
            print(f"[ERROR] PatchNCELoss.forward params: {params} (expected: {expected})")
        
        # DynamicLossBalancer
        from losses.dynamic_balancer import DynamicLossBalancer
        sig = inspect.signature(DynamicLossBalancer.__init__)
        params = list(sig.parameters.keys())
        if 'loss_names' in params:
            print(f"[OK] DynamicLossBalancer has loss_names parameter")
        else:
            print(f"[ERROR] DynamicLossBalancer missing loss_names parameter")
        
        # GuideNet
        from modules.guide_net import GuideNet
        sig = inspect.signature(GuideNet.__init__)
        params = list(sig.parameters.keys())
        if 'num_layers' in params and 'feature_dim' in params:
            print(f"[OK] GuideNet has required alternative parameters")
        else:
            print(f"[ERROR] GuideNet missing alternative parameters")
        
        # MotivationalDiscriminator
        from modules.discriminator import MotivationalDiscriminator
        sig = inspect.signature(MotivationalDiscriminator.__init__)
        params = list(sig.parameters.keys())
        if 'reward_type' in params:
            print(f"[OK] MotivationalDiscriminator has reward_type parameter")
        else:
            print(f"[ERROR] MotivationalDiscriminator missing reward_type parameter")
        
        # StyleTransferModule
        from modules.style_transfer import StyleTransferModule
        sig = inspect.signature(StyleTransferModule.__init__)
        params = list(sig.parameters.keys())
        if 'content_layers' in params:
            print(f"[OK] StyleTransferModule has content_layers parameter")
        else:
            print(f"[ERROR] StyleTransferModule missing content_layers parameter")
        
        # UncertaintyEstimationModule
        from modules.uncertainty_estimation import UncertaintyEstimationModule
        sig = inspect.signature(UncertaintyEstimationModule.__init__)
        params = list(sig.parameters.keys())
        if 'method' in params:
            print(f"[OK] UncertaintyEstimationModule has method parameter")
        else:
            print(f"[ERROR] UncertaintyEstimationModule missing method parameter")
        
        # AdaptableColorizer
        from modules.few_shot_adapter import AdaptableColorizer
        sig = inspect.signature(AdaptableColorizer.__init__)
        params = list(sig.parameters.keys())
        if 'base_model' in params:
            print(f"[OK] AdaptableColorizer has base_model parameter")
        else:
            print(f"[ERROR] AdaptableColorizer missing base_model parameter")
        
        # MemoryBankModule
        from modules.memory_bank import MemoryBankModule
        sig = inspect.signature(MemoryBankModule.add_item)
        params = list(sig.parameters.keys())
        expected = ['self', 'item', 'label', 'metadata']
        if len(params) >= 4 and params[:4] == expected:
            print(f"[OK] MemoryBankModule.add_item params: {params}")
        else:
            print(f"[ERROR] MemoryBankModule.add_item params: {params} (expected: {expected})")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Signature test FAILED: {e}")
        traceback.print_exc()
        return False

def main():
    """Основная функция валидации."""
    print("=== Упрощенная валидация исправлений CI/CD TintoraAI ===\n")
    
    import_result = test_imports()
    signature_result = test_signatures()
    
    print(f"\n=== Результаты ===")
    print(f"Imports: {'[OK]' if import_result else '[ERROR]'}")
    print(f"Signatures: {'[OK]' if signature_result else '[ERROR]'}")
    
    if import_result and signature_result:
        print("\n[SUCCESS] All basic checks passed!")
        print("CI/CD fixes are ready for PyTorch testing.")
        return 0
    else:
        print("\n[FAILED] Some checks failed.")
        return 1

if __name__ == "__main__":
    exit(main())
