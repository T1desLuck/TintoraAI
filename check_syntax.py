#!/usr/bin/env python3
"""
Простая проверка синтаксиса ключевых модулей TintoraAI
"""

import ast
import os
import sys

def check_file_syntax(filepath):
    """Проверяет синтаксис Python файла"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Проверяем синтаксис
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def main():
    """Основная функция проверки"""
    
    # Ключевые файлы для проверки
    key_files = [
        'core/swin_unet.py',
        'core/vit_semantic.py', 
        'core/fpn_pyramid.py',
        'core/cross_attention_bridge.py',
        'core/feature_fusion.py',
        'losses/patch_nce.py',
        'losses/vgg_perceptual.py',
        'losses/gan_loss.py',
        'losses/dynamic_balancer.py',
        'modules/discriminator.py',
        'modules/guide_net.py',
        'modules/style_transfer.py',
        'modules/memory_bank.py',
        'modules/uncertainty_estimation.py',
        'modules/few_shot_adapter.py'
    ]
    
    print("=== Проверка синтаксиса ключевых модулей TintoraAI ===\n")
    
    all_passed = True
    
    for filepath in key_files:
        if os.path.exists(filepath):
            success, error = check_file_syntax(filepath)
            if success:
                print(f"[OK] {filepath}")
            else:
                print(f"[ERROR] {filepath} - {error}")
                all_passed = False
        else:
            print(f"[MISSING] {filepath}")
            all_passed = False
    
    print(f"\n=== Result: {'ALL FILES OK' if all_passed else 'ERRORS FOUND'} ===")
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
