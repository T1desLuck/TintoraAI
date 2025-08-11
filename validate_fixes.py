#!/usr/bin/env python3
"""
Скрипт для валидации исправлений сигнатур классов TintoraAI.

Этот скрипт проверяет, что все исправленные классы корректно инициализируются
с альтернативными параметрами, которые ожидают CI/CD тесты.
"""

import sys
import os
import traceback
from typing import Dict, List, Any

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_class_initialization(class_module: str, class_name: str, test_params: Dict[str, Any]) -> bool:
    """
    Тестирует инициализацию класса с заданными параметрами.
    
    Args:
        class_module: Модуль класса
        class_name: Имя класса
        test_params: Параметры для тестирования
        
    Returns:
        bool: True если инициализация прошла успешно
    """
    try:
        # Импортируем модуль
        module = __import__(class_module, fromlist=[class_name])
        cls = getattr(module, class_name)
        
        # Пробуем инициализировать с тестовыми параметрами
        instance = cls(**test_params)
        
        print(f"✓ {class_name} успешно инициализирован с параметрами: {list(test_params.keys())}")
        return True
        
    except Exception as e:
        print(f"✗ {class_name} не удалось инициализировать: {str(e)}")
        print(f"  Параметры: {test_params}")
        traceback.print_exc()
        return False

def main():
    """Основная функция для проверки всех исправленных классов."""
    
    print("=== Валидация исправлений сигнатур TintoraAI ===\n")
    
    # Список тестов для проверки
    tests = [
        # Core модули
        {
            'module': 'core.swin_unet',
            'class': 'SwinUNet',
            'params': {
                'img_size': 224,
                'in_channels': 1,  # альтернативный параметр
                'out_channels': 2,  # альтернативный параметр
                'patch_size': 4,
                'embed_dim': 96
            }
        },
        {
            'module': 'core.vit_semantic',
            'class': 'ViTSemantic',
            'params': {
                'img_size': 224,
                'in_channels': 1,  # альтернативный параметр
                'patch_size': 16,
                'embed_dim': 768
            }
        },
        {
            'module': 'core.fpn_pyramid',
            'class': 'FPNPyramid',
            'params': {
                'in_channels': [256, 512, 1024, 2048],  # альтернативный параметр
                'out_channels': 256,  # альтернативный параметр
                'num_levels': 4
            }
        },
        
        # Модули
        {
            'module': 'modules.guide_net',
            'class': 'GuideNet',
            'params': {
                'semantic_dim': 768,
                'color_dim': 256,
                'in_channels': 1,  # альтернативный параметр
                'advice_channels': 256,  # альтернативный параметр
                'device': 'cpu'  # альтернативный параметр
            }
        },
        {
            'module': 'modules.discriminator',
            'class': 'MotivationalDiscriminator',
            'params': {
                'input_nc': 3,
                'ndf': 64,
                'in_channels': 3,  # альтернативный параметр
                'device': 'cpu'  # альтернативный параметр
            }
        },
        {
            'module': 'modules.style_transfer',
            'class': 'StyleTransferModule',
            'params': {
                'content_dim': 512,
                'style_dim': 512,
                'in_channels': 512,  # альтернативный параметр
                'out_channels': 512,  # альтернативный параметр
                'device': 'cpu'  # альтернативный параметр
            }
        },
        {
            'module': 'modules.memory_bank',
            'class': 'MemoryBankModule',
            'params': {
                'feature_dim': 256,
                'memory_size': 1000,
                'device': 'cpu'  # альтернативный параметр
            }
        },
        {
            'module': 'modules.few_shot_adapter',
            'class': 'AdaptableColorizer',
            'params': {
                'base_model_dim': 512,
                'num_classes': 10,
                'adapter_type': 'linear',  # альтернативный параметр
                'device': 'cpu'  # альтернативный параметр
            }
        },
        
        # Функции потерь
        {
            'module': 'losses.patch_nce',
            'class': 'PatchNCELoss',
            'params': {
                'nce_layers': [0, 4, 8, 12, 16],
                'temperature': 0.07,
                'nce_includes_all_negatives_from_minibatch': False
            }
        },
        {
            'module': 'losses.vgg_perceptual',
            'class': 'VGGPerceptualLoss',
            'params': {
                'layers': ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'],
                'weights': [1.0, 1.0, 1.0, 1.0]
            }
        },
        {
            'module': 'losses.gan_loss',
            'class': 'GANLoss',
            'params': {
                'gan_mode': 'lsgan',
                'target_real_label': 1.0,
                'target_fake_label': 0.0,
                'device': 'cpu'
            }
        },
        {
            'module': 'losses.dynamic_balancer',
            'class': 'DynamicLossBalancer',
            'params': {
                'loss_names': ['pixel', 'perceptual', 'gan'],
                'initial_weights': [1.0, 1.0, 1.0],
                'strategy': 'adaptive'
            }
        }
    ]
    
    # Запускаем тесты
    passed = 0
    failed = 0
    
    for test in tests:
        print(f"Тестирование {test['class']}...")
        if test_class_initialization(test['module'], test['class'], test['params']):
            passed += 1
        else:
            failed += 1
        print()
    
    # Выводим результаты
    print("=== Результаты валидации ===")
    print(f"Успешно: {passed}")
    print(f"Ошибки: {failed}")
    print(f"Всего: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 Все тесты прошли успешно! Исправления сигнатур работают корректно.")
        return True
    else:
        print(f"\n❌ {failed} тестов не прошли. Требуются дополнительные исправления.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
