#!/usr/bin/env python3
"""
Скрипт для запуска CI/CD тестов TintoraAI.

Этот скрипт запускает все тесты проекта и выводит подробные результаты,
включая информацию о прохождении тестов сигнатур классов.
"""

import sys
import os
import subprocess
import unittest
from typing import List, Dict, Any

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_unittest_suite(test_module: str) -> bool:
    """
    Запускает набор тестов unittest.
    
    Args:
        test_module: Модуль с тестами
        
    Returns:
        bool: True если все тесты прошли успешно
    """
    try:
        print(f"\n=== Запуск тестов {test_module} ===")
        
        # Импортируем модуль тестов
        module = __import__(test_module, fromlist=[''])
        
        # Создаем test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)
        
        # Запускаем тесты
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Возвращаем результат
        success = result.wasSuccessful()
        print(f"Результат: {'УСПЕХ' if success else 'ОШИБКА'}")
        
        if not success:
            print(f"Ошибки: {len(result.errors)}")
            print(f"Неудачи: {len(result.failures)}")
            
            # Выводим детали ошибок
            for test, error in result.errors:
                print(f"\nОШИБКА в {test}:")
                print(error)
                
            for test, failure in result.failures:
                print(f"\nНЕУДАЧА в {test}:")
                print(failure)
        
        return success
        
    except Exception as e:
        print(f"Ошибка при запуске тестов {test_module}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_pytest_tests(test_dir: str = "tests") -> bool:
    """
    Запускает тесты через pytest.
    
    Args:
        test_dir: Директория с тестами
        
    Returns:
        bool: True если все тесты прошли успешно
    """
    try:
        print(f"\n=== Запуск pytest тестов из {test_dir} ===")
        
        # Запускаем pytest
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            test_dir, 
            '-v', 
            '--tb=short',
            '--no-header'
        ], capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        success = result.returncode == 0
        print(f"Результат pytest: {'УСПЕХ' if success else 'ОШИБКА'} (код: {result.returncode})")
        
        return success
        
    except Exception as e:
        print(f"Ошибка при запуске pytest: {str(e)}")
        return False

def check_imports() -> bool:
    """
    Проверяет импорты всех основных модулей.
    
    Returns:
        bool: True если все импорты успешны
    """
    print("\n=== Проверка импортов ===")
    
    modules_to_check = [
        'core.swin_unet',
        'core.vit_semantic', 
        'core.fpn_pyramid',
        'core.cross_attention_bridge',
        'core.feature_fusion',
        'modules.guide_net',
        'modules.discriminator',
        'modules.style_transfer',
        'modules.memory_bank',
        'modules.few_shot_adapter',
        'losses.patch_nce',
        'losses.vgg_perceptual',
        'losses.gan_loss',
        'losses.dynamic_balancer'
    ]
    
    failed_imports = []
    
    for module_name in modules_to_check:
        try:
            __import__(module_name)
            print(f"✓ {module_name}")
        except Exception as e:
            print(f"✗ {module_name}: {str(e)}")
            failed_imports.append((module_name, str(e)))
    
    if failed_imports:
        print(f"\nОшибки импорта ({len(failed_imports)}):")
        for module, error in failed_imports:
            print(f"  {module}: {error}")
        return False
    else:
        print(f"\nВсе импорты успешны ({len(modules_to_check)} модулей)")
        return True

def main():
    """Основная функция для запуска всех тестов."""
    
    print("=== CI/CD Тесты TintoraAI ===")
    print(f"Python версия: {sys.version}")
    print(f"Рабочая директория: {os.getcwd()}")
    
    results = []
    
    # 1. Проверка импортов
    import_success = check_imports()
    results.append(("Импорты", import_success))
    
    # 2. Запуск тестов валидации исправлений
    try:
        from validate_fixes import main as validate_main
        print("\n=== Запуск валидации исправлений ===")
        validation_success = validate_main()
        results.append(("Валидация исправлений", validation_success))
    except Exception as e:
        print(f"Ошибка при запуске валидации: {str(e)}")
        results.append(("Валидация исправлений", False))
    
    # 3. Запуск unittest тестов
    test_modules = [
        'tests.test_core',
        'tests.test_modules', 
        'tests.test_losses'
    ]
    
    for test_module in test_modules:
        try:
            success = run_unittest_suite(test_module)
            results.append((test_module, success))
        except ImportError:
            print(f"Модуль {test_module} не найден, пропускаем")
            continue
    
    # 4. Запуск pytest (если доступен)
    if os.path.exists('tests'):
        pytest_success = run_pytest_tests()
        results.append(("pytest", pytest_success))
    
    # Итоговые результаты
    print("\n" + "="*50)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("="*50)
    
    passed = 0
    failed = 0
    
    for test_name, success in results:
        status = "УСПЕХ" if success else "ОШИБКА"
        print(f"{test_name:30} {status}")
        if success:
            passed += 1
        else:
            failed += 1
    
    print("-" * 50)
    print(f"Успешно: {passed}")
    print(f"Ошибки: {failed}")
    print(f"Всего: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
        print("Проект TintoraAI готов к развертыванию.")
        return True
    else:
        print(f"\n❌ {failed} ТЕСТОВ НЕ ПРОШЛИ")
        print("Требуются дополнительные исправления.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
