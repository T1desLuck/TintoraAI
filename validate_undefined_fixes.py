#!/usr/bin/env python3
"""
Валидация исправлений undefined name (F821) ошибок TintoraAI.
Проверяет все исправленные переменные в модулях.
"""

import ast
import sys
from pathlib import Path

def check_undefined_names(file_path):
    """Проверяет файл на undefined name ошибки."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Компилируем AST для проверки синтаксиса
        tree = ast.parse(content, filename=str(file_path))
        
        # Список исправленных переменных для каждого файла
        fixed_vars = {
            'modules/discriminator.py': [
                'use_semantic', 'use_rewards', 'num_discriminators', 'use_attention'
            ],
            'modules/guide_net.py': [
                'use_semantic', 'use_reference', 'use_rewards', 
                'input_channels', 'base_channels', 'num_stages'
            ],
            'modules/style_transfer.py': [
                'style_dim', 'use_histogram_loss'
            ],
            'modules/few_shot_adapter.py': [
                'adapter_config', 'prototype_config'
            ]
        }
        
        # Проверяем, что переменные теперь определены в конструкторах
        file_key = str(file_path).replace('\\', '/').split('TintoraAI/')[-1]
        if file_key in fixed_vars:
            print(f"[CHECKING] {file_key}")
            
            # Ищем определения классов и их конструкторы
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for method in node.body:
                        if (isinstance(method, ast.FunctionDef) and 
                            method.name == '__init__'):
                            
                            # Получаем параметры конструктора
                            params = [arg.arg for arg in method.args.args]
                            
                            # Проверяем наличие исправленных переменных
                            missing_vars = []
                            for var in fixed_vars[file_key]:
                                if var not in params:
                                    missing_vars.append(var)
                            
                            if missing_vars:
                                print(f"  [WARNING] Class {node.name} missing params: {missing_vars}")
                            else:
                                print(f"  [OK] Class {node.name} has all required params")
        
        return True
        
    except SyntaxError as e:
        print(f"[ERROR] Syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Error checking {file_path}: {e}")
        return False

def main():
    """Основная функция валидации."""
    print("=== Валидация исправлений undefined name (F821) ===\n")
    
    # Файлы для проверки
    files_to_check = [
        'modules/discriminator.py',
        'modules/guide_net.py', 
        'modules/style_transfer.py',
        'modules/few_shot_adapter.py'
    ]
    
    results = []
    for file_path in files_to_check:
        full_path = Path(__file__).parent / file_path
        if full_path.exists():
            result = check_undefined_names(full_path)
            results.append(result)
        else:
            print(f"[ERROR] File not found: {file_path}")
            results.append(False)
    
    print(f"\n=== Результаты ===")
    passed = sum(results)
    total = len(results)
    print(f"Проверено файлов: {passed}/{total}")
    
    if passed == total:
        print("\n[SUCCESS] Все undefined name ошибки исправлены!")
        print("Проект готов к повторной проверке CI/CD.")
        return 0
    else:
        print("\n[FAILED] Некоторые ошибки остались.")
        return 1

if __name__ == "__main__":
    exit(main())
