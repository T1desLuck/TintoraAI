#!/usr/bin/env python3
"""
Русскоязычный лаунчер тестов проекта TintoraAI.

Использование (консоль):
  python test_project.py help
  python test_project.py list
  python test_project.py run all
  python test_project.py run modules
  python test_project.py run modules::test_crb
  python test_project.py run inference
  python test_project.py run losses
  python test_project.py run forward
  python test_project.py run dlb

Использование (Jupyter/Colab):
  from test_project import run, show_help, list_targets
  show_help()
  list_targets()
  run("modules")
  run("modules::test_crb")

Скрипт печатает понятные сообщения на русском об успешном прохождении или причинах ошибки.
"""
from __future__ import annotations
import sys
import argparse
import os
import subprocess
from pathlib import Path
from typing import List, Dict

try:
    import pytest  # noqa: F401
except Exception as e:
    print(
        "[ОШИБКА] Пакет pytest не найден. Установите его командой:\n"
        "  python -m pip install pytest\n"
        f"Технические детали: {e}",
        file=sys.stderr,
    )
    sys.exit(2)

# Карта удобных алиасов → цели pytest
TARGETS: Dict[str, List[str]] = {
    "all": ["tests"],
    "dlb": ["tests/test_dlb.py"],
    "forward": ["tests/test_forward.py"],
    "inference": ["tests/test_inference.py"],
    "losses": ["tests/test_losses.py"],
    "modules": ["tests/test_modules.py"],
    # Новый алиас для быстрых интеграционных тестов Adapter/LoRA
    "adapters_lora_fast": [
        "tests/test_train_adapter_fast.py",
        "tests/test_train_lora_fast.py",
    ],
}


def show_help() -> None:
    DESCRIPTIONS = {
        "commands": {
            "help": "Показать подробную справку по использованию лаунчера.",
            "list": "Показать доступные цели (алиасы) с описаниями.",
            "run <цель>": "Запустить тесты по выбранной цели или конкретный тест (через ::).",
            "preview [опции]": "Сохранить предпросмотр подготовки изображения (включая L-дефекты).",
        },
        "targets": {
            "all": "Запускает полный набор тестов (вся папка tests/).",
            "dlb": "Проверка динамического балансировщика потерь (tests/test_dlb.py).",
            "forward": "Быстрый прогон forward модели TintoraAI (tests/test_forward.py).",
            "inference": "Функции инференса: одиночный и тайловый режим (tests/test_inference.py).",
            "losses": "Продвинутые функции потерь (tests/test_losses.py).",
            "modules": "Отдельные модули: backbone/heads/CRB/decoder (tests/test_modules.py).",
            "adapters_lora_fast": "Быстрые интеграционные тесты Adapter/LoRA (микро‑обучение/экспорт)",
        },
        "tips": [
            "Можно запускать точечные тесты: modules::test_crb (любой test_* внутри файла).",
            'В ноутбуке удобно использовать run("modules") или run("modules::test_crb").',
        ],
    }

    lines = []
    lines.append("TintoraAI — помощник по тестам\n")
    lines.append("Команды:")
    for cmd, desc in DESCRIPTIONS["commands"].items():
        lines.append(f"  {cmd:<26} — {desc}")
    lines.append("")
    lines.append("Цели (алиасы):")
    for tgt in ["all", "dlb", "forward", "inference", "losses", "modules", "adapters_lora_fast"]:
        desc = DESCRIPTIONS["targets"].get(tgt, "")
        lines.append(f"  {tgt:<26} — {desc}")
    lines.append("")
    lines.append("Точечные тесты PyTest:")
    lines.append(
        "  modules::test_crb            — Запуск конкретного теста внутри файла."
    )
    lines.append(
        "  modules::test_decoder_unetpp — Любой test_* из tests/test_modules.py."
    )
    lines.append("")
    lines.append("Примеры (консоль):")
    lines.append("  python test_project.py run modules")
    lines.append("  python test_project.py run modules::test_crb")
    lines.append(
        "  python test_project.py preview --input assets/color.jpg --save_L --enable_defects"
    )
    lines.append("")
    lines.append("Примеры (Jupyter/Colab):")
    lines.append("  from test_project import run, show_help, list_targets")
    lines.append("  show_help()")
    lines.append("  list_targets()")
    lines.append('  run("modules")')
    lines.append('  run("modules::test_crb")')
    lines.append("")
    for tip in DESCRIPTIONS["tips"]:
        lines.append(f"Подсказка: {tip}")

    print("\n".join(lines))


def list_targets() -> None:
    DESCRIPTIONS = {
        "all": "Запустить весь набор тестов (вся папка tests/).",
        "dlb": "DynamicLossBalancer: базовые проверки.",
        "forward": "Forward модели TintoraAI: формы и ключи выхода.",
        "inference": "Инференс: одиночный и тайловый режим, диапазоны значений.",
        "losses": "Advanced лоссы: корректность вычислений и финитность значений.",
        "modules": "Backbones/Heads/CRB/Decoder: формы и функциональность.",
        "adapters_lora_fast": "Быстрые тесты Adapter/LoRA (микро‑обучение/экспорт)",
    }
    print("Доступные цели для запуска:")
    for alias in sorted(TARGETS.keys()):
        print(f"  - {alias:<10} — {DESCRIPTIONS.get(alias, '')}")
    print("\nПоддерживаются и точечные цели PyTest, например: modules::test_crb")


def _resolve_targets(arg: str) -> List[str]:
    """Преобразовать введённую цель в список путей/точек PyTest."""
    if "::" in arg:
        # Точечный тест, например: modules::test_crb
        file_alias, test_name = arg.split("::", 1)
        if file_alias not in TARGETS:
            # позволяем путь напрямую
            if file_alias.endswith(".py"):
                return [f"{file_alias}::{test_name}"]
            print(f"[ОШИБКА] Неизвестный алиас файла: {file_alias}")
            sys.exit(2)
        file_path = TARGETS[file_alias][0]
        return [f"{file_path}::{test_name}"]
    # Обычный алиас или прямой путь
    if arg in TARGETS:
        return TARGETS[arg]
    # позволяем прямой путь к файлу/папке
    return [arg]


def _run_pytest(targets: List[str]) -> int:
    # -q: тихий режим, можно убрать если нужен подробный лог
    # Возвращаем код возврата pytest: 0 — успех, >0 — есть падения
    try:
        from pytest import main as pytest_main

        code = pytest_main(["-q", *targets])
        return int(code)
    except SystemExit as e:
        return int(getattr(e, "code", 1))


def _run_preview(argv_rest: List[str]) -> int:
    """Запуск предпросмотра подготовки изображений без PyTest.

    Прокидывает PYTHONPATH и вызывает scripts/preview_preprocessing.py с переданными флагами.
    """
    repo_root = Path(__file__).resolve().parent
    script = repo_root / "scripts" / "preview_preprocessing.py"
    if not script.exists():
        print(f"[ОШИБКА] Не найден скрипт предпросмотра: {script}")
        return 2

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    # Разбираем опции предпросмотра
    p = argparse.ArgumentParser(prog="test_project.py preview", add_help=True)
    p.add_argument("--input", required=True, help="Путь к входному RGB изображению")
    p.add_argument("--config", default=str(repo_root / "configs" / "default.yaml"))
    p.add_argument("--mode", choices=["train", "val"], default="train")
    p.add_argument("--dataset", choices=["advanced", "simple"], default=None)
    p.add_argument("--output", default=None, help="Выходная папка для превью")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--disable_flip", action="store_true")
    p.add_argument("--disable_jitter", action="store_true")
    p.add_argument("--save_L", action="store_true")
    p.add_argument("--enable_defects", action="store_true")
    ns = p.parse_args(argv_rest)

    cmd = [
        sys.executable,
        str(script),
        "--input",
        ns.input,
        "--config",
        ns.config,
        "--mode",
        ns.mode,
    ]
    if ns.dataset is not None:
        cmd += ["--dataset", ns.dataset]
    if ns.output is not None:
        cmd += ["--output", ns.output]
    cmd += ["--seed", str(ns.seed)]
    if ns.disable_flip:
        cmd.append("--disable_flip")
    if ns.disable_jitter:
        cmd.append("--disable_jitter")
    if ns.save_L:
        cmd.append("--save_L")
    if ns.enable_defects:
        cmd.append("--enable_defects")

    print("[ИНФО] Предпросмотр подготовки изображения…")
    print("Команда:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, env=env, cwd=repo_root)
        print("[УСПЕХ] Предпросмотр выполнен. Файлы сохранены в указанную папку.")
        return 0
    except subprocess.CalledProcessError as e:
        print(
            f"[НЕУСПЕХ] Предпросмотр завершился с ошибкой (код {e.returncode}). См. лог выше."
        )
        return int(e.returncode or 1)


def run(target: str) -> bool:
    """Публичная функция для Jupyter/Colab: запускает тест(ы) по цели и печатает результат на русском.

    Возвращает True при успехе, False при падениях.
    """
    targets = _resolve_targets(target)
    print(f"[ИНФО] Запуск тестов для цели: {target} → {targets}")
    code = _run_pytest(targets)
    if code == 0:
        print("[УСПЕХ] Тесты завершились успешно. Ошибок не выявлено.")
        return True
    else:
        print(
            "[НЕУСПЕХ] Обнаружены падения тестов. См. вывод выше для деталей по упавшим кейсам.\n"
            "Подсказка: можно запустить конкретный тест, например: modules::test_crb"
        )
        return False


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("command", nargs="?", default="help")
    parser.add_argument("arg", nargs="?")
    # Остальные аргументы передаются в подкоманды (например, preview)
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    ns = parser.parse_args(argv)

    if ns.command in ("help", "--help", "-h"):
        show_help()
        return 0
    if ns.command == "list":
        list_targets()
        return 0
    if ns.command == "run":
        if not ns.arg:
            print("[ОШИБКА] Укажите цель: python test_project.py run <цель>")
            list_targets()
            return 2
        targets = _resolve_targets(ns.arg)
        code = _run_pytest(targets)
        if code == 0:
            print("[УСПЕХ] Тесты завершились успешно. Ошибок не выявлено.")
        else:
            print(
                "[НЕУСПЕХ] Обнаружены падения тестов. См. вывод выше для деталей.\n"
                "Подсказка: сузьте запуск до конкретного теста, например: modules::test_crb"
            )
        return code
    if ns.command == "preview":
        # Передаём оставшиеся аргументы парсеру предпросмотра
        # ns.rest включает ведущий "--" если пользователь вставил его; удалим его.
        rest = [a for a in ns.rest if a != "--"]
        return _run_preview(rest)

    print(
        f"[ОШИБКА] Неизвестная команда: {ns.command}\nИспользуйте: help | list | run <цель>"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
