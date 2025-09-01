import sys
import argparse
from pathlib import Path

# Гарантируем, что папка "src" доступна для импорта при запуске из корня проекта
PROJ_ROOT = Path(__file__).parent.resolve()
SRC_DIR = PROJ_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main():
    parser = argparse.ArgumentParser(description="Точка входа TintoraAI")
    # Верхнеуровневый флаг конфигурации, чтобы `python main.py` запускал обучение
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJ_ROOT / "configs" / "default.yaml"),
        help="Путь к YAML-конфигу для обучения",
    )

    subparsers = parser.add_subparsers(dest="command", required=False)

    # Подкоманда train (без доп. аргументов; использует верхний флаг --config)
    subparsers.add_parser("train", help="Запуск обучения")

    # Подкоманда infer
    p_infer = subparsers.add_parser("infer", help="Инференс для одного изображения")
    p_infer.add_argument(
        "--input", type=str, required=False, help="Путь к входному изображению"
    )
    p_infer.add_argument(
        "--checkpoint", type=str, default=str(PROJ_ROOT / "checkpoints" / "latest.pth")
    )
    p_infer.add_argument("--output", type=str, default="output.png")
    p_infer.add_argument("--tile", type=int, default=0)
    p_infer.add_argument("--overlap", type=int, default=32)
    p_infer.add_argument("--cpu", action="store_true")
    p_infer.add_argument(
        "--pad-div",
        type=int,
        default=None,
        help="Делитель для паддинга при инференсе; если не указан, берётся из конфига",
    )

    # Поведение по умолчанию: train
    parser.set_defaults(command="train")

    args, _ = parser.parse_known_args()

    if args.command == "train":
        # Формируем argv для src.train.main, чтобы он корректно разобрал свои аргументы
        sys.argv = ["src.train", "--config", args.config]
        from src.train import main as train_main

        return train_main()

    elif args.command == "infer":
        # Формируем argv для src.inference
        argv = ["src.inference"]
        if args.input:
            argv += ["--input", args.input]
        argv += [
            "--checkpoint",
            args.checkpoint,
            "--output",
            args.output,
            "--overlap",
            str(getattr(args, "overlap", 32)),
            "--tile",
            str(getattr(args, "tile", 0)),
        ]
        # Прокинем паддинг-дивизор только если он явно указан
        if getattr(args, "pad_div", None) is not None:
            argv += ["--pad-div", str(args.pad_div)]
        if getattr(args, "cpu", False):
            argv.append("--cpu")
        sys.argv = argv
        from src.inference import main as infer_main

        return infer_main()

    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
