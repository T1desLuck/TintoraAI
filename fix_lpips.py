import subprocess
import lpips
import os
import shutil
from pathlib import Path

# Установить 0.1.4 и скачать веса
subprocess.run(["pip", "install", "lpips==0.1.4"])
lpips.LPIPS(net='vgg')  # принудительно скачать веса

# Найти веса
torch_cache = Path.home() / ".cache/torch/hub/checkpoints"
lpips_dir = Path(lpips.__file__).parent

# Сохранить веса
backup_dir = Path("/tmp/lpips_backup")
if torch_cache.exists():
    shutil.copytree(torch_cache, backup_dir, dirs_exist_ok=True)

# Переустановить GitHub версию
subprocess.run(["pip", "uninstall", "lpips", "-y"])
subprocess.run(["pip", "install", "git+https://github.com/richzhang/PerceptualSimilarity.git"])

# Восстановить веса
import lpips  # переимпорт
new_lpips_dir = Path(lpips.__file__).parent / "weights"
new_lpips_dir.mkdir(exist_ok=True)
if backup_dir.exists():
    for f in backup_dir.glob("vgg*"):
        shutil.copy(f, new_lpips_dir)