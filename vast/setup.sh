#!/bin/bash
set -euo pipefail

# --- Параметры (вложены по твоему ТЗ) ---
PROJECT_ROOT="/workspace/TintoraAI"
RCLONE_CONF_PATH="$PROJECT_ROOT/rclone.conf"    # локальный конфиг rclone для этого проекта
GDRIVE_REMOTE="gdrive:dataset.tar"              # remote: путь к архиву на гдиске
TMP_ARCHIVE="$PROJECT_ROOT/dataset.tar.tmp"
FINAL_ARCHIVE="$PROJECT_ROOT/dataset.tar"
DATASET_DIR="$PROJECT_ROOT/dataset"
TRAIN_N=121452
VAL_N=15181
TEST_N=15183

# --- Установки: rclone, unzip (идем аккуратно, идем idempotent) ---
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y rclone unzip || true

# --- Создаём проектную папку (на всякий) ---
mkdir -p "$PROJECT_ROOT"
chmod 0755 "$PROJECT_ROOT"

# --- Пишем rclone конфиг в файл проекта (используем тот токен который ты дал) ---
cat > "$RCLONE_CONF_PATH" <<'RCONF'
[gdrive]
type = drive
scope = drive
token = {"access_token":"ya29.A0AS3H6Nzff0QWm9povecO0MSTfHUIAglAGNVzFDWo2PGLV04TWf7qWgMtYAQz-v5lcPoTLqnXDo49Mj_KCXnO-xd9X4taEyhXv6LSHAg-EtC6XbpYarn6P6bOPCgog5BCfGqolgUg3iVb3h5haiHKT4hSLhV7EI6l22gg6_AQLtZoO90st74b1uTckg2jYdQp7ER5X8QaCgYKAQQSAQ4SFQHGX2MiXt6IUdmCUIf57bT80ZaLaQ0206","token_type":"Bearer","refresh_token":"1//0c3W-7zBi7G5QCgYIARAAGAwSNwF-L9Irv3zBYytLmRlj8ue5lznvGcqKXKoYwRbEsqgsAcOttjr_HSYXnekAdHxzeKYOqleXna0","expiry":"2025-08-26T18:36:14.2605497+05:00","expires_in":3599}
team_drive =
RCONF

chmod 0600 "$RCLONE_CONF_PATH"

# --- Скачиваем архив атомарно: сначала в tmp, затем переименовываем в final ---
echo "Скачиваем архив с GDrive..."
rclone --config "$RCLONE_CONF_PATH" copyto "$GDRIVE_REMOTE" "$TMP_ARCHIVE" --transfers=4 --progress
# проверка наличия и размера
if [ ! -s "$TMP_ARCHIVE" ]; then
  echo "Ошибка: скачанный архив пустой или не найден"; exit 2
fi
mv -f "$TMP_ARCHIVE" "$FINAL_ARCHIVE"
sync

# --- Распаковываем архив в проект (явно указываем путь) ---
echo "Распаковываем архив..."
mkdir -p "$PROJECT_ROOT"
tar -xf "$FINAL_ARCHIVE" -C "$PROJECT_ROOT"
if [ $? -ne 0 ]; then echo "Ошибка распаковки"; exit 3; fi

# только после полной распаковки удаляем архив
rm -f "$FINAL_ARCHIVE"
sync

# --- Ожидаем, что распаковка создала папку $PROJECT_ROOT/dataset (если другое имя, исправь) ---
if [ ! -d "$DATASET_DIR" ]; then
  echo "Ошибка: ожидаемая папка $DATASET_DIR не найдена после распаковки"; exit 4
fi

# --- Создаём структуры train/val/test ---
mkdir -p "$PROJECT_ROOT/data"/{train,val,test}
chmod 0755 "$PROJECT_ROOT/data" "$PROJECT_ROOT/data"/{train,val,test}

# --- ВНИМАНИЕ: готовим список файлов корректно, чтобы не словить 'argument list too long' ---
ALL_LIST="/tmp/all_images_$$.txt"
SHUF_LIST="/tmp/all_images_shuf_$$.txt"

find "$DATASET_DIR" -type f > "$ALL_LIST"
if [ ! -s "$ALL_LIST" ]; then echo "Ошибка: в $DATASET_DIR нет файлов"; exit 5; fi

shuf "$ALL_LIST" > "$SHUF_LIST"

TOTAL=$(wc -l < "$SHUF_LIST")
echo "Всего файлов в dataset: $TOTAL"

# Проверяем что хватает файлов на запланированные размеры
WANT_SUM=$((TRAIN_N + VAL_N + TEST_N))
if [ "$TOTAL" -lt "$WANT_SUM" ]; then
  echo "Ошибка: всего $TOTAL файлов, а требуется $WANT_SUM (train+val+test)"; exit 6
fi

# --- Перемещаем ровно по порядку: сначала train, затем val, затем test ---
head -n "$TRAIN_N" "$SHUF_LIST" | xargs -d '\n' -I{} mv {} "$PROJECT_ROOT/data/train/"
sed -n "$((TRAIN_N+1)),$((TRAIN_N+VAL_N))p" "$SHUF_LIST" | xargs -d '\n' -I{} mv {} "$PROJECT_ROOT/data/val/"
sed -n "$((TRAIN_N+VAL_N+1)),$((TRAIN_N+VAL_N+TEST_N))p" "$SHUF_LIST" | xargs -d '\n' -I{} mv {} "$PROJECT_ROOT/data/test/"

# --- Удаляем временные списки and пустую папку dataset (только если пустая) ---
rm -f "$ALL_LIST" "$SHUF_LIST"
# проверим, пустая ли папка dataset
if [ -z "$(find "$DATASET_DIR" -type f -maxdepth 1 -print -quit)" ]; then
  rm -rf "$DATASET_DIR"
else
  echo "Внимание: после распределения в $DATASET_DIR остались файлы — проверь логи"; exit 7
fi

# --- Создаём виртуальное окружение и устанавливаем зависимости ---
python3 -m venv "$PROJECT_ROOT/venv"
# активируем и ставим зависимости
# shellcheck disable=SC1090
source "$PROJECT_ROOT/venv/bin/activate"
pip install -U pip setuptools wheel
pip install -r "$PROJECT_ROOT/requirements.txt"

echo "Setup внутри проекта завершён успешно."