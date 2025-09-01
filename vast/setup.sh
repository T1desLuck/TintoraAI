#!/bin/bash
set -e

# Обновление системы и установка зависимостей
apt update && apt install -y git python3-pip python3-venv unzip rclone

# Конфигурация rclone
mkdir -p ~/.config/rclone
cat <<EOF > ~/.config/rclone/rclone.conf
[gdrive]
type = drive
scope = drive
token = {"access_token":"ya29.A0AS3H6Nzff0QWm9povecO0MSTfHUIAglAGNVzFDWo2PGLV04TWf7qWgMtYAQz-v5lcPoTLqnXDo49Mj_KCXnO-xd9X4taEyhXv6LSHAg-EtC6XbpYarn6P6bOPCgog5BCfGqolgUg3iVb3h5haiHKT4hSLhV7EI6l22gg6_AQLtZoO90st74b1uTckg2jYdQp7ER5X8QaCgYKAQQSAQ4SFQHGX2MiXt6IUdmCUIf57bT80ZaLaQ0206","token_type":"Bearer","refresh_token":"1//0c3W-7zBi7G5QCgYIARAAGAwSNwF-L9Irv3zBYytLmRlj8ue5lznvGcqKXKoYwRbEsqgsAcOttjr_HSYXnekAdHxzeKYOqleXna0","expiry":"2025-08-26T18:36:14.2605497+05:00","expires_in":3599}
team_drive =
EOF

# Клонирование приватного репозитория
GIT_TOKEN="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"
git clone https://$GIT_TOKEN@github.com/T1desLuck/TintoraAI.git
cd TintoraAI

# Виртуальное окружение
python3 -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt

cd ..

# Скачивание датасета
rclone copy gdrive:dataset.tar ./ --progress
tar -xf dataset.tar
rm dataset.tar

# Разбиение на train/val/test
mkdir -p TintoraAI/data/{train,val,test}
shuf -e dataset/* -n 121452 -z | xargs -0 -I{} mv {} TintoraAI/data/train/
shuf -e dataset/* -n 15181 -z | xargs -0 -I{} mv {} TintoraAI/data/val/
shuf -e dataset/* -n 15183 -z | xargs -0 -I{} mv {} TintoraAI/data/test/
