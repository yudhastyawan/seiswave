#!/bin/bash

# run_web_ui.sh
# Pintasan cepat untuk menjalankan SeisWave Web UI

# Mendapatkan direktori tempat script ini berada
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "--------------------------------------------------------"
echo "🌊 SeisWave Web UI Launcher"
echo "--------------------------------------------------------"

# Path venv yang diharapkan
VENV_PATH="./venv"

# 1. Cek and Aktivasi Virtual Environment
if [ -d "$VENV_PATH" ]; then
    echo "✅ Mengaktifkan Virtual Environment..."
    source "$VENV_PATH/bin/activate"
else
    echo "❌ Error: Virtual environment tidak ditemukan di $VENV_PATH"
    echo "Silakan buat dengan: python -m venv venv"
    exit 1
fi

# 2. Set PYTHONPATH agar modul seiswave terbaca tanpa perlu install
export PYTHONPATH=$PYTHONPATH:.

# 3. Jalankan Streamlit (Mode Testing Langsung)
echo "🚀 Meluncurkan Aplikasi (Streamlit)..."
streamlit run seiswave/web_app.py
