#!/bin/bash
# ================================================================
# start.sh — Script de inicio rápido para OmniVLA CPU con Docker
# Uso: bash start.sh
# ================================================================

set -e

# ──  Verificar que existe el modelo ───────────────────────────
if [ ! -f "./omnivla-edge/omnivla-edge.pth" ]; then
    echo ""
    echo "[INFO] Modelo no encontrado. Descargando omnivla-edge..."
    sudo apt install git-lfs -y
    git lfs install
    rm -rf omnivla-edge
    git clone https://huggingface.co/NHirose/omnivla-edge
    echo "[OK] Modelo descargado."
else
    echo "[OK] Modelo encontrado."
fi