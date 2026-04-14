#!/bin/bash
# ================================================================
# start.sh — Script de inicio rápido para OmniVLA CPU con Docker
# Uso: bash start.sh
# ================================================================

set -e

echo ""
echo "=================================================="
echo "  OmniVLA-edge  —  Docker CPU Setup"
echo "=================================================="
echo ""

# ── 1. Verificar Docker ──────────────────────────────────────────
if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker no está instalado."
    echo "        Descárgalo en: https://www.docker.com/products/docker-desktop"
    exit 1
fi
echo "[OK] Docker encontrado: $(docker --version)"

# ── 2. Verificar que existe el modelo ───────────────────────────
if [ ! -f "./omnivla-edge/omnivla-edge.pth" ]; then
    echo ""
    echo "[INFO] Modelo no encontrado. Descargando omnivla-edge..."
    git lfs install
    git clone https://huggingface.co/NHirose/omnivla-edge
    echo "[OK] Modelo descargado."
else
    echo "[OK] Modelo encontrado."
fi

# ── 3. Crear carpeta de outputs ──────────────────────────────────
mkdir -p outputs

# ── 4. Build de la imagen Docker ─────────────────────────────────
echo ""
echo "[INFO] Construyendo imagen Docker (solo la primera vez, ~5 min)..."
docker compose build

# ── 5. Convertir modelo a OpenVINO (solo si no existe ya) ────────
if [ ! -f "./omnivla-edge-ov/omnivla_edge.xml" ]; then
    echo ""
    echo "[INFO] Convirtiendo modelo a OpenVINO para acelerar CPU..."
    echo "       (solo se hace una vez, puede tardar 2-3 min)"
    docker compose run --rm omnivla-convert
    echo "[OK] Modelo OpenVINO listo."
else
    echo "[OK] Modelo OpenVINO ya existe."
fi

# ── 6. Correr inferencia ─────────────────────────────────────────
echo ""
echo "[INFO] Ejecutando inferencia..."
docker compose up omnivla-cpu

echo ""
echo "[OK] Resultado guardado en: ./inference/"
echo "     Busca el archivo .jpg con la trayectoria generada."
