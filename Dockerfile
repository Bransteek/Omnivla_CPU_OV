FROM python:3.10-slim

LABEL description="OmniVLA edge - CPU inference with OpenVINO optimization"

# ── Sistema base ──────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Copiar el proyecto ────────────────────────────────────────────
COPY . /app/


RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cpu

# ── OpenVINO para optimización en CPU Intel ───────────────────────
RUN pip install --no-cache-dir \
    openvino==2024.0.0 \
    openvino-dev==2024.0.0 \
    nncf==2.9.0

# ── Dependencias del proyecto (sin tensorflow, sin torch fijo) ────
RUN pip install --no-cache-dir \
    accelerate>=0.25.0 \
    draccus==0.8.0 \
    einops \
    huggingface_hub>=0.20.0 \
    json-numpy \
    jsonlines \
    matplotlib \
    peft>=0.10.0 \
    protobuf \
    rich \
    sentencepiece==0.1.99 \
    timm==0.9.10 \
    tokenizers>=0.19.0 \
    wandb \
    imageio \
    utm \
    lmdb \
    zarr \
    datasets \
    av \
    Pillow \
    numpy \
    opencv-python-headless \
    setuptools

# ── CLIP desde el repo oficial (no openai-clip que viene roto) ────
RUN pip install --no-cache-dir \
    git+https://github.com/openai/CLIP.git

# ── Instalar el proyecto en modo editable ────────────────────────
RUN pip install --no-cache-dir -e . || true

# ── Variables de entorno para OpenVINO y CPU ─────────────────────
ENV OPENVINO_ENABLED=1
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV TOKENIZERS_PARALLELISM=false

# ── Punto de entrada ─────────────────────────────────────────────
CMD ["python", "inference/run_omnivla_edge_cpu.py"]