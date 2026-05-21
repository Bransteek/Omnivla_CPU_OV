FROM osrf/ros:humble-desktop

ENV DEBIAN_FRONTEND=noninteractive

# 1. Instalamos dependencias de sistema y los paquetes de simulación de Turtlebot3
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    ros-humble-cv-bridge \
    ros-humble-geometry-msgs \
    ros-humble-sensor-msgs \
    ros-humble-turtlebot3-gazebo \
    ros-humble-turtlebot3-simulations \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. SOLUCIÓN AL ERROR DE NUMPY: Instalar numpy < 2.0 antes que nada
RUN pip3 install --no-cache-dir "numpy<2.0"

# 3. Instalamos PyTorch (Versión CPU)
RUN pip3 install --no-cache-dir \
    torch==2.2.0 torchvision==0.17.0 \
    --index-url https://download.pytorch.org/whl/cpu

# 4. Resto de dependencias de OmniVLA
RUN pip3 install --no-cache-dir \
    openvino==2024.0.0 \
    openvino-dev==2024.0.0 \
    nncf==2.9.0 \
    matplotlib \
    Pillow \
    opencv-python-headless \
    utm \
    einops \
    transformers==4.38.2 \
    timm==0.9.10 \
    accelerate>=0.25.0 \
    huggingface_hub>=0.20.0 \
    draccus==0.8.0 \
    json-numpy \
    rich \
    setuptools \
    peft \
    protobuf \
    sentencepiece==0.1.99 \
    tokenizers \
    imageio \
    zarr \
    datasets \
    efficientnet_pytorch

# 5. CLIP y fijar NumPy de nuevo por si acaso alguna dependencia lo actualizó
RUN pip3 install --no-cache-dir git+https://github.com/openai/CLIP.git && \
    pip3 install "numpy<2.0" 

RUN pip install --no-cache-dir -e . || true

COPY . .

ENV PYTHONUNBUFFERED=1