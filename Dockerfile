FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG TORCH_VERSION=2.6.0
ARG PYTORCH3D_REF=stable
ARG FLASH_ATTN_VERSION=2.7.3
ARG KAOLIN_VERSION=0.18.0
ARG ROBO_ORCHARD_CORE_REF=ae83c81b8b37ab9a99f4d0c4d994eac4f492e1ee
#ARG ROBO_ORCHARD_LAB_REF=04342840d59078f7eaab809a41390f40cac2ebfe
ARG ROBO_ORCHARD_LAB_REF=cc71435356758a70c971e0acd7b574762612394e

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=US/Pacific \
    NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility \
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    PYOPENGL_PLATFORM=egl \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0" \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    MS_ASSET_DIR=/data/.maniskill \
    PATH="/app/venv/bin:$PATH" \
    VIRTUAL_ENV="/app/venv"

# Combined system setup and package installation
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf && \
    dpkg --add-architecture i386 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        # GPU libraries
        libxau6 libxau6:i386 libxdmcp6 libxdmcp6:i386 libxcb1 libxcb1:i386 \
        libxext6 libxext6:i386 libx11-6 libx11-6:i386 libglvnd0 libglvnd0:i386 \
        libgl1 libgl1:i386 libglx0 libglx0:i386 libegl1 libegl1:i386 \
        libgles2 libgles2:i386 pkg-config libglvnd-dev libglvnd-dev:i386 \
        libgl1-mesa-dev libgl1-mesa-dev:i386 libegl1-mesa-dev libegl1-mesa-dev:i386 \
        libgles2-mesa-dev libgles2-mesa-dev:i386 libx11-xcb-dev libxkbcommon-dev \
        libwayland-dev libxrandr-dev libvulkan1 \
        # Build tools and dependencies
        build-essential vim tree curl wget unzip git cmake \
        libxml2 libxml2-dev libxslt1-dev libfreetype6-dev \
        dirmngr gnupg2 lsb-release xvfb kmod swig patchelf ffmpeg rsync \
        libopenmpi-dev libcups2-dev libssl-dev \
        python3 python3-pip python3-dev python3-setuptools python3-venv \
        libboost-all-dev libosmesa6-dev libglib2.0-0 libxrender1 \
        libglew-dev xpra libglfw3-dev pybind11-dev libeigen3-dev \
        liboctomap-dev libfcl-dev libjpeg-dev gnupg ca-certificates \
        libsm6 libxext6 libxext-dev ninja-build && \
    # Install Google Chrome
    wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | gpg --dearmor > /usr/share/keyrings/google-chrome.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list && \
    apt-get update && \
    apt-get install -y google-chrome-stable && \
    # Make Python 3 the default
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    # Cleanup
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy GPU configuration files
COPY ./10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
COPY ./nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY ./nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json

# Create working directory and Python environment
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt /app/

# Create venv and install Python packages in one layer
RUN umask 0000 && \
    python3.10 -m venv /app/venv && \
    /app/venv/bin/pip install --upgrade pip wheel setuptools packaging ninja && \
    /app/venv/bin/pip install torch==${TORCH_VERSION} torchvision --index-url https://download.pytorch.org/whl/cu124 && \
    git clone --filter=blob:none --branch ${PYTORCH3D_REF} https://github.com/facebookresearch/pytorch3d.git /tmp/pytorch3d && \
    /app/venv/bin/pip install /tmp/pytorch3d --no-build-isolation && \
    rm -rf /tmp/pytorch3d && \
    /app/venv/bin/pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation && \
    /app/venv/bin/pip install psutil==5.9.8 && \
    /app/venv/bin/pip install flash_attn==${FLASH_ATTN_VERSION} --no-build-isolation && \
    /app/venv/bin/pip install "pydantic==2.10.6" && \
    /app/venv/bin/pip install "ray[default]==2.49.1" && \
    /app/venv/bin/pip install "protobuf>=4.25.8,<7.0.0" && \
    /app/venv/bin/pip install "numpy==1.26.4" "opencv-python==4.10.0.84" && \
    FILTER_PATTERN='^(nvdiffrast(@.*)?|flash[_-]attn([<>=!~].*)?|pytorch3d([<>=!~].*)?|torch([<>=!~].*)?|torchvision([<>=!~].*)?|triton([<>=!~].*)?|nvidia-[a-z0-9_-]+([<>=!~].*)?|psutil([<>=!~].*)?|pydantic([<>=!~].*)?|pydantic_core([<>=!~].*)?|protobuf([<>=!~].*)?|numpy([<>=!~].*)?|opencv-python([<>=!~].*)?|ray(\[default\])?([<>=!~].*)?|robo_orchard_core[[:space:]]+@.*|-e[[:space:]]+git\+https://github.com/HorizonRobotics/RoboOrchardLab.*)$' && \
    grep -Evi "${FILTER_PATTERN}" /app/requirements.txt > /tmp/requirements.filtered.txt && \
    /app/venv/bin/pip install -r /tmp/requirements.filtered.txt && \
    /app/venv/bin/pip install --no-deps "robo_orchard_core@git+https://github.com/HorizonRobotics/robo_orchard_core.git@${ROBO_ORCHARD_CORE_REF}" && \
#    /app/venv/bin/pip install --no-deps -e "git+https://github.com/HorizonRobotics/RoboOrchardLab@${ROBO_ORCHARD_LAB_REF}#egg=robo_orchard_lab" && \
    /app/venv/bin/pip install --no-deps -e "git+https://github.com/underactuated/RoboOrchardLab.git@${ROBO_ORCHARD_LAB_REF}#egg=robo_orchard_lab" && \
    /app/venv/bin/pip install psutil==7.2.2 && \
    /app/venv/bin/pip install --force-reinstall torch==${TORCH_VERSION} torchvision --index-url https://download.pytorch.org/whl/cu124 && \
    /app/venv/bin/python -c "import torch; version=torch.__version__; assert '+cu124' in version, f'Expected PyTorch with CUDA 12.4 (+cu124 suffix), but got version {version}'" && \
    /app/venv/bin/pip install kaolin==${KAOLIN_VERSION} -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-${TORCH_VERSION}_cu124.html && \
    chmod -R 777 /app/venv && \
    /app/venv/bin/pip cache purge && \
    rm -f /tmp/requirements.filtered.txt && \
    find /app/venv -name "*.pyc" -delete && \
    find /app/venv -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
