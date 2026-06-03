# Stage 1: base environment with CUDA and Miniconda
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Avoid interactive frontend
ENV DEBIAN_FRONTEND=noninteractive
RUN useradd -m -u 1000 user
WORKDIR /workspace

# Install basic tools and dependencies
RUN apt-get update && apt-get install -y \
    wget

# Install MuJoCo
RUN mkdir -p ~/.mujoco && \
    wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz && \
    tar -xvzf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco && \
    rm mujoco210-linux-x86_64.tar.gz

# Set environment variables for mujoco
ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    unzip \    
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    python3-pip \
    python3-numpy \
    python3-scipy \
    libc6-dev \
    libglib2.0-dev\
    net-tools \
    libglfw3 \
    libglx-mesa0 \
    libgl1-mesa-dri \
    libxrender1 \
    libxext6 \
    libsm6 \
    patchelf \
    gcc \
    g++ \
    vim \
    xpra\
    xserver-xorg-dev \   
    libssl-dev \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Set compiler environment variables to fix osmesa.h not found error

ENV CPATH=/usr/include/
#ENV CFLAGS="-I/usr/include"
#ENV LDFLAGS="-L/usr/lib"


# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Create conda environment
# RUN conda config --remove channels defaults && \
#     conda config --add channels conda-forge && \
#     conda config --set channel_priority strict

# Copy the conda environment file
COPY environment.yml /workspace/environment.yml

RUN conda tos accept --override-channels --channel defaults && \
    conda env create -f /workspace/environment.yml && \
    conda clean --all -y 
#     # && \
#     # /bin/bash -c "source activate gflower && \
#     # export CPATH=/usr/include && \
#     # export CFLAGS='-I/usr/include' && \
#     # export LDFLAGS='-L/usr/lib'"
ENV CONDA_DEFAULT_ENV=gflower
ENV PATH=$CONDA_DIR/envs/gflower/bin:$PATH


# # Add CONDA to .bashrc
RUN echo "source activate gflower" >> ~/.bashrc

# Set working directory
WORKDIR /workspace

CMD ["/bin/bash"]
