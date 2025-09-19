FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:$PATH

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    wget \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh
RUN conda config --set always_yes true && \
    conda config --set channel_priority flexible && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Set working directory
WORKDIR /workspace

# Copy the project files
COPY . .

# Initialize submodules
RUN git submodule update --init --recursive

# Install dependencies
RUN conda env create -f environment.yml
RUN conda run -n vocalign mim install "mmcv==2.1.0"

# Set conda run as entrypoint for easier usage
ENTRYPOINT ["conda", "run", "-n", "vocalign", "--no-capture-output"]
CMD ["bash"]
