# Dockerfile for SAM + CLIP
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
# Install system dependencies for PyTorch and Detectron2
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    git \
    g++ \
    build-essential \
    cmake \
    libopenblas-dev \
    libomp-dev \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python packages in stages
# First PyTorch and Torchvision with CPU wheels
# If that fails, don't bother with the rest 
COPY requirements-docker.txt .
RUN pip install --no-cache-dir torch==2.1.0+cpu torchvision==0.16.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements-docker.txt


# Install Detectron2
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install Jupyter Notebook
RUN pip install jupyter

# Expose port for Jupyter Notebook
EXPOSE 8888

# Copy your notebooks or scripts into the container (optional)
# Uncomment this when the checkpoint files are removed
COPY . /app

WORKDIR /app

RUN chmod +x trainer.py

# ENTRYPOINT ["python"]
# CMD ["python", "trainer.py"]

# Start Jupyter Notebook server
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''"]
