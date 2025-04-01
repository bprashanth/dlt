# Dockerfile for SAM + CLIP
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git ffmpeg libsm6 libxext6 build-essential libglib2.0-0 libgl1-mesa-glx wget

# Upgrade pip
RUN pip install --upgrade pip

# Install Python packages in stages
# First PyTorch and Torchvision with CPU wheels
# If that fails, don't bother with the rest 
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.1.0+cpu torchvision==0.16.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# Clone Segment Anything and GroundingDINO into /models and download weights
RUN mkdir -p /models && \
    git clone https://github.com/facebookresearch/segment-anything.git /models/segment-anything && \
    git clone https://github.com/IDEA-Research/GroundingDINO.git /models/GroundingDINO && \
    wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O /models/groundingdino_swint_ogc.pth && \
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O /models/sam_vit_b_01ec64.pth

COPY . /app

