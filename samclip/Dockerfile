# Dockerfile for SAM + CLIP
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git ffmpeg libsm6 libxext6

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone SAM
RUN git clone https://github.com/facebookresearch/segment-anything.git

COPY . /app

