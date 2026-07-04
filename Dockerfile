# Reproducible environment for SweepJEPA.
#
# The US foundation-model repos have finicky deps; pinning a Python 3.10 base and
# installing torch from the CUDA index keeps the toolchain reproducible.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# System libs needed by torchvision image I/O (libjpeg/libpng) and OpenCV-style
# decoders used by some US-FM preprocessing.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential libjpeg-dev libpng-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

ENV PYTHONPATH=/workspace

# Default: run the test suite so the image self-verifies on build.
CMD ["pytest", "-q"]
