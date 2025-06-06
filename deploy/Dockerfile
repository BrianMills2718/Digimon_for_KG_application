# Dockerfile (CPU-only)  v2
# Build with:  docker build -t digimon .

FROM continuumio/miniconda3:latest

# ── system libs needed by scipy / faiss / igraph on CPU ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git gcc g++ libopenblas-dev libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy env specs first to leverage Docker layer-cache
COPY environment.yml requirements.txt /tmp/

# Create the conda env (named “digimon”)
RUN conda env create -f /tmp/environment.yml && conda clean -afy

# Make every subsequent RUN/CMD use that env
SHELL ["conda", "run", "-n", "digimon", "/bin/bash", "-c"]

# Copy the rest of the project
COPY . /app

# Default shell when you `docker run -it`
CMD ["bash"]
