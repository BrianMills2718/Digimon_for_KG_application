#!/bin/bash
# run_digimon.sh  —  v2  (make executable once: chmod +x run_digimon.sh)

# ===== tweakables =======================================================
DATASET_NAME="HotpotQAsmallest"          # folder name under ./Data/
CONFIG_PATH="Option/Method/RAPTOR.yaml"  # which method to run
IMAGE_NAME="digimon"                     # Docker image tag
# ========================================================================

echo "🚀  Running DIGIMON   |  Dataset: $DATASET_NAME  |  Config: $CONFIG_PATH"

docker run --rm -it \
  -v "${PWD}:/app" \                # mount local repo → /app inside container
  "$IMAGE_NAME" \
  bash -c "python main.py -opt $CONFIG_PATH -dataset_name $DATASET_NAME"
