#!/bin/bash
set -e

source /root/miniconda3/bin/activate cosmos-predict1

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/checkpoints}"

if [ ! -f "$CHECKPOINT_DIR/Diffusion_Renderer_Inverse_Cosmos_7B/model.pt" ] || \
   [ ! -f "$CHECKPOINT_DIR/Diffusion_Renderer_Forward_Cosmos_7B/model.pt" ]; then
    echo "Model weights not found in $CHECKPOINT_DIR — downloading..."
    if [ -n "$HUGGING_FACE_HUB_TOKEN" ]; then
        huggingface-cli login --token "$HUGGING_FACE_HUB_TOKEN" --add-to-git-credential
    fi
    python scripts/download_diffusion_renderer_checkpoints.py --checkpoint_dir "$CHECKPOINT_DIR"
else
    echo "Model weights found in $CHECKPOINT_DIR — skipping download."
fi

exec "$@"
