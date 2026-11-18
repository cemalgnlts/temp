#!/usr/bin/env bash

HUB=/runpod-volume/huggingface-cache/hub
NAME="models--${H3_CACHE_REPO//\//--}"
REPO=$(find "$HUB" -maxdepth 1 -iname "$NAME" 2>/dev/null | head -n1)
SNAP=$(ls -d "$REPO"/snapshots/*/ 2>/dev/null | head -n1)

if [[ -z "$SNAP" ]]; then
  echo "[cache] missing model cache: $NAME"; exit 0
fi

for d in diffusion_models text_encoders vae; do
  mkdir -p "/comfyui/models/$d"
  ln -sf "$SNAP$d"/*.safetensors "/comfyui/models/$d/"
done

echo "[cache] found: $SNAP"
