FROM runpod/worker-comfyui:5.10.0-base

RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc g++ python3.12-dev \
 && rm -rf /var/lib/apt/lists/*

# ComfyUI 0.45
RUN cd /comfyui \
 && git fetch --depth 1 origin tag v0.45.0 --no-tags \
 && git checkout -f v0.45.0

# Force CUDA 13 torch
RUN uv pip install --python /opt/venv/bin/python \
      --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio \
      torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 \
 && uv pip install --python /opt/venv/bin/python -r /comfyui/requirements.txt \
      "transformers>=4.50.3,<5" "huggingface-hub<1.0"

# Setup verify
RUN python -c "import torch; v=torch.version.cuda; print(torch.__version__, v); assert v.startswith('13'), v" \
 && cd /comfyui && timeout 300 python main.py --quick-test-for-ci --cpu

COPY link_cached_models.sh /link_cached_models.sh
COPY save_s4.py /comfyui/custom_nodes/save_s4.py

RUN chmod +x /link_cached_models.sh

CMD ["/bin/bash", "-c", "/link_cached_models.sh; exec /start.sh"]
