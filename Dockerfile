FROM runpod/worker-comfyui:5.10.0-base

# H3 node'ları için ComfyUI 0.45
RUN cd /comfyui \
 && git fetch --depth 1 origin tag v0.45.0 --no-tags \
 && git checkout -f v0.45.0

# Quant çekirdekleri için CUDA 13 torch (zorla yeniden kur)
RUN uv pip install --python /opt/venv/bin/python \
      --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio \
      torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 \
 && uv pip install --python /opt/venv/bin/python -r /comfyui/requirements.txt \
      "transformers>=4.50.3,<5" "huggingface-hub<1.0"

# Derleme sırasında doğrula: torch CUDA 13 mü, ComfyUI açılıyor mu
RUN python -c "import torch; v=torch.version.cuda; print(torch.__version__, v); assert v.startswith('13'), v" \
 && cd /comfyui && timeout 300 python main.py --quick-test-for-ci --cpu
