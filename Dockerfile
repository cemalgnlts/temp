FROM runpod/worker-comfyui:5.10.0-base

# H3 node'ları için ComfyUI 0.45, quant çekirdekleri için CUDA 13 torch
RUN cd /comfyui \
 && git fetch --depth 1 origin tag v0.45.0 --no-tags \
 && git checkout -f v0.45.0 \
 && uv pip install --python /opt/venv/bin/python torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu130 \
 && uv pip install --python /opt/venv/bin/python -r requirements.txt \
      "transformers>=4.50.3,<5" "huggingface-hub<1.0"

# Derleme sırasında ComfyUI'nin açıldığını doğrula
RUN cd /comfyui && timeout 300 python main.py --quick-test-for-ci --cpu
