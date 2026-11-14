#!/bin/bash

set -euo pipefail

### Configuration ###
WORKSPACE_DIR="${WORKSPACE:-/workspace}"
COMFYUI_DIR="${WORKSPACE_DIR}/ComfyUI"
MODELS_DIR="${COMFYUI_DIR}/models"
INPUTS_DIR="${COMFYUI_DIR}/input"
WORKFLOWS_DIR="${COMFYUI_DIR}/user/default/workflows"
HF_SEMAPHORE_DIR="${WORKSPACE_DIR}/hf_download_sem_$$"
HF_MAX_PARALLEL=3
WGET_MAX_PARALLEL=5
MODEL_LOG="${MODEL_LOG:-/var/log/portal/comfyui.log}"

# ---- Speed-oriented choices (override with env vars on the template) ----
# Diffusion model quant: w6a8 (16 GB, smallest -> least offloading) | int8_convrot (21 GB, official template default) | fp8_scaled
H3_UNET_QUANT="${H3_UNET_QUANT:-w6a8}"
# Turbo LoRA: 4step (trained at 768p, fastest) | 8step (official template's turbo setting)
H3_TURBO="${H3_TURBO:-4step}"
# Video VAE: sol (distilled whole-clip decoder, faster decode) | stock
H3_VIDEO_VAE="${H3_VIDEO_VAE:-sol}"

# Default request payload (what your_workflow.json is written with)
H3_WIDTH="${H3_WIDTH:-1344}"
H3_HEIGHT="${H3_HEIGHT:-768}"
H3_LENGTH="${H3_LENGTH:-124}"            # frames at 24 fps on the 17k+5 grid: 56=~2.3s, 124=~5s, 243=~10s, 362=~15s

# Warm-up benchmark: only there to load the models, so it is small, fixed and runs once
BENCH_WIDTH=640
BENCH_HEIGHT=352
BENCH_LENGTH=56                          # ~2.3s, the shortest grid value above 2 seconds
BENCH_PROMPT="A red fox walks through a snowy forest. Audio: footsteps on snow."

H3_REPO="https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main"

case "$H3_UNET_QUANT" in
  w6a8|int8_convrot|fp8_scaled) UNET_FILE="minimax_h3_fl2va_pruned_${H3_UNET_QUANT}.safetensors" ;;
  *) echo "Invalid H3_UNET_QUANT: $H3_UNET_QUANT" >&2; exit 1 ;;
esac

case "$H3_TURBO" in
  4step)
    LORA_FILE="minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"
    LORA_URL="$H3_REPO/loras/$LORA_FILE"
    TURBO_STEPS=4 ;;
  8step)
    LORA_FILE="minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
    LORA_URL="https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/$LORA_FILE"
    TURBO_STEPS=8 ;;
  *) echo "Invalid H3_TURBO: $H3_TURBO" >&2; exit 1 ;;
esac

case "$H3_VIDEO_VAE" in
  sol)   VIDEO_VAE_FILE="minimax_h3_sol_vae_int8_convrot.safetensors" ;;
  stock) VIDEO_VAE_FILE="minimax_h3_video_vae_int8_convrot.safetensors" ;;
  *) echo "Invalid H3_VIDEO_VAE: $H3_VIDEO_VAE" >&2; exit 1 ;;
esac

TEXT_ENCODER_FILE="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
AUDIO_VAE_FILE="minimax_h3_audio_vae_fp32.safetensors"

# Model declarations: "URL|OUTPUT_PATH"
HF_MODELS=(
  "$H3_REPO/diffusion_models/$UNET_FILE|$MODELS_DIR/diffusion_models/$UNET_FILE"
  "$H3_REPO/text_encoders/$TEXT_ENCODER_FILE|$MODELS_DIR/text_encoders/$TEXT_ENCODER_FILE"
  "$H3_REPO/vae/$VIDEO_VAE_FILE|$MODELS_DIR/vae/$VIDEO_VAE_FILE"
  "$H3_REPO/vae/$AUDIO_VAE_FILE|$MODELS_DIR/vae/$AUDIO_VAE_FILE"
  "$LORA_URL|$MODELS_DIR/loras/$LORA_FILE"
)

# Wget declarations: "URL|OUTPUT_PATH"
WGET_DOWNLOADS=(
)

### End Configuration ###

# Ensure log directory exists
mkdir -p "$(dirname "$MODEL_LOG")"


log() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$message" | tee -a "$MODEL_LOG"
}

script_cleanup() {
    log "Cleaning up semaphore directory..."
    rm -rf "$HF_SEMAPHORE_DIR"
    # Clean up any stale lock files from this run
    find "$MODELS_DIR" -name "*.lock" -type f -mmin +60 -delete 2>/dev/null || true
    find "$INPUTS_DIR" -name "*.lock" -type f -mmin +60 -delete 2>/dev/null || true
}

# If this script fails we cannot let a serverless worker be marked as ready.
script_error() {
    local exit_code=$?
    local line_number=$1
    log "[ERROR] Provisioning Script failed at line $line_number with exit code $exit_code"
    exit "$exit_code"
}

trap script_cleanup EXIT
trap 'script_error $LINENO' ERR

# HuggingFace download helper using flock for robust locking
download_hf_file() {
    local url="$1"
    local output_path="$2"
    local lockfile="${output_path}.lock"
    local max_retries=5
    local retry_delay=2

    # Acquire slot for parallel download limiting
    local slot
    slot=$(acquire_slot "$HF_SEMAPHORE_DIR/hf" "$HF_MAX_PARALLEL")

    # Ensure parent directory exists for lockfile
    mkdir -p "$(dirname "$output_path")"

    # Use flock for atomic locking - automatically released if process dies
    (
        # Acquire exclusive lock (wait up to 300 seconds)
        if ! flock -x -w 300 200; then
            log "[ERROR] Could not acquire lock for $output_path after 300s"
            release_slot "$slot"
            exit 1
        fi

        # Check if file already exists (must be inside lock to avoid race)
        if [ -f "$output_path" ]; then
            log "File already exists: $output_path (skipping)"
            release_slot "$slot"
            exit 0
        fi

        # Extract repo and file path from HuggingFace URL
        local repo file_path
        repo=$(echo "$url" | sed -n 's|https://huggingface.co/\([^/]*/[^/]*\)/resolve/.*|\1|p')
        file_path=$(echo "$url" | sed -n 's|https://huggingface.co/[^/]*/[^/]*/resolve/[^/]*/\(.*\)|\1|p')

        if [ -z "$repo" ] || [ -z "$file_path" ]; then
            log "[ERROR] Invalid HuggingFace URL: $url"
            release_slot "$slot"
            exit 1
        fi

        local temp_dir
        temp_dir=$(mktemp -d)
        local attempt=1
        local current_delay=$retry_delay

        # Retry loop for rate limits and transient failures
        while [ $attempt -le $max_retries ]; do
            log "Downloading $repo/$file_path (attempt $attempt/$max_retries)..."

            if hf download "$repo" \
                "$file_path" \
                --local-dir "$temp_dir" 2>&1 | tee -a "$MODEL_LOG"; then

                # Verify the file was actually downloaded
                if [ -f "$temp_dir/$file_path" ]; then
                    # Success - move file and clean up
                    mv "$temp_dir/$file_path" "$output_path"
                    rm -rf "$temp_dir"
                    release_slot "$slot"
                    log "✓ Successfully downloaded: $output_path"
                    exit 0
                else
                    log "✗ Download command succeeded but file not found at $temp_dir/$file_path"
                fi
            fi

            log "✗ Download failed (attempt $attempt/$max_retries), retrying in ${current_delay}s..."
            sleep $current_delay
            current_delay=$((current_delay * 2))  # Exponential backoff
            attempt=$((attempt + 1))
        done

        # All retries failed
        log "[ERROR] Failed to download $output_path after $max_retries attempts"
        rm -rf "$temp_dir"
        release_slot "$slot"
        exit 1

    ) 200>"$lockfile"

    local result=$?
    # Clean up lockfile after completion
    rm -f "$lockfile"
    return $result
}

# Wget download helper using flock for robust locking
download_wget_file() {
    local url="$1"
    local output_path="$2"
    local lockfile="${output_path}.lock"
    local max_retries=5
    local retry_delay=2

    # Acquire slot for parallel download limiting
    local slot
    slot=$(acquire_slot "$HF_SEMAPHORE_DIR/wget" "$WGET_MAX_PARALLEL")

    # Ensure parent directory exists
    mkdir -p "$(dirname "$output_path")"

    # Use flock for atomic locking - automatically released if process dies
    (
        # Acquire exclusive lock (wait up to 300 seconds)
        if ! flock -x -w 300 200; then
            log "[ERROR] Could not acquire lock for $output_path after 300s"
            release_slot "$slot"
            exit 1
        fi

        # Check if file already exists (must be inside lock to avoid race)
        if [ -f "$output_path" ]; then
            log "File already exists: $output_path (skipping)"
            release_slot "$slot"
            exit 0
        fi

        local temp_file
        temp_file=$(mktemp)
        local attempt=1
        local current_delay=$retry_delay

        # Retry loop for rate limits and transient failures
        while [ $attempt -le $max_retries ]; do
            log "Downloading $url (attempt $attempt/$max_retries)..."

            if wget \
                --quiet \
                --show-progress \
                --timeout=60 \
                --tries=1 \
                --output-document="$temp_file" \
                "$url" 2>&1 | tee -a "$MODEL_LOG"; then

                # Verify the file was actually downloaded and has content
                if [ -f "$temp_file" ] && [ -s "$temp_file" ]; then
                    # Success - move file and clean up
                    mv "$temp_file" "$output_path"
                    release_slot "$slot"
                    log "✓ Successfully downloaded: $output_path"
                    exit 0
                else
                    log "✗ Download command succeeded but file is empty or missing"
                fi
            fi

            log "✗ Download failed (attempt $attempt/$max_retries), retrying in ${current_delay}s..."
            sleep $current_delay
            current_delay=$((current_delay * 2))  # Exponential backoff
            attempt=$((attempt + 1))
        done

        # All retries failed
        log "[ERROR] Failed to download $output_path after $max_retries attempts"
        rm -f "$temp_file"
        release_slot "$slot"
        exit 1

    ) 200>"$lockfile"

    local result=$?
    # Clean up lockfile after completion
    rm -f "$lockfile"
    return $result
}

acquire_slot() {
    local prefix="$1"
    local max_slots="$2"
    
    while true; do
        local count
        count=$(find "$(dirname "$prefix")" -name "$(basename "$prefix")_*" 2>/dev/null | wc -l)
        if [ "$count" -lt "$max_slots" ]; then
            local slot="${prefix}_$$_$RANDOM"
            touch "$slot"
            echo "$slot"
            return 0
        fi
        sleep 0.5
    done
}

release_slot() {
    rm -f "$1"
}

update_comfyui() {
    local ref="${COMFYUI_REF:-v0.45.0}"
    log "Updating ComfyUI to $ref..."
    (
        cd "$COMFYUI_DIR"
        git fetch origin tag "$ref" --no-tags
        git checkout -f "$ref"
        if command -v uv >/dev/null 2>&1; then
            uv pip install -r requirements.txt
        else
            pip install -r requirements.txt
        fi
    ) 2>&1 | tee -a "$MODEL_LOG"
    log "✓ ComfyUI updated to $ref"
}

main() {
    log "Starting ComfyUI provisioning (MiniMax H3 video: unet=$H3_UNET_QUANT turbo=$H3_TURBO vae=$H3_VIDEO_VAE)..."
    
    # Activate virtual environment if it exists
    if [ -f /venv/main/bin/activate ]; then
        # shellcheck source=/dev/null
        . /venv/main/bin/activate
    fi

    update_comfyui

    # Clean up any leftover semaphores from previous runs
    rm -rf "$HF_SEMAPHORE_DIR"
    mkdir -p "$HF_SEMAPHORE_DIR"
    mkdir -p "$WORKFLOWS_DIR"
    mkdir -p "$INPUTS_DIR"
    mkdir -p "$MODELS_DIR"/{diffusion_models,text_encoders,vae,loras}

    # Write workflows
    write_api_workflow >/dev/null 2>&1 &

    # Periodically cleanup old generations
    set_cleanup_job

    # Collect all background job PIDs
    local pids=()

    # Download all HuggingFace models in parallel
    for model in "${HF_MODELS[@]}"; do
        url="${model%%|*}"
        output_path="${model##*|}"
        
        # Trim whitespace
        url=$(echo "$url" | xargs)
        output_path=$(echo "$output_path" | xargs)
        
        log "Queuing HF download: $url -> $output_path"
        download_hf_file "$url" "$output_path" &
        pids+=($!)
    done

    # Download all wget files in parallel
    for item in "${WGET_DOWNLOADS[@]}"; do
        # Skip empty entries
        [[ -z "${item// }" ]] && continue
        
        url="${item%%|*}"
        output_path="${item##*|}"
        
        # Trim whitespace
        url=$(echo "$url" | xargs)
        output_path=$(echo "$output_path" | xargs)
        
        log "Queuing wget download: $url -> $output_path"
        download_wget_file "$url" "$output_path" &
        pids+=($!)
    done

    # Wait for each job and check exit status
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            log "[ERROR] Download process $pid failed"
            failed=1
        fi
    done

    if [ $failed -eq 1 ]; then
        log "[ERROR] One or more downloads failed"
        exit 1
    fi

    log "✓ All downloads completed successfully"
}

# MiniMax H3 text-to-video with native audio, API format.
# Converted from the official template (Comfy-Org/workflow_templates: video_minimax_h3_t2v.json)
# with the turbo LoRA always on. For image-to-video, add a LoadImage node and connect it to
# node 6 as "first_frame" (and optionally "last_frame").
write_api_workflow() {
    local workflow_template
    read -r -d '' workflow_template << 'WORKFLOW_JSON' || true
{
  "1": {
    "class_type": "UNETLoader",
    "inputs": {
      "unet_name": "@@UNET@@",
      "weight_dtype": "default"
    }
  },
  "2": {
    "class_type": "CLIPLoader",
    "inputs": {
      "clip_name": "@@TEXT_ENCODER@@",
      "type": "minimax",
      "device": "default"
    }
  },
  "3": {
    "class_type": "VAELoader",
    "inputs": {
      "vae_name": "@@VIDEO_VAE@@"
    }
  },
  "4": {
    "class_type": "VAELoader",
    "inputs": {
      "vae_name": "@@AUDIO_VAE@@"
    }
  },
  "5": {
    "class_type": "LoraLoaderModelOnly",
    "inputs": {
      "model": ["1", 0],
      "lora_name": "@@LORA@@",
      "strength_model": 1
    }
  },
  "6": {
    "class_type": "MiniMaxH3ImageToVideo",
    "inputs": {
      "clip": ["2", 0],
      "vae": ["3", 0],
      "prompt": "@@PROMPT@@",
      "width": @@WIDTH@@,
      "height": @@HEIGHT@@,
      "length": @@LENGTH@@
    }
  },
  "7": {
    "class_type": "BasicGuider",
    "inputs": {
      "model": ["5", 0],
      "conditioning": ["6", 0]
    }
  },
  "8": {
    "class_type": "RandomNoise",
    "inputs": {
      "noise_seed": "__RANDOM_INT__"
    }
  },
  "9": {
    "class_type": "KSamplerSelect",
    "inputs": {
      "sampler_name": "res_multistep"
    }
  },
  "10": {
    "class_type": "BasicScheduler",
    "inputs": {
      "model": ["5", 0],
      "scheduler": "simple",
      "steps": @@STEPS@@,
      "denoise": 1
    }
  },
  "11": {
    "class_type": "SamplerCustomAdvanced",
    "inputs": {
      "noise": ["8", 0],
      "guider": ["7", 0],
      "sampler": ["9", 0],
      "sigmas": ["10", 0],
      "latent_image": ["6", 1]
    }
  },
  "12": {
    "class_type": "VAEDecode",
    "inputs": {
      "samples": ["11", 0],
      "vae": ["3", 0]
    }
  },
  "13": {
    "class_type": "VAEDecodeAudio",
    "inputs": {
      "samples": ["11", 0],
      "vae": ["4", 0]
    }
  },
  "14": {
    "class_type": "CreateVideo",
    "inputs": {
      "images": ["12", 0],
      "audio": ["13", 0],
      "fps": 24
    }
  },
  "15": {
    "class_type": "SaveVideo",
    "inputs": {
      "video": ["14", 0],
      "filename_prefix": "video/minimax_h3",
      "format": "auto",
      "format.codec": "auto"
    }
  }
}
WORKFLOW_JSON

    # Fill in the model names shared by both payloads
    workflow_template="${workflow_template//@@UNET@@/$UNET_FILE}"
    workflow_template="${workflow_template//@@TEXT_ENCODER@@/$TEXT_ENCODER_FILE}"
    workflow_template="${workflow_template//@@VIDEO_VAE@@/$VIDEO_VAE_FILE}"
    workflow_template="${workflow_template//@@AUDIO_VAE@@/$AUDIO_VAE_FILE}"
    workflow_template="${workflow_template//@@LORA@@/$LORA_FILE}"
    workflow_template="${workflow_template//@@STEPS@@/$TURBO_STEPS}"

    # Default request payload
    local default_prompt="Cinematic live-action shot: a red fox trots through a snowy pine forest at sunrise, golden light through the trees, breath visible in the cold air. The camera tracks low alongside it. Audio: soft crunch of paws on snow, distant birdsong, a light wind."
    local workflow_json="$workflow_template"
    workflow_json="${workflow_json//@@PROMPT@@/$default_prompt}"
    workflow_json="${workflow_json//@@WIDTH@@/$H3_WIDTH}"
    workflow_json="${workflow_json//@@HEIGHT@@/$H3_HEIGHT}"
    workflow_json="${workflow_json//@@LENGTH@@/$H3_LENGTH}"

    # Warm-up payload: same graph (so every model gets loaded), smallest sensible size
    local benchmark_json="$workflow_template"
    benchmark_json="${benchmark_json//@@PROMPT@@/$BENCH_PROMPT}"
    benchmark_json="${benchmark_json//@@WIDTH@@/$BENCH_WIDTH}"
    benchmark_json="${benchmark_json//@@HEIGHT@@/$BENCH_HEIGHT}"
    benchmark_json="${benchmark_json//@@LENGTH@@/$BENCH_LENGTH}"

    # Write first file (original format)
    rm -f /opt/comfyui-api-wrapper/payloads/*
    cat > /opt/comfyui-api-wrapper/payloads/your_workflow.json << EOF
{
    "input": {
        "request_id": "",
        "workflow_json": ${workflow_json}
    }
}
EOF

    # Wait for directory to exist (from git clone), then write second file
    local benchmark_dir="$WORKSPACE_DIR/vast-pyworker/workers/comfyui-json/misc"
    while [[ ! -d "$benchmark_dir" ]]; do
        sleep 1
    done

    echo "$benchmark_json" > "$benchmark_dir/benchmark.json"

    # The SDK default is 1 warm-up + 8 measured runs. One run is enough to load the models,
    # so patch the worker before it starts (it is launched only after its dependencies install).
    local worker_py="$benchmark_dir/../worker.py"
    while [[ ! -f "$worker_py" ]]; do
        sleep 1
    done
    if ! grep -q 'runs=1, do_warmup=False' "$worker_py"; then
        sed -i 's/generator=make_benchmark_payload,/generator=make_benchmark_payload, runs=1, do_warmup=False,/' "$worker_py"
    fi
}

# Add a cron job to remove older (oldest +24 hours) output files if disk space is low
set_cleanup_job() {
    local script_dir="/opt/instance-tools/bin"
    local script_path="${script_dir}/clean-output.sh"
    
    # Ensure directory exists
    mkdir -p "$script_dir"
    
    if [[ ! -f "$script_path" ]]; then
        cat > "$script_path" << 'CLEAN_OUTPUT'
#!/bin/bash
output_dir="${WORKSPACE:-/workspace}/ComfyUI/output/"
min_free_mb=512
available_space=$(df -m "${output_dir}" | awk 'NR==2 {print $4}')
if [[ "$available_space" -lt "$min_free_mb" ]]; then
    oldest=$(find "${output_dir}" -mindepth 1 -type f -printf "%T@\n" 2>/dev/null | sort -n | head -1 | awk '{printf "%.0f", $1}')
    if [[ -n "$oldest" ]]; then
        cutoff=$(awk "BEGIN {printf \"%.0f\", ${oldest}+86400}")
        # Only delete files
        find "${output_dir}" -mindepth 1 -type f ! -newermt "@${cutoff}" -delete
        # Delete broken symlinks
        find "${output_dir}" -mindepth 1 -xtype l -delete
        # Now delete *empty* directories separately
        find "${output_dir}" -mindepth 1 -type d -empty -delete
    fi
fi
CLEAN_OUTPUT
        chmod +x "$script_path"
    fi

    # Check if cron job already exists (avoiding pipefail issues)
    local cron_exists=0
    if crontab -l 2>/dev/null | grep -qF 'clean-output.sh'; then
        cron_exists=1
    fi
    
    if [[ "$cron_exists" -eq 0 ]]; then
        # Add the cron job
        (crontab -l 2>/dev/null || true; echo "*/10 * * * * ${script_path}") | crontab -
    fi
}

main
