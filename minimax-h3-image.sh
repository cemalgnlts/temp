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

# Model declarations: "URL|OUTPUT_PATH"
HF_MODELS=(
  "https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/main/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors|$MODELS_DIR/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors"
  "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors|$MODELS_DIR/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
  "https://huggingface.co/Comfy-Org/Qwen3.5/resolve/main/text_encoders/qwen3.5_9b_w4a8_gptq.safetensors|$MODELS_DIR/text_encoders/qwen3.5_9b_w4a8_gptq.safetensors"
  "https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/main/vae/minimax_h3_image_vae_fp16.safetensors|$MODELS_DIR/vae/minimax_h3_image_vae_fp16.safetensors"
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
    log "Starting ComfyUI provisioning..."
    
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
    #mkdir -p "$MODELS_DIR"/{checkpoints,text_encoders,latent_upscale_models,loras}
    mkdir -p "$MODELS_DIR"/{diffusion_models,text_encoders,vae}

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

# This workflow is as provided by ComfyUI template browser
# Adjustments as recommended at https://huggingface.co/Comfy-Org/ACE-Step_ComfyUI_repackaged/discussions/1#6845136255e580607333edda, converted to API format
write_api_workflow() {
    # Define the workflow JSON once
    local default_prompt="A red fox sitting in a snowy forest at sunrise, soft golden light, detailed fur, photorealistic"
    local BENCH_PROMPT="A red fox in a snowy forest"
    local workflow_json
    read -r -d '' workflow_json << 'WORKFLOW_JSON' || true
{
  "1": {
    "class_type": "UNETLoader",
    "inputs": {
      "unet_name": "minimax_h3_image_pruned_int8_convrot.safetensors",
      "weight_dtype": "default"
    }
  },
  "2": {
    "class_type": "CLIPLoader",
    "inputs": {
      "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
      "type": "minimax",
      "device": "default"
    }
  },
  "3": {
    "class_type": "VAELoader",
    "inputs": {
      "vae_name": "minimax_h3_image_vae_fp16.safetensors"
    }
  },
  "4": {
    "class_type": "MiniMaxH3ImageConditioning",
    "inputs": {
      "clip": ["2", 0],
      "vae": ["3", 0],
      "prompt": "A red fox sitting in a snowy forest at sunrise, soft golden light, detailed fur, photorealistic",
      "width": 2048,
      "height": 2048
    }
  },
  "5": {
    "class_type": "ConditioningZeroOut",
    "inputs": {
      "conditioning": ["4", 0]
    }
  },
  "6": {
    "class_type": "KSampler",
    "inputs": {
      "model": ["1", 0],
      "positive": ["4", 0],
      "negative": ["5", 0],
      "latent_image": ["4", 1],
      "seed": "__RANDOM_INT__",
      "steps": 6,
      "cfg": 1,
      "sampler_name": "euler",
      "scheduler": "simple",
      "denoise": 1
    }
  },
  "7": {
    "class_type": "VAEDecode",
    "inputs": {
      "samples": ["6", 0],
      "vae": ["3", 0]
    }
  },
  "8": {
    "class_type": "SaveImage",
    "inputs": {
      "images": ["7", 0],
      "filename_prefix": "minimax_h3"
    }
  }
}
WORKFLOW_JSON

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

    # Warm-up payload: same graph (so every model gets loaded), small size and a short prompt
    local benchmark_json="$workflow_json"
    benchmark_json="${benchmark_json//\"width\": 2048/\"width\": 1024}"
    benchmark_json="${benchmark_json//\"height\": 2048/\"height\": 1024}"
    benchmark_json="${benchmark_json//$default_prompt/$BENCH_PROMPT}"

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
