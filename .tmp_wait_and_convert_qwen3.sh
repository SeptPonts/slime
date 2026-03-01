#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR=/root/autodl-tmp/models/poker-qwen3-8b-merged
OUT_DIR=/root/autodl-tmp/models/poker-qwen3-8b-merged_torch_dist
cd /root/autodl-tmp/slime
set -a; source .env; set +a

echo "[$(date '+%F %T')] waiting for all safetensors shards..."
for _ in $(seq 1 360); do
  if [[ -f "$MODEL_DIR/model-00001-of-00004.safetensors" && -f "$MODEL_DIR/model-00002-of-00004.safetensors" && -f "$MODEL_DIR/model-00003-of-00004.safetensors" && -f "$MODEL_DIR/model-00004-of-00004.safetensors" ]]; then
    break
  fi
  sleep 10
done

if [[ ! -f "$MODEL_DIR/model-00001-of-00004.safetensors" || ! -f "$MODEL_DIR/model-00002-of-00004.safetensors" || ! -f "$MODEL_DIR/model-00003-of-00004.safetensors" || ! -f "$MODEL_DIR/model-00004-of-00004.safetensors" ]]; then
  echo "[$(date '+%F %T')] timeout waiting model shards"
  exit 1
fi

echo "[$(date '+%F %T')] model shards ready, start conversion"
source scripts/models/qwen3-8B.sh
PYTHONPATH=/root/autodl-tmp/Megatron-LM /root/autodl-tmp/micromamba/envs/slime/bin/python tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$MODEL_DIR" \
  --save "$OUT_DIR"

echo "[$(date '+%F %T')] conversion done"
