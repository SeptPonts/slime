#!/bin/bash

set -euo pipefail

# Conservative cleanup: stop Ray/sglang related processes only.
ray stop --force >/dev/null 2>&1 || true
pkill -f sglang >/dev/null 2>&1 || true
pkill -f "slime.router" >/dev/null 2>&1 || true
sleep 2

# Keep stdout/stderr unbuffered for Ray log streaming.
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
source "${REPO_ROOT}/scripts/models/qwen3-8B.sh"

RUN_PROFILE="${RUN_PROFILE:-full}"
if [[ "${RUN_PROFILE}" != "full" && "${RUN_PROFILE}" != "smoke" ]]; then
    echo "RUN_PROFILE must be one of: full, smoke"
    exit 1
fi

# -----------------------------
# Topology / parallel constants
# -----------------------------
ACTOR_NUM_NODES=1
ACTOR_NUM_GPUS_PER_NODE=4
ROLLOUT_NUM_GPUS_PER_ENGINE=2
RAY_NUM_GPUS=4
USE_COLOCATE=1

TP=2
PP=1
CP=1
EP=1
ETP=1

# -----------------------------
# Training constants (full)
# -----------------------------
NUM_ROLLOUT_FULL=3000
ROLLOUT_BS_FULL=16
N_SAMPLES_FULL=8
GLOBAL_BS_FULL=128
MAX_RESP_LEN_FULL=4096

# -----------------------------
# DAPO / perf constants
# -----------------------------
OVER_SAMPLING_BS=32
ENABLE_DYNAMIC_SAMPLING=1
ENABLE_PARTIAL_ROLLOUT=1
MAX_TOKENS_PER_GPU=8192
SGLANG_MEM_FRAC=0.45
SGLANG_SERVER_CONCURRENCY=256

# -----------------------------
# Checkpoint / data paths
# -----------------------------
HF_CHECKPOINT="${HF_CHECKPOINT:-Stardust00/poker-qwen3-8b-merged}"
REF_LOAD="${REF_LOAD:-/path/to/poker-qwen3-8b-merged_torch_dist}"
LOAD_CKPT="${LOAD_CKPT:-}"
SAVE_DIR="${SAVE_DIR:-/tmp/slime_holdem_qwen3_8b}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
PROMPT_DATA="${PROMPT_DATA:-/path/to/holdem_train.jsonl}"

# Optional eval dataset. Keep disabled by default.
ENABLE_EVAL="${ENABLE_EVAL:-0}"
EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-holdem_eval}"
EVAL_PROMPT_DATA="${EVAL_PROMPT_DATA:-}"

# -----------------------------
# Profile switch
# -----------------------------
if [[ "${RUN_PROFILE}" == "full" ]]; then
    NUM_ROLLOUT=${NUM_ROLLOUT_FULL}
    ROLLOUT_BS=${ROLLOUT_BS_FULL}
    N_SAMPLES=${N_SAMPLES_FULL}
    GLOBAL_BS=${GLOBAL_BS_FULL}
    MAX_RESP_LEN=${MAX_RESP_LEN_FULL}
else
    NUM_ROLLOUT=2
    ROLLOUT_BS=4
    N_SAMPLES=2
    GLOBAL_BS=8
    MAX_RESP_LEN=512
    ENABLE_EVAL=0
fi

ROLLOUT_NUM_GPUS=$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))

# -----------------------------
# Sanity checks for coupled args
# -----------------------------
if (( OVER_SAMPLING_BS < ROLLOUT_BS )); then
    echo "OVER_SAMPLING_BS (${OVER_SAMPLING_BS}) must be >= ROLLOUT_BS (${ROLLOUT_BS})"
    exit 1
fi

if (( (ROLLOUT_BS * N_SAMPLES) % GLOBAL_BS != 0 )); then
    echo "(ROLLOUT_BS * N_SAMPLES) must be divisible by GLOBAL_BS"
    exit 1
fi

if (( RAY_NUM_GPUS != ROLLOUT_NUM_GPUS )); then
    echo "RAY_NUM_GPUS (${RAY_NUM_GPUS}) must equal ACTOR_NUM_NODES*ACTOR_NUM_GPUS_PER_NODE (${ROLLOUT_NUM_GPUS})"
    exit 1
fi

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "${NVLINK_COUNT}" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi

echo "RUN_PROFILE=${RUN_PROFILE}"
echo "HAS_NVLINK: ${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"
echo "HF_CHECKPOINT=${HF_CHECKPOINT}"
echo "REF_LOAD=${REF_LOAD}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_LOAD}"
   --save "${SAVE_DIR}"
   --save-interval "${SAVE_INTERVAL}"
)
if [[ -n "${LOAD_CKPT}" ]]; then
    CKPT_ARGS+=(--load "${LOAD_CKPT}")
fi

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key instruction
   --label-key output
   --reward-key score
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BS}"
   --n-samples-per-prompt "${N_SAMPLES}"
   --rollout-max-response-len "${MAX_RESP_LEN}"
   --rollout-temperature 1
   --global-batch-size "${GLOBAL_BS}"
   --balance-data
)
if (( ENABLE_DYNAMIC_SAMPLING )); then
    ROLLOUT_ARGS+=(
        --over-sampling-batch-size "${OVER_SAMPLING_BS}"
        --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
    )
fi
if (( ENABLE_PARTIAL_ROLLOUT )); then
    ROLLOUT_ARGS+=(--partial-rollout)
fi

EVAL_ARGS=()
if (( ENABLE_EVAL )) && [[ -n "${EVAL_PROMPT_DATA}" ]]; then
    EVAL_ARGS=(
        --eval-interval 25
        --eval-prompt-data "${EVAL_DATASET_NAME}" "${EVAL_PROMPT_DATA}"
        --n-samples-per-eval-prompt 1
        --eval-max-response-len 1024
        --eval-top-p 1
    )
fi

PERF_ARGS=(
   --tensor-model-parallel-size "${TP}"
   --sequence-parallel
   --pipeline-model-parallel-size "${PP}"
   --context-parallel-size "${CP}"
   --expert-model-parallel-size "${EP}"
   --expert-tensor-parallel-size "${ETP}"
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.0
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-holdem
   --wandb-group qwen3-8B-holdem
)
if [[ -n "${WANDB_KEY:-}" ]]; then
    WANDB_ARGS+=(--wandb-key "${WANDB_KEY}")
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRAC}"
   --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_holdem.generate
   --custom-rm-path generate_with_holdem.reward_func
)

# OOM tuning order: max-tokens-per-gpu -> context-parallel-size ->
# sglang-mem-fraction-static -> sglang-server-concurrency.

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${RAY_NUM_GPUS}" \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}:${REPO_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

cd "${REPO_ROOT}"
TRAIN_ARGS=(
   --actor-num-nodes "${ACTOR_NUM_NODES}"
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}"
   --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
)
if (( USE_COLOCATE )); then
    TRAIN_ARGS+=(--colocate)
fi
TRAIN_ARGS+=(
   "${MODEL_ARGS[@]}"
   "${CKPT_ARGS[@]}"
   "${ROLLOUT_ARGS[@]}"
   "${OPTIMIZER_ARGS[@]}"
   "${GRPO_ARGS[@]}"
   "${WANDB_ARGS[@]}"
   "${PERF_ARGS[@]}"
   "${SGLANG_ARGS[@]}"
   "${MISC_ARGS[@]}"
   "${CUSTOM_ARGS[@]}"
)
if (( ${#EVAL_ARGS[@]} > 0 )); then
    TRAIN_ARGS+=("${EVAL_ARGS[@]}")
fi

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py "${TRAIN_ARGS[@]}"
