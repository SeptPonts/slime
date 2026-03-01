#!/usr/bin/env bash

set -euo pipefail

log() {
  echo "[build_conda_a800] $*"
}

die() {
  echo "[build_conda_a800][ERROR] $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

check_url() {
  local url="$1"
  curl -I -L --connect-timeout 5 --max-time 15 "$url" >/dev/null 2>&1
}

clone_repo() {
  local target_dir="$1"
  local github_path="$2"
  shift 2
  local fallback_urls=("$@")
  local primary_url="${GIT_PRIMARY_PREFIX}${github_path}.git"
  local try_url

  if [ -d "${target_dir}/.git" ]; then
    log "reuse repo: ${target_dir}"
    return 0
  fi

  rm -rf "${target_dir}"
  log "clone primary: ${primary_url}"
  if git clone "${primary_url}" "${target_dir}"; then
    return 0
  fi

  if [ "${ENABLE_GITEE_FALLBACK}" != "1" ]; then
    die "clone failed and gitee fallback disabled: ${github_path}"
  fi

  for try_url in "${fallback_urls[@]}"; do
    [ -n "${try_url}" ] || continue
    rm -rf "${target_dir}"
    log "clone fallback: ${try_url}"
    if git clone "${try_url}" "${target_dir}"; then
      return 0
    fi
  done

  die "all clone sources failed for ${github_path}"
}

apply_patch_once() {
  local repo_dir="$1"
  local patch_file="$2"

  if git -C "${repo_dir}" apply --check "${patch_file}" >/dev/null 2>&1; then
    git -C "${repo_dir}" apply "${patch_file}"
    return 0
  fi

  if git -C "${repo_dir}" apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
    log "patch already applied: ${patch_file}"
    return 0
  fi

  die "patch cannot apply cleanly: ${patch_file}"
}

pip_install_git_with_fallback() {
  local primary_url="$1"
  local fallback_url="$2"
  local ref="$3"
  shift 3
  local pip_args=("$@")
  local spec

  spec="git+${primary_url}@${ref}"
  if micromamba run -n "${ENV_NAME}" pip install "${pip_args[@]}" "${spec}"; then
    return 0
  fi

  if [ "${ENABLE_GITEE_FALLBACK}" = "1" ] && [ -n "${fallback_url}" ]; then
    spec="git+${fallback_url}@${ref}"
    micromamba run -n "${ENV_NAME}" pip install "${pip_args[@]}" "${spec}"
    return 0
  fi

  die "pip install from git failed: ${primary_url}@${ref}"
}

# ----------------------------- configurable vars -----------------------------
ENV_NAME="${ENV_NAME:-slime}"
BASE_DIR="${BASE_DIR:-/root/autodl-tmp}"
WORK_DIR="${WORK_DIR:-${BASE_DIR}}"
CACHE_DIR="${CACHE_DIR:-${BASE_DIR}/.cache}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${BASE_DIR}/micromamba}"

CHECK_ONLY="${CHECK_ONLY:-0}"
MIN_SYS_GB="${MIN_SYS_GB:-2}"
MIN_DATA_GB="${MIN_DATA_GB:-100}"

CUDA_LINE="${CUDA_LINE:-12.8}" # allowed: 12.8 / 12.9
CUDA_CHANNEL_LABEL="${CUDA_CHANNEL_LABEL:-}"
TORCH_CUDA_TAG="${TORCH_CUDA_TAG:-}"

PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-}"
PYTORCH_INDEX_URL_FALLBACK="${PYTORCH_INDEX_URL_FALLBACK:-}"
ENABLE_PYTORCH_FALLBACK="${ENABLE_PYTORCH_FALLBACK:-1}"

GIT_PRIMARY_PREFIX="${GIT_PRIMARY_PREFIX:-https://gitclone.com/github.com/}"
ENABLE_GITEE_FALLBACK="${ENABLE_GITEE_FALLBACK:-1}"

SGLANG_COMMIT="${SGLANG_COMMIT:-24c91001cf99ba642be791e099d358f4dfe955f5}"
MEGATRON_COMMIT="${MEGATRON_COMMIT:-3714d81d418c9f1bca4594fc35f9e8289f652862}"
SLIME_REF="${SLIME_REF:-main}"

MAX_JOBS="${MAX_JOBS:-64}"

# Optional fallback URLs for git+pip dependencies.
MBRIDGE_GIT_URL="${MBRIDGE_GIT_URL:-${GIT_PRIMARY_PREFIX}ISEEKYAN/mbridge.git}"
MBRIDGE_GIT_URL_FALLBACK="${MBRIDGE_GIT_URL_FALLBACK:-}"
APEX_GIT_URL="${APEX_GIT_URL:-${GIT_PRIMARY_PREFIX}NVIDIA/apex.git}"
APEX_GIT_URL_FALLBACK="${APEX_GIT_URL_FALLBACK:-https://gh-proxy.com/https://github.com/NVIDIA/apex.git}"
TORCH_MEMORY_SAVER_GIT_URL="${TORCH_MEMORY_SAVER_GIT_URL:-${GIT_PRIMARY_PREFIX}fzyzcjy/torch_memory_saver.git}"
TORCH_MEMORY_SAVER_GIT_URL_FALLBACK="${TORCH_MEMORY_SAVER_GIT_URL_FALLBACK:-}"
MEGATRON_BRIDGE_GIT_URL="${MEGATRON_BRIDGE_GIT_URL:-${GIT_PRIMARY_PREFIX}fzyzcjy/Megatron-Bridge.git}"
MEGATRON_BRIDGE_GIT_URL_FALLBACK="${MEGATRON_BRIDGE_GIT_URL_FALLBACK:-}"

case "${CUDA_LINE}" in
  12.8)
    CUDA_CHANNEL_LABEL="${CUDA_CHANNEL_LABEL:-cuda-12.8.1}"
    TORCH_CUDA_TAG="${TORCH_CUDA_TAG:-cu128}"
    ;;
  12.9)
    CUDA_CHANNEL_LABEL="${CUDA_CHANNEL_LABEL:-cuda-12.9.1}"
    TORCH_CUDA_TAG="${TORCH_CUDA_TAG:-cu129}"
    ;;
  *)
    die "unsupported CUDA_LINE=${CUDA_LINE}, expected 12.8 or 12.9"
    ;;
esac

PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://mirror.sjtu.edu.cn/pytorch-wheels/${TORCH_CUDA_TAG}}"
PYTORCH_INDEX_URL_FALLBACK="${PYTORCH_INDEX_URL_FALLBACK:-https://download.pytorch.org/whl/${TORCH_CUDA_TAG}}"

mkdir -p "${BASE_DIR}" "${WORK_DIR}" "${CACHE_DIR}"
mkdir -p "${CACHE_DIR}/tmp" "${CACHE_DIR}/pip" "${CACHE_DIR}/huggingface" "${CACHE_DIR}/torch" "${CACHE_DIR}/xdg"

export TMPDIR="${CACHE_DIR}/tmp"
export XDG_CACHE_HOME="${CACHE_DIR}/xdg"
export PIP_CACHE_DIR="${CACHE_DIR}/pip"
export HF_HOME="${CACHE_DIR}/huggingface"
export TORCH_HOME="${CACHE_DIR}/torch"
export PIP_INDEX_URL

log "BASE_DIR=${BASE_DIR}"
log "WORK_DIR=${WORK_DIR}"
log "MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX}"
log "CUDA_LINE=${CUDA_LINE}, CUDA_CHANNEL_LABEL=${CUDA_CHANNEL_LABEL}, TORCH_CUDA_TAG=${TORCH_CUDA_TAG}"
log "PIP_INDEX_URL=${PIP_INDEX_URL}"
log "PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL}"
log "GIT_PRIMARY_PREFIX=${GIT_PRIMARY_PREFIX}, ENABLE_GITEE_FALLBACK=${ENABLE_GITEE_FALLBACK}"

require_cmd curl
require_cmd git
require_cmd awk
require_cmd sed
require_cmd tar
require_cmd nvidia-smi

sys_avail_kb="$(df -Pk / | awk 'NR==2{print $4}')"
data_avail_kb="$(df -Pk "${BASE_DIR}" | awk 'NR==2{print $4}')"
sys_min_kb="$((MIN_SYS_GB * 1024 * 1024))"
data_min_kb="$((MIN_DATA_GB * 1024 * 1024))"

[ "${sys_avail_kb}" -ge "${sys_min_kb}" ] || die "system disk too small: need >= ${MIN_SYS_GB}GB free"
[ "${data_avail_kb}" -ge "${data_min_kb}" ] || die "data disk too small at ${BASE_DIR}: need >= ${MIN_DATA_GB}GB free"

check_url "${PIP_INDEX_URL}" || die "pip index unreachable: ${PIP_INDEX_URL}"
check_url "${PYTORCH_INDEX_URL}" || log "pytorch mirror unreachable, will try fallback if enabled"
check_url "https://gitclone.com/" || log "gitclone unreachable, fallback may be required"

nvidia-smi >/dev/null

if [ "${CHECK_ONLY}" = "1" ]; then
  log "CHECK_ONLY=1, preflight checks passed."
  exit 0
fi

if ! command -v micromamba >/dev/null 2>&1; then
  log "micromamba not found, installing to ${BASE_DIR}/bin"
  mkdir -p "${BASE_DIR}/bin"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "${BASE_DIR}/bin" --strip-components=1 bin/micromamba
  export PATH="${BASE_DIR}/bin:${PATH}"
fi

export MAMBA_ROOT_PREFIX

if ! micromamba env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  micromamba create -n "${ENV_NAME}" python=3.12 pip -c conda-forge -y
fi

ENV_PREFIX="${MAMBA_ROOT_PREFIX}/envs/${ENV_NAME}"
export CUDA_HOME="${ENV_PREFIX}"

micromamba install -n "${ENV_NAME}" cuda cuda-nvtx cuda-nvtx-dev nccl -c "nvidia/label/${CUDA_CHANNEL_LABEL}" -y
micromamba install -n "${ENV_NAME}" -c conda-forge cudnn -y

micromamba run -n "${ENV_NAME}" pip install cuda-python==13.1.0

if ! micromamba run -n "${ENV_NAME}" pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url "${PYTORCH_INDEX_URL}"; then
  if [ "${ENABLE_PYTORCH_FALLBACK}" = "1" ]; then
    log "retry torch install with fallback index"
    micromamba run -n "${ENV_NAME}" pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url "${PYTORCH_INDEX_URL_FALLBACK}"
  else
    die "torch install failed and fallback disabled"
  fi
fi

clone_repo "${WORK_DIR}/sglang" "sgl-project/sglang" \
  "https://gitee.com/mirrors/sglang.git"
git -C "${WORK_DIR}/sglang" fetch --all --tags
git -C "${WORK_DIR}/sglang" checkout "${SGLANG_COMMIT}"
(cd "${WORK_DIR}/sglang" && micromamba run -n "${ENV_NAME}" pip install -e "python[all]")

micromamba run -n "${ENV_NAME}" pip install cmake ninja
MAX_JOBS="${MAX_JOBS}" micromamba run -n "${ENV_NAME}" pip -v install flash-attn==2.7.4.post1 --no-build-isolation

pip_install_git_with_fallback "${MBRIDGE_GIT_URL}" "${MBRIDGE_GIT_URL_FALLBACK}" "89eb10887887bc74853f89a4de258c0702932a1c" --no-deps
micromamba run -n "${ENV_NAME}" pip install --no-build-isolation "transformer_engine[pytorch]==2.10.0"
micromamba run -n "${ENV_NAME}" pip install flash-linear-attention==0.4.0

if ! NVCC_APPEND_FLAGS="--threads 4" micromamba run -n "${ENV_NAME}" pip -v install --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" "git+${APEX_GIT_URL}@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4"; then
  if [ "${ENABLE_GITEE_FALLBACK}" = "1" ] && [ -n "${APEX_GIT_URL_FALLBACK}" ]; then
    NVCC_APPEND_FLAGS="--threads 4" micromamba run -n "${ENV_NAME}" pip -v install --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" "git+${APEX_GIT_URL_FALLBACK}@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4"
  else
    die "apex install failed"
  fi
fi

pip_install_git_with_fallback "${TORCH_MEMORY_SAVER_GIT_URL}" "${TORCH_MEMORY_SAVER_GIT_URL_FALLBACK}" "dc6876905830430b5054325fa4211ff302169c6b" --no-cache-dir --force-reinstall
pip_install_git_with_fallback "${MEGATRON_BRIDGE_GIT_URL}" "${MEGATRON_BRIDGE_GIT_URL_FALLBACK}" "dev_rl" --no-build-isolation

micromamba run -n "${ENV_NAME}" pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation

clone_repo "${WORK_DIR}/Megatron-LM" "NVIDIA/Megatron-LM" \
  "https://gitee.com/mirrors/Megatron-LM.git"
git -C "${WORK_DIR}/Megatron-LM" fetch --all --tags --recurse-submodules=yes
git -C "${WORK_DIR}/Megatron-LM" checkout "${MEGATRON_COMMIT}"
git -C "${WORK_DIR}/Megatron-LM" submodule update --init --recursive
(cd "${WORK_DIR}/Megatron-LM" && micromamba run -n "${ENV_NAME}" pip install -e .)

SLIME_DIR=""
if [ -d "${PWD}/docker/patch/v0.5.7" ] && [ -f "${PWD}/setup.py" ]; then
  SLIME_DIR="${PWD}"
elif [ -d "${WORK_DIR}/slime/docker/patch/v0.5.7" ] && [ -f "${WORK_DIR}/slime/setup.py" ]; then
  SLIME_DIR="${WORK_DIR}/slime"
else
  clone_repo "${WORK_DIR}/slime" "THUDM/slime" \
    "https://gitee.com/THUDM/slime.git" \
    "https://gitee.com/mirrors/slime.git"
  git -C "${WORK_DIR}/slime" fetch --all --tags
  git -C "${WORK_DIR}/slime" checkout "${SLIME_REF}"
  SLIME_DIR="${WORK_DIR}/slime"
fi

(cd "${SLIME_DIR}" && micromamba run -n "${ENV_NAME}" pip install -e .)

micromamba run -n "${ENV_NAME}" pip install nvidia-cudnn-cu12==9.16.0.29
micromamba run -n "${ENV_NAME}" pip install "numpy<2"

apply_patch_once "${WORK_DIR}/sglang" "${SLIME_DIR}/docker/patch/v0.5.7/sglang.patch"
apply_patch_once "${WORK_DIR}/Megatron-LM" "${SLIME_DIR}/docker/patch/v0.5.7/megatron.patch"

log "done. env=${ENV_NAME}, base=${BASE_DIR}"
log "verify with: micromamba run -n ${ENV_NAME} python -c 'import torch; print(torch.__version__, torch.version.cuda)'"
