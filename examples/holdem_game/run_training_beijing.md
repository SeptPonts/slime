# A800 北京节点 Holdem 训练与 Smoke 执行手册（中文版）

## 0. 目标与原则

本手册目标是：先验证链路可跑通，再进入正式训练。
固定顺序：
1. Megatron-only（只测训练后端）
2. SGLang-only（只测 rollout/reward 后端）
3. Holdem smoke（最小连通性）

执行原则：
- 第一优先级是稳定，不是吞吐。
- 大文件（模型/数据/输出）全部放数据盘 `/root/autodl-tmp`。
- 中国大陆网络环境默认启用镜像策略。

## 1. 机器画像与路径约束

- GPU：`4 x NVIDIA A800-SXM4-80GB`（`sm_80`）
- 当前项目：`/root/autodl-tmp/slime`
- Megatron-LM：`/root/autodl-tmp/Megatron-LM`
- 建议数据盘：`/root/autodl-tmp`
- 常见额外挂载盘：`/root/autodl-fs`（如存在，可用于归档长期产物）

关键约束：
- 系统盘容量通常较小，不要把模型和 checkpoint 放 `/root` 下其他目录。
- 训练输出默认会很大（一次 smoke 也可能到 100GB 级别，取决于保存策略）。

## 2. 网络与凭据基线（每次开跑前）

```bash
cd /root/autodl-tmp/slime
set -a; source /root/autodl-tmp/slime/.env; set +a
export HF_ENDPOINT=https://hf-mirror.com
```

说明：
- `.env` 中有 `HF_TOKEN`、`WANDB_KEY`，但不会自动进入当前 shell，必须显式 `source`。
- 若 GitHub 拉取不稳定，可按需启用代理改写：

```bash
git config --local url."https://gitclone.com/github.com/".insteadof "https://github.com/"
```

## 3. 仓库同步策略

```bash
cd /root/autodl-tmp/slime
git remote set-url origin https://github.com/SeptPonts/slime.git
git remote add upstream https://github.com/THUDM/slime.git 2>/dev/null || true
git fetch --all --prune
git checkout main
git pull --ff-only origin main
```

验收：
- `git remote -v` 显示 `origin` 指向 `SeptPonts/slime`
- `git status` 干净

## 4. 资产路径约定

- HF 模型：`/root/autodl-tmp/models/poker-qwen3-8b-merged`
- 转换后权重：`/root/autodl-tmp/models/poker-qwen3-8b-merged_torch_dist`
- 训练集：`/root/autodl-tmp/data/pokerbench_train.jsonl`
- Megatron 调试子集：`/root/autodl-tmp/data/pokerbench_megatron_debug.jsonl`
- Holdem smoke 子集：`/root/autodl-tmp/data/pokerbench_holdem_smoke.jsonl`
- Smoke 输出：`/root/autodl-tmp/exp/holdem_smoke_beijing`

## 5. A800 保守参数覆盖（建议）

| 参数 | 建议值 | 理由 |
|---|---:|---|
| `ROLLOUT_NUM_GPUS_PER_ENGINE` | `2` | 4 卡上更稳的引擎切分 |
| `MAX_TOKENS_PER_GPU` | `6144`（smoke） | 比 8192 更稳，降低 OOM 概率 |
| `SGLANG_MEM_FRAC` | `0.45`（smoke） | 给系统和通信留余量 |
| `SGLANG_SERVER_CONCURRENCY` | `96`（smoke） | 避免 router 压满 |
| `ENABLE_DYNAMIC_SAMPLING` | `0`（smoke） | 先去掉动态复杂性 |
| `ENABLE_PARTIAL_ROLLOUT` | `0`（smoke） | 先做完整闭环验证 |

## 6. 已踩坑复盘（重点）

以下问题都在本机真实出现过，已给出可复用修复。

### 坑 1：`ray: command not found`

现象：
- 直接跑 `run-qwen3-8B-finetuned.sh` 报：`line xxx: ray: command not found`

根因：
- 当前 shell 不是带 micromamba 环境的登录态，`PATH` 没包含 `.../envs/slime/bin`。

修复：
```bash
export PATH=/root/autodl-tmp/micromamba/envs/slime/bin:$PATH
which ray
```

预防：
- 每次开跑前先 `which ray python3`，确保都指向 slime 环境。

### 坑 2：flashinfer JIT 链接失败 `cannot find -lcudart`

现象：
- SGLang 启动期或首轮 rollout 报 `cannot find -lcudart`。

根因：
- 运行时库搜索到了 `.../envs/slime/lib64`，但该目录不存在；
- 实际 `libcudart.so*` 在 `.../envs/slime/lib` 或 `/usr/local/cuda/lib64`。

修复：
```bash
ln -sfn /root/autodl-tmp/micromamba/envs/slime/lib /root/autodl-tmp/micromamba/envs/slime/lib64
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/root/autodl-tmp/micromamba/envs/slime/lib:$LD_LIBRARY_PATH
```

预防：
- 开跑前检查：
```bash
ls -l /root/autodl-tmp/micromamba/envs/slime/lib64
ldconfig -p | grep cudart || true
```

### 坑 3：`no_available_workers` / router 503 / 端口冲突

现象：
- 日志出现 `no_available_workers`、`503`、引擎无法分配。

根因：
- 上一次异常退出后遗留 `ray/train/sglang` 进程，占用资源或端口。

修复：
```bash
ray stop --force || true
pkill -f sglang || true
pkill -f train.py || true
```

预防：
- 每轮实验前执行一次清理，再启动新的 Ray head。

### 坑 4：HF 直连超时/慢

现象：
- 下载模型或数据频繁超时。

根因：
- 大陆网络到 HuggingFace 官方源不稳定。

修复：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

预防：
- 下载阶段默认走镜像；断点续传工具优先。

### 坑 5：`.env` 变量存在但作业里读不到

现象：
- 本地看得到 token，任务里仍提示未登录或 key 缺失。

根因：
- `.env` 仅在文件里，未导出到当前 shell 或 runtime env。

修复：
```bash
set -a; source /root/autodl-tmp/slime/.env; set +a
env | grep -E 'HF_TOKEN|WANDB_KEY'
```

预防：
- 所有训练命令统一从“已 source .env”的同一 shell 发起。

### 坑 6：收尾告警（非致命）

现象：
- 成功结束附近出现 `wandb BrokenPipeError`、`destroy_process_group() was not called`。

结论：
- 本次实际观测中是退出阶段告警，不影响 Ray job 成功。
- 先按“非阻塞告警”处理，后续在 full 训练阶段再单独清理。

## 7. 三阶段执行与验收

### 7.1 Megatron-only（先测训练）

目标：
- 不启动 SGLang，仅验证训练链路最小可用。

验收：
- 至少 1 step 成功；
- Ray job 正常退出；
- 日志中无 SGLang 初始化。

### 7.2 SGLang-only（再测 rollout）

目标：
- 不做 Megatron 更新，只测 SGLang + Holdem 自定义 generate/reward。

验收：
- `Application startup complete`；
- 有 `POST /generate ... 200`；
- 无 OOM / 无端口冲突。

### 7.3 Holdem smoke（最小连通性）

已验证可用的启动方式（A800 保守参数）：

```bash
cd /root/autodl-tmp/slime
set -a; source /root/autodl-tmp/slime/.env; set +a
export PATH=/root/autodl-tmp/micromamba/envs/slime/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/root/autodl-tmp/micromamba/envs/slime/lib:$LD_LIBRARY_PATH
ln -sfn /root/autodl-tmp/micromamba/envs/slime/lib /root/autodl-tmp/micromamba/envs/slime/lib64

ray stop --force || true
pkill -f sglang || true
pkill -f train.py || true

RUN_PROFILE=smoke \
HF_CHECKPOINT=/root/autodl-tmp/models/poker-qwen3-8b-merged \
REF_LOAD=/root/autodl-tmp/models/poker-qwen3-8b-merged_torch_dist \
PROMPT_DATA=/root/autodl-tmp/data/pokerbench_holdem_smoke.jsonl \
SAVE_DIR=/root/autodl-tmp/exp/holdem_smoke_beijing \
MAX_TOKENS_PER_GPU=6144 \
SGLANG_MEM_FRAC=0.45 \
SGLANG_SERVER_CONCURRENCY=96 \
ENABLE_DYNAMIC_SAMPLING=0 \
ENABLE_PARTIAL_ROLLOUT=0 \
OVER_SAMPLING_BS=8 \
bash /root/autodl-tmp/slime/examples/holdem_game/run-qwen3-8B-finetuned.sh
```

成功判据：
- Ray job `succeeded`
- 日志包含：
  - `Application startup complete`
  - `POST /generate ... 200`
  - `Final collected ... samples from rollout to train`
  - `[holdem][rollout 0]`
- 输出目录存在：`/root/autodl-tmp/exp/holdem_smoke_beijing`

## 8. 失败处理约定（当前版本）

按你的要求：
- 失败后不自动重试，不自动改参。
- 只做证据化采集并停住：
  - 首个致命报错
  - 报错上下文前后 100 行
  - job id 与时间戳
- 输出“可证据化根因 + 下一步最小修复动作”。

## 9. 当前状态快照（本轮结果）

- Holdem smoke 已成功跑通。
- 期间踩过的关键坑：`ray PATH`、`-lcudart`、残留进程冲突，均已落地修复。
- 产物目录：`/root/autodl-tmp/exp/holdem_smoke_beijing`
- 建议下一步：基于该稳定配置进入 full 训练，再逐步提升 token/concurrency。

## 10. 新增踩坑（full 续训阶段）

### 坑 7：从 checkpoint 续训时 `optimizer.load_state_dict` OOM

现象：
- 开启 `LOAD_CKPT` 后，训练在 actor 初始化阶段报 `torch.OutOfMemoryError`；
- 栈在 `optimizer.load_state_dict(...)`，发生在真正 rollout 前。

根因：
- 该 checkpoint 的优化器状态恢复内存峰值偏高；
- 在当前并行形态和 4xA800 场景下，直接恢复 optimizer/rng 可能导致瞬时显存不足。

修复（本次已验证可继续跑）：
- 续训时跳过 optimizer/rng 恢复，只恢复模型权重与训练进度：

```bash
NO_LOAD_OPTIM=1 \
NO_LOAD_RNG=1 \
LOAD_CKPT=/root/autodl-tmp/exp/holdem_full50_beijing \
RUN_PROFILE=full \
NUM_ROLLOUT_FULL=50 \
SGLANG_MEM_FRAC=0.70 \
ENABLE_DYNAMIC_SAMPLING=1 \
ENABLE_PARTIAL_ROLLOUT=1 \
SGLANG_SERVER_CONCURRENCY=128 \
bash /root/autodl-tmp/slime/examples/holdem_game/run-qwen3-8B-finetuned.sh
```

验证证据：
- 已出现 `Final collected 128 samples from rollout to train`；
- 已推进到 `rollout 40` 与 `step 40`；
- 当前 Ray job 处于 `RUNNING`。

影响说明：
- 这是“工程可用优先”的恢复策略，会丢失 optimizer 动量状态连续性；
- 但能确保在故障窗口快速恢复并继续训练。
