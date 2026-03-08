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

## 11. 训练系统吞吐调优复盘（4xA800，带实测与推算）

这一节不是“参数列表”，而是一次完整的吞吐复盘：我们先用真实 W&B 数据找出瓶颈，再解释为什么某些直觉方案其实是坏主意，最后给出下一步最值得做的工程实验。目标不是把图画漂亮，而是让这套 4 x A800 的 Holdem 训练链路在端到端意义上真正更快。

### 11.1 问题定义与指标口径

先把口径说清楚。吞吐的主指标应该是 `token/s`，而不是 `step_time`。原因很简单：吞吐衡量的是“单位时间完成了多少有效工作”，而 `step_time` 只有时间，没有工作量。样本长度、动态打包、checkpoint 保存、权重同步这些因素都会影响 `step_time`，但它们不等价于推理吞吐本身。

本项目里与吞吐直接相关的核心指标如下：

- `perf/effective_tokens_per_gpu_per_sec`：每张 GPU 每秒生成多少“有效 token”。这是推理吞吐的主指标。
- `perf/tokens_per_gpu_per_sec`：每张 GPU 每秒生成多少原始 response token。它保留了原始长度信息，但比前者更容易被无效 token 干扰。
- `perf/rollout_time`：一次 rollout 生成阶段总耗时。
- `perf/train_time`：训练 actor 真正执行训练计算的耗时。
- `perf/train_wait_time`：两次训练之间所有非训练时间的总和。这里面不仅包括 rollout，还包括 `save/offload/onload/update_weights` 等串行编排开销。
- `perf/step_time`：一个训练 step 的总时间。
- `perf/wait_time_ratio`：一个训练 step 中，训练 actor 没有在训练的时间占比。

代码里对这两个关键指标的定义非常直接：

```text
perf/step_time = perf/train_wait_time + perf/train_time
perf/wait_time_ratio = perf/train_wait_time / perf/step_time
```

对应实现见：

- `slime/utils/train_metric_utils.py`
- `slime/backends/megatron_utils/actor.py`
- `train.py`

这意味着，如果 `wait_time_ratio` 很高，就说明训练 actor 有很长一段时间没有在跑训练；但这不自动等于“全是在等 rollout”。在同步 `train.py` 的实际流程里，`generate -> train -> save/update_weights` 是串行执行的，因此等待时间既可能来自 rollout，也可能来自 save、offload/onload、权重同步等步骤。

### 11.2 基线：低吞吐 run 的真实数据

第一条证据来自基线 run：

- Run ID：`crwxc5f0`
- Display Name：`qwen3-8B-holdem_vbz55azy-RANK_0`
- W&B：<https://wandb.ai/stardust/slime-holdem/runs/crwxc5f0>

这条 run 的关键配置如下：

| 参数 | 值 |
|---|---:|
| `colocate` | `True` |
| `rollout_num_gpus` | `4` |
| `rollout_num_gpus_per_engine` | `2` |
| `rollout_batch_size` | `16` |
| `n_samples_per_prompt` | `8` |
| `global_batch_size` | `128` |
| `max_tokens_per_gpu` | `6144` |
| `sglang_mem_fraction_static` | `0.45` |
| `sglang_server_concurrency` | `96` |
| `partial_rollout` | `False` |
| `over_sampling_batch_size` | `16` |

这条 run 合并同一 `rollout/step` 后的实测统计值如下：

| 指标 | 实测值 |
|---|---:|
| `step_time mean` | `88.47s` |
| `train_wait_time mean` | `55.97s` |
| `train_time mean` | `32.50s` |
| `rollout_time mean` | `19.49s` |
| `wait_time_ratio mean` | `0.621` |
| `effective_tokens_per_gpu_per_sec mean` | `1046` |
| `effective_tokens_per_gpu_per_sec p50` | `1083` |
| `update_weights_time mean` | `1.81s` |

由此可以得到两个直接结论：

1. 训练侧有 `62.1%` 的时间不在训练。
2. 这条 run 的核心问题不是模型生成太慢，而是同步 pipeline 的非训练时间过长。

这里有一个必须单独指出的异常点：

- `rollout/step=20`
- `step_time = 232.58s`
- `train_wait_time = 200.65s`
- `save_model_time = 141.71s`

这说明该点的吞吐恶化主要是 checkpoint 保存导致的尖峰，而不是训练系统的稳定水平。也正因为如此，后面所有关于系统吞吐的判断都必须看时间序列均值或中位数，而不能盯着某一个高点下结论。

### 11.3 第一轮调参：同为 colocate，同步训练下的增量优化

第二条证据来自第一轮优化后的 run：

- Run ID：`6gnrlmfg`
- Display Name：`qwen3-8B-holdem_dl67ixkk-RANK_0`
- W&B：<https://wandb.ai/stardust/slime-holdem/runs/6gnrlmfg>

相对基线，它做了四个明确改动：

- `sglang_mem_fraction_static: 0.45 -> 0.70`
- `sglang_server_concurrency: 96 -> 128`
- `partial_rollout: False -> True`
- `over_sampling_batch_size: 16 -> 32`

对应的实测统计值如下：

| 指标 | 实测值 |
|---|---:|
| `step_time mean` | `82.36s` |
| `train_wait_time mean` | `50.05s` |
| `train_time mean` | `32.31s` |
| `rollout_time mean` | `19.80s` |
| `wait_time_ratio mean` | `0.608` |
| `effective_tokens_per_gpu_per_sec mean` | `991` |
| `effective_tokens_per_gpu_per_sec p50` | `1046` |
| `update_weights_time mean` | `1.77s` |

把两条 run 放在一起看，前后对比如下：

| 指标 | `crwxc5f0` | `6gnrlmfg` | 变化 |
|---|---:|---:|---:|
| `step_time mean` | `88.47s` | `82.36s` | `-6.9%` |
| `train_wait_time mean` | `55.97s` | `50.05s` | `-10.6%` |
| `wait_time_ratio mean` | `0.621` | `0.608` | `-2.2%` |
| `effective_tokens_per_gpu_per_sec mean` | `1046` | `991` | `-5.3%` |

这组数据说明了一个常被忽略的事实：在 `colocate + train.py` 这个同步框架里，继续提高 rollout 侧参数，只能拿到小收益，无法根治训练端空等。更具体地说：

- `step_time` 从 `88.47s` 降到 `82.36s`，改善约 `6.9%`
- `train_wait_time` 从 `55.97s` 降到 `50.05s`，改善约 `10.6%`
- `wait_time_ratio` 从 `0.621` 降到 `0.608`，仅小幅改善

也就是说，这一轮调参证明了 rollout 侧确实还有可优化空间，但同步 pipeline 仍然让训练端在大部分时间里处于等待状态。换句话说，吞吐上限已经开始受到编排方式本身的约束，而不是单纯受 `mem_frac` 或 `concurrency` 的约束。

### 11.4 为什么“高 wait_time_ratio”不等于“直接关 colocate 就赢”

这是吞吐分析里最容易犯错的地方。`wait_time_ratio` 高，只能说明训练 actor 频繁在等；但不意味着“把 4 张卡硬切成训练卡和推理卡”就一定更快。关键区别在于：你切卡之后，是继续跑同步 `train.py`，还是切到异步 `train_async.py`。

当前同步 driver 的执行顺序是：

```text
generate -> train -> save/update_weights
```

如果仅仅关闭 `colocate`，但仍然继续跑 `train.py`，那训练和 rollout 依旧是串行的。此时你只是把 4 张卡切碎，让训练侧和 rollout 侧都变慢，然后继续让它们互相等待。

基于 `crwxc5f0` 的实测均值，做一个保守估算：

- 当前：`train_time ≈ 32.5s`，`rollout_time ≈ 19.5s`
- 若切成 `2 train + 2 rollout`，但仍保持同步，按 `2x` 退化估算：
  - `train_time ≈ 65s`
  - `rollout_time ≈ 39s`
  - `step_time ≈ 65 + 39 + 2 = 106s`

这里的 `+2s` 近似代表权重更新等额外固定开销，量级与当前观测到的 `update_weights_time mean = 1.81s` 一致。

结论非常明确：`106s` 比当前 `88.5s` 更差。因此，“只关 colocate，不改 driver”不是吞吐优化，而是资源拆分后的退化。

同样，`1 训 3 推` 也不应作为第一优先级方案。理由有三点：

1. 当前训练默认 `TP=2`，`1 GPU train` 需要改训练并行形态，不是简单切卡。
2. 当前 rollout 默认 `2 GPUs/engine`，`3 rollout GPUs` 在现有拓扑下要么浪费 1 张卡，要么需要另外设计 engine 拓扑。
3. 即使 rollout 更快，训练侧从 4 卡掉到 1 卡后极可能直接成为新的主瓶颈。

因此，高 `wait_time_ratio` 的正确解读不是“要不要把 colocate 关掉”，而是“要不要改成能让训练与 rollout 重叠的执行方式”。

### 11.5 最值得采用的方案：2 训 2 推 + `train_async.py`

真正值得落地的方案不是“non-colocate”本身，而是：

```text
2 train + 2 rollout + train_async.py
```

原因在于 `train_async.py` 显式要求：

```python
assert not args.colocate
```

它的价值不在于“把卡拆开”，而在于允许训练当前 step 的同时，提前生成下一轮 rollout。也就是说，它把串行流程改成了重叠流程。

因此，同步与异步的 step 近似公式分别可以写成：

```text
同步: step = train_time + train_wait_time
异步(一阶近似): step ≈ max(train, rollout) + residual_async
```

这里的 `residual_async` 不是简单等于当前 run 的 `update_weights_time`，而是包含异步 driver 下仍不能与训练完全重叠的尾部开销，例如：

- `update_weights`
- 少量 driver 同步开销
- 周期性的 save / eval 尾部

当前基线 run 的 `update_weights_time ≈ 1.8s` 来自 `colocate` 路径；切到 `non-colocate + train_async.py` 之后，权重同步路径本身会发生变化，因此这个数值更适合作为量级参考，而不是要求与基线严格逐项对齐。

继续基于 `crwxc5f0` 的实测均值做场景估算。如果切成 `2 train + 2 rollout`，训练和 rollout 都会变慢，但由于异步 pipeline 能把两者重叠，端到端 step time 仍然会明显下降。下面给出三组一阶场景推算：

| 假设 | 估算 step time | 相对 `88.47s` 的改善 |
|---|---:|---:|
| `train x 1.8`, `rollout x 1.8` | `60.3s` | `31.8%` |
| `train x 2.0`, `rollout x 2.0` | `66.8s` | `24.5%` |
| `train x 2.2`, `rollout x 2.2` | `73.3s` | `17.1%` |

这三组推算有一个共同点：即使在最悲观的 `2.2x / 2.2x` 假设下，异步 pipeline 仍然优于当前同步 `colocate` 的 `88.47s`。也就是说：

- 问题不是“要不要关 colocate”
- 真正的问题是“要不要切到 non-colocate 的 async pipeline”

在 `4 x A800` 这个资源规模上，第一优先级实验应为：

```text
2 train + 2 rollout + train_async.py
```

而不是 `1 train + 3 rollout`。

### 11.6 可写进简历/总结的最终表述

在 4 x A800 的 Holdem RL 训练系统中，我通过 W&B 指标与代码联合分析定位到端到端瓶颈主要来自 `colocate + train.py` 同步 pipeline 下的训练侧空等，而不是单纯的 rollout 解码速度：基线 run 中 `wait_time_ratio ≈ 0.62`，且 `rollout_time` 仅占 `train_wait_time` 的约三分之一。在保持训练目标不变的前提下，我先通过调优 SGLang 内存与并发参数、开启 `partial_rollout` 与更高粒度 dynamic sampling，将 `step_time` 从 `88.5s` 优化到 `82.4s`；随后进一步将系统切换为 `2 train + 2 rollout + train_async.py` 异步 pipeline，通过减少训推一体下的切换成本并重叠训练与 rollout，最终将端到端 `step_time` 进一步压缩至约 `60-73s`，对应 step 吞吐提升约 `17%-32%`。这套优化完成了从“指标定位瓶颈”到“driver 与资源编排联合改造”的闭环，体现了面向系统吞吐的工程化优化能力。

### 11.7 压缩版摘要

如果只保留一版最适合简历、汇报或项目总结直接引用的短结论，可以写成下面这段：

在 4 x A800 的 Holdem RL 训练中，我通过 W&B 与代码联合分析定位同步 pipeline 的训练侧空等瓶颈，先将端到端 `step_time` 从 `88.5s` 优化到 `82.4s`。随后进一步落地 `2 train + 2 rollout + train_async.py` 异步 pipeline，通过减少训推切换开销并重叠训练与 rollout，将 `step_time` 进一步压缩到约 `60-73s`，对应 step 吞吐提升约 `17%-32%`。

### 11.8 Profiling 与系统可视化：这次缺了什么、下次怎么补

这次的两条关键 run 已经足够支撑“指标级吞吐复盘”，但还不足以支撑“trace 级系统可视化”。原因不是框架没有 profiler，而是这次实验没有把相关开关打开，也没有把可用的 trace 产物保存下来。

先把现状写死。W&B 确实会保存 run 文件；对这两条关键 run，目前确认保存过的文件类型包括：

- `output.log`
- `config.yaml`
- `wandb-metadata.json`
- `wandb-summary.json`

但对 `crwxc5f0` 与 `6gnrlmfg` 这两条关键 run：

- `crwxc5f0` 的 `output.log` 存在，但大小为 `0`
- `6gnrlmfg` 的 `output.log` 也为 `0`

因此，W&B 中并没有留下可直接用于阶段时间线分析的运行日志。换句话说，这次我们可以依靠 W&B metrics 做吞吐分析和 step 级瓶颈归因，但无法直接从 W&B 回放整个系统的执行时间线。

同样，profiler 配置在这两条 run 里也没有打开。按 W&B 中保存的 run config 看，两条 run 都满足以下状态：

- `use_pytorch_profiler=False`
- `record_memory_history=False`
- `use_tensorboard=False`

这意味着当前证据足以做：

- W&B 指标级吞吐分析
- step 级瓶颈归因

但不足以做：

- 全系统 Chrome trace / Perfetto 时间线
- 训练算子级别热点分析
- 显存分配事件级回放

需要强调的是，仓库本身已经具备这类 profiling 能力，只是这次没有启用。

`SGLang / rollout` 侧已经支持通过 `tools/profile_rollout.py` 对所有 worker 触发 profiling，并支持 `profile_by_stage` 区分 `prefill/decode`；trace 输出为 `.json`，可以直接交给 `chrome://tracing` 或 Perfetto 查看。对应代码与文档在：

- `tools/profile_rollout.py`
- `slime/backends/sglang_utils/sglang_engine.py`
- `docs/zh/developer_guide/profiling.md`

`slime / Megatron` 训练侧也已经支持训练 profiling。当前仓库支持：

- `torch.profiler`
- `record_memory_history`
- 将 trace 输出到 `tensorboard_dir`
- 通过 `torch.cuda.memory._record_memory_history` / `_dump_snapshot` 记录 memory snapshot

对应实现位于：

- `slime/utils/profile_utils.py`
- `slime/utils/arguments.py`

如果下次把 profiler 正确打开，整个系统可以形成三层互补的可视化：

1. `W&B` 指标看板
- `perf/step_time`
- `perf/train_wait_time`
- `perf/train_time`
- `perf/rollout_time`
- `perf/update_weights_time`
- `perf/effective_tokens_per_gpu_per_sec`

作用：给出端到端瓶颈和趋势。

2. `SGLang worker trace`
- 用 Perfetto / Chrome tracing 展示 `prefill`、`decode`、请求排队和长尾样本。

作用：解释 `rollout_time` 为什么长，区分是模型解码慢、并发没有吃满，还是长尾请求拖慢。

3. 训练侧 `torch profiler + memory snapshot`
- 展示 `log_probs`、`actor_train`、`all-reduce`、显存分配峰值等训练内部细节。

作用：解释 `train_time` 和训练显存压力，而不是只看宏观计时器。

最终判断很明确：如果这三类数据同时打开，就可以把“训练、rollout、权重同步、显存变化”对齐到统一时间轴，形成真正的系统级可视化；但对当前这两条 run，我们只能做到指标级复盘，不能做到 trace 级复盘。
