# Holdem Game (Qwen3-8B RL)

这个目录现在包含一套可落地的 Holdem agentic RL 训练入口：

- 训练脚本：`/Users/qitongli/Developer/slime/examples/holdem_game/run-qwen3-8B-finetuned.sh`
- 自定义 rollout/reward：`/Users/qitongli/Developer/slime/examples/holdem_game/generate_with_holdem.py`

## 这次改了什么（给后来者）

1. 训练脚本改成了单脚本双档位：
- `RUN_PROFILE=full`（默认，正式训练）
- `RUN_PROFILE=smoke`（小规模首跑验证）

2. 参数按联动关系收敛成常量，避免双脚本配置漂移：
- 资源拓扑（actor/rollout/ray）
- 并行（TP/PP/CP/EP/ETP）
- rollout/batch
- DAPO 动态采样
- SGLang 显存/并发

3. 默认算法路径：
- `grpo + decoupled clip (eps_clip + eps_clip_high)`
- dynamic sampling
- partial rollout

4. 补齐了缺失的 `generate_with_holdem.py`：
- `generate_with_holdem.generate`
- `generate_with_holdem.reward_func`

5. 清理策略从激进改为保守：
- 不再 `pkill -9 python`
- 仅回收 ray/sglang/slime router 相关进程

## 模块结构（拆分后）

为降低单文件复杂度，`generate_with_holdem.py` 现在是兼容入口薄层，真实逻辑按职责拆分：

1. 入口兼容层：
- `/Users/qitongli/Developer/slime/examples/holdem_game/generate_with_holdem.py`
- re-export 三个符号：`generate`、`reward_func`、`log_rollout_data`

2. rollout 逻辑：
- `/Users/qitongli/Developer/slime/examples/holdem_game/holdem_rollout.py`
- 负责 tool-calling 生成流程、token/logprob 写回、response-only loss mask 维护

3. reward 逻辑：
- `/Users/qitongli/Developer/slime/examples/holdem_game/holdem_reward.py`
- 负责协议校验、动作解析、分层奖励计算、异常保守降分

4. reward 配置：
- `/Users/qitongli/Developer/slime/examples/holdem_game/holdem_reward_config.py`
- 负责奖励常量（如动作距离矩阵、权重）

后续扩展建议：

1. 想改奖励规则，优先改 `holdem_reward.py` / `holdem_reward_config.py`，不碰 rollout。
2. 想改多轮生成或工具调用流程，优先改 `holdem_rollout.py`，不碰 reward。

## 环境前置

至少确认以下组件可用：

1. GPU / 驱动 / CUDA（目标假设：一机四卡，A100 40G 档位参数）
2. `ray`、`sglang`、训练依赖
3. `pybase64`（否则会在导入 `slime.rollout.sglang_rollout` 时报错）
4. 你的 Megatron 与 slime 运行路径可被脚本里的 `PYTHONPATH` 覆盖到

示例安装（按你的环境调整）：

```bash
pip install pybase64
```

## 数据与模型约定

1. 数据字段默认：
- `--input-key instruction`
- `--label-key output`

2. reward 返回结构化字典，训练取：
- `--reward-key score`

3. checkpoint 约定：
- `HF_CHECKPOINT`：HF 模型路径或仓库名（默认 `Stardust00/poker-qwen3-8b-merged`）
- `REF_LOAD`：Megatron `torch_dist` 路径（必填，默认占位值需改）

## 可直接运行的 smoke 命令模板（首跑）

在首跑时，优先走 smoke。下面就是可直接改路径执行的模板：

```bash
cd /Users/qitongli/Developer/slime/examples/holdem_game

RUN_PROFILE=smoke \
HF_CHECKPOINT=Stardust00/poker-qwen3-8b-merged \
REF_LOAD=/absolute/path/to/poker-qwen3-8b-merged_torch_dist \
SAVE_DIR=/absolute/path/to/output/holdem_qwen3_8b_smoke \
PROMPT_DATA=/absolute/path/to/holdem_train.jsonl \
WANDB_KEY=your_wandb_key \
bash /Users/qitongli/Developer/slime/examples/holdem_game/run-qwen3-8B-finetuned.sh
```

如果不想跑 wandb，可以不传 `WANDB_KEY`，并在脚本中去掉 `--use-wandb`。

## full 训练命令模板

```bash
cd /Users/qitongli/Developer/slime/examples/holdem_game

RUN_PROFILE=full \
HF_CHECKPOINT=Stardust00/poker-qwen3-8b-merged \
REF_LOAD=/absolute/path/to/poker-qwen3-8b-merged_torch_dist \
LOAD_CKPT=/absolute/path/to/previous_slime_ckpt_optional \
SAVE_DIR=/absolute/path/to/output/holdem_qwen3_8b_full \
PROMPT_DATA=/absolute/path/to/holdem_train.jsonl \
WANDB_KEY=your_wandb_key \
bash /Users/qitongli/Developer/slime/examples/holdem_game/run-qwen3-8B-finetuned.sh
```

## 与 profile 相关的默认行为

1. `RUN_PROFILE=full`
- `num-rollout=3000`
- `rollout-batch-size=16`
- `n-samples-per-prompt=8`
- `global-batch-size=128`
- `rollout-max-response-len=4096`

2. `RUN_PROFILE=smoke`
- `num-rollout=2`
- `rollout-batch-size=4`
- `n-samples-per-prompt=2`
- `global-batch-size=8`
- `rollout-max-response-len=512`
- 默认关 eval

## 已内置的关键联动与断言

脚本内置了这些约束检查，启动前不满足会直接退出：

1. `OVER_SAMPLING_BS >= ROLLOUT_BS`
2. `(ROLLOUT_BS * N_SAMPLES) % GLOBAL_BS == 0`
3. `RAY_NUM_GPUS == ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE`

## 监控与排障建议

优先关注这些指标：

1. 性能：
- `tokens_per_gpu_per_sec`
- `longest_sample_tokens_per_sec`

2. 一致性：
- `train/train_rollout_logprob_abs_diff`

3. 稳定性：
- `rollout/raw_reward`
- `train/ppo_kl`
- `grad_norm`

OOM 调参顺序：

1. 先降 `--max-tokens-per-gpu`
2. 再加 `--context-parallel-size`
3. 再降 `--sglang-mem-fraction-static`
4. 最后降 `--sglang-server-concurrency`

## 典型问题

1. `ModuleNotFoundError: No module named 'pybase64'`
- 安装 `pybase64` 后重试。

2. `loss mask length != response length`
- 通常是自定义 generate 没有维护 response-only loss mask；当前 `generate_with_holdem.py` 已按该约束实现。

3. prompt 解析失败
- 当前路径依赖 `instruction/output` 字段语义，不建议在该流程里开启 `--apply-chat-template`。
