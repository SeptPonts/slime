# slime 文档系统化总结（重组版）

> 目标：在不沿用原目录结构的前提下，给出可执行、可检索、带出处的系统总结，重点覆盖配置参数与性能/故障排查。

## 1. 系统全景与数据流

slime 的核心定位是“RL 后训练框架”，通过 Ray 把训练（Megatron/FSDP）与推理采样（SGLang + router）组织成统一闭环；它强调推理接口可定制、训练后端可替换、训推可同卡/分卡、训练循环可同步/异步。[来源](/Users/qitongli/Developer/slime/docs/zh/index.rst:3) [来源](/Users/qitongli/Developer/slime/docs/zh/blogs/introducing_slime.md:35) [来源](/Users/qitongli/Developer/slime/docs/zh/blogs/introducing_slime.md:39)

从执行视角，主循环可抽象为：数据采样（rollout）-> 参数更新（train）-> 权重同步回推理端 -> 下一轮采样；并通过 `--num-rollout` 控制总轮次。[来源](/Users/qitongli/Developer/slime/docs/zh/get_started/quick_start.md:153) [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/quick_start.md:175)

在架构边界上，slime 默认支持训推分离（分别分配 GPU），也支持 `--colocate` 的训推一体；并能按需求切换 `train.py`（同步）与 `train_async.py`（管线化）。[来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:28) [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:32) [来源](/Users/qitongli/Developer/slime/docs/zh/examples/qwen3-4B.md:297)

## 2. 配置参数总览（重点）

### 2.1 资源与拓扑（actor/rollout/critic/colocate/prefill）

资源编排主轴：`--actor-num-nodes`、`--actor-num-gpus-per-node`、`--rollout-num-gpus`、`--rollout-num-gpus-per-engine`；其中 engine 粒度参数近似于 SGLang 的 `tp_size` 概念。[来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:16) [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:24)

PPO 场景要额外预算 critic 并行资源（`--critic-num-nodes` / `--critic-num-gpus-per-node`），资源不是“复用 actor”，而是并列申请；总卡数要显式核算。[来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:230) [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:251)

`--colocate` 开启后会忽略 `--rollout-num-gpus`，并把训推放在同一组卡上；PD 分离由 `--prefill-num-servers` 指定 prefill server 数量，文档建议在多轮/agentic RL 中开启。[来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:32) [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/pd-disaggregation.md:5) [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/pd-disaggregation.md:7)

### 2.2 模型与检查点（hf-checkpoint/ref-load/load/save/ckpt-step/ckpt-format）

Megatron 侧关键现实：不能直接依赖 HF ckpt 配置模型，需手动提供模型结构参数并加载 Megatron ckpt；推荐 `torch_dist` 格式（并行切分更灵活）。[来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:45) [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:115)

加载语义上：`--ref-load` 是 reference 模型，`--load` 是 actor 初始化来源（缺失时回退到 ref），`--save` 是 actor 存储路径；步数由 `--ckpt-step` 或 `latest_checkpointed_iteration.txt` 决定。[来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:131) [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:135)

`--hf-checkpoint` 在训练流程中主要用于 SGLang 初始化（含 tokenizer），训练首步前会同步 Megatron 权重到 SGLang，因此续训通常不要求 hf ckpt 跟训练步对齐。[来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:147) [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:151)

### 2.3 rollout/eval/data（批量关系、采样参数、字段映射、metadata）

rollout 和 train 的样本守恒关系是第一约束：
`(rollout_batch_size × n_samples_per_prompt) = (global_batch_size × num_steps_per_rollout)`；`global_batch_size` 会自动推导或被校验。[来源](/Users/qitongli/Developer/slime/docs/zh/get_started/quick_start.md:170) [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/quick_start.md:172)

训练采样主参数：`--rollout-batch-size`、`--n-samples-per-prompt`、`--rollout-max-response-len`、`--rollout-temperature`；评估可用 `--eval-*` 参数覆盖训练采样策略。[来源](/Users/qitongli/Developer/slime/docs/zh/get_started/quick_start.md:156) [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/quick_start.md:209)

数据输入统一要求 `.jsonl`（每行 json），通过 `--input-key`、`--label-key`、`--apply-chat-template` 映射；额外上下文用 `metadata`/`--metadata-key` 传入，供多轮生成与奖励函数读取。[来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:160) [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:184) [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/quick_start.md:467)

### 2.4 并行与显存性能（tp/pp/cp/ep/etp/recompute/dynamic-batch/max_tokens）

Megatron 并行主线是 `tp/sp/pp/cp/ep/etp`；重计算用 `--recompute-granularity`、`--recompute-method`、`--recompute-num-layers`。 [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:94) [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:100)

slime 特有高频性能参数是 `--use-dynamic-batch-size` + `--max-tokens-per-gpu`：前者开启动态打包并忽略 `--micro-batch-size`，后者定义每卡 token 上限，CP 模式下按 `CP * max_tokens_per_gpu` 共享总长度预算。[来源](/Users/qitongli/Developer/slime/docs/zh/get_started/quick_start.md:228) [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/quick_start.md:230)

### 2.5 RL 算法与损失（advantage/KL/TIS/per-token-loss/PPO critic）

算法入口是 `--advantage-estimator`（GRPO/GSPO/Reinforce++/PPO 等）；`--calculate-per-token-loss` 决定按 token 均值而非样本均值归约；`--use-tis` 对 off-policy 校正有效。[来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:188) [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:194)

KL 相关参数（`--use-kl-loss`、`--kl-loss-coef` 等）既可用于正则项，也可仅做观测指标（coef=0）。[来源](/Users/qitongli/Developer/slime/docs/zh/get_started/quick_start.md:258)

PPO 与 GRPO 的关键差异在于 critic 资源与参数组（critic load/save/lr/value-clip 等）；PPO 资源预算必须提前拆分。 [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:222) [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:253)

### 2.6 SGLang / Router / MoE（--sglang-* 透传、R3、会话亲和、并发与 mem-fraction）

SGLang 参数通过 `--sglang-*` 前缀透传；但与资源调度直接耦合的参数由 slime 接管（如 `tp-size` 对应 `--rollout-num-gpus-per-engine`，`model-path` 对应 `--hf-checkpoint`）。[来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:344) [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:350)

router 分两类：默认 SGLang Model Gateway；启用 `--use-slime-router` 可切到 SlimeRouter 以保留训练所需 metadata（尤其 MoE R3 的 `routed_experts`）。[来源](/Users/qitongli/Developer/slime/docs/zh/advanced/slime-router.md:21) [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/slime-router.md:73)

R3 的硬约束是 `--use-slime-router --use-rollout-routing-replay` 联合开启；会话亲和由 `--sglang-router-policy consistent_hashing` 驱动。 [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/slime-router.md:66) [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/slime-router.md:123)

`--sglang-mem-fraction-static` 是训推争抢显存时的关键阀门；并发上可用 `--sglang-server-concurrency` 约束 server 侧压力，或通过 `--sglang-cuda-graph-bs` 扩充图配置覆盖范围。 [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:153) [来源](/Users/qitongli/Developer/slime/docs/zh/examples/qwen3-4B.md:281)

### 2.7 自定义扩展接口（rollout/custom_generate/custom_rm/...）

slime 的扩展主方式是“函数路径注入”，覆盖点贯穿 rollout、样本过滤、奖励、损失、数据转换、日志、数据源、megatron hook、router middleware。[来源](/Users/qitongli/Developer/slime/docs/zh/get_started/customization.md:3) [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/customization.md:9)

最关键的三个接口是：
- `--rollout-function-path`（替换整个 rollout 流程）；
- `--custom-generate-function-path`（替换生成步骤）；
- `--custom-rm-path`（替换奖励计算）。
这些接口对多轮 agent、tool-calling、RAG 是第一入口。[来源](/Users/qitongli/Developer/slime/docs/zh/get_started/customization.md:35) [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/customization.md:55) [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/customization.md:75)

### 2.8 examples 参数增量（docs 弱覆盖）

下表只收录 examples 里高频、且对训练闭环有直接影响的参数组，用于补齐“文档主线提到但实战细节不够”的部分。

| 参数名 | 作用 | 联动约束 | 风险点 | 来源 |
| --- | --- | --- | --- | --- |
| `--custom-config-path` | 注入多轮/环境配置（如交互环境路径、max_turns）。 | 常与 `--custom-generate-function-path` 配套。 | 配置与 rollout 函数不匹配会导致运行期错误。 | [来源](/Users/qitongli/Developer/slime/examples/geo3k_vlm_multi_turn/run_geo3k_vlm_multi_turn.py:63) |
| `--custom-reward-post-process-path` | 对 reward 结果做二次处理（如写入 teacher logprobs）。 | OPD-sglang 路径通常与 `--custom-rm-path`、`--rm-url` 联动。 | 不处理 token 对齐会让蒸馏信号失真。 | [来源](/Users/qitongli/Developer/slime/examples/on_policy_distillation/run-qwen3-8B-opd.sh:74) [来源](/Users/qitongli/Developer/slime/examples/on_policy_distillation/on_policy_distillation.py:26) |
| `--custom-tis-function-path` | 指定 TIS/MIS 权重计算函数。 | 常与 `--use-tis`、`--custom-config-path` 联动。 | 算法层级/边界配错会导致方差暴增或过度截断。 | [来源](/Users/qitongli/Developer/slime/examples/train_infer_mismatch_helper/run-qwen3-4b-mis.sh:125) [来源](/Users/qitongli/Developer/slime/examples/train_infer_mismatch_helper/mis.yaml:9) |
| `--deterministic-mode` | 训练侧确定性模式。 | 与 true-on-policy 相关参数和环境变量联合使用。 | 单开该参数不足以保证训推完全对齐。 | [来源](/Users/qitongli/Developer/slime/examples/true_on_policy/run_simple.py:108) |
| `--eval-config` | 用 YAML 统一管理多任务评估配置。 | 与 `eval.defaults` + `eval.datasets` 配套。 | 单个数据集字段缺失会直接影响评估可观测性。 | [来源](/Users/qitongli/Developer/slime/examples/eval_multi_task/multi_task.sh:59) [来源](/Users/qitongli/Developer/slime/examples/eval_multi_task/multi_task.yaml:1) |
| `--get-mismatch-metrics` | 仅采集 mismatch 指标，不修改 loss。 | 常与 rollout logprobs 同时启用。 | 容易误判“已修复 mismatch”，实际只是在观测。 | [来源](/Users/qitongli/Developer/slime/examples/train_infer_mismatch_helper/README.md:22) |
| `--opd-kl-coef` | OPD KL 惩罚系数。 | 仅在 `--use-opd` 打开时生效。 | 系数过高会压制任务奖励学习。 | [来源](/Users/qitongli/Developer/slime/examples/on_policy_distillation/README.md:18) |
| `--opd-teacher-ckpt-step` | 指定 teacher checkpoint step。 | 主要用于 megatron teacher 路径。 | teacher 步数/版本漂移会造成蒸馏目标不稳定。 | [来源](/Users/qitongli/Developer/slime/examples/on_policy_distillation/README.md:20) |
| `--opd-teacher-load` | megatron teacher 的加载路径。 | `--opd-type=megatron` 必需；`sglang` 模式禁止设置。 | 路径模式冲突会直接报错。 | [来源](/Users/qitongli/Developer/slime/examples/on_policy_distillation/README.md:19) |
| `--opd-type` | 选择 OPD teacher 路径（`sglang`/`megatron`）。 | 与 `--opd-teacher-load`、`--rm-url` 互斥/互补。 | 教师模式选错会引起资源或架构不兼容。 | [来源](/Users/qitongli/Developer/slime/examples/on_policy_distillation/README.md:17) [来源](/Users/qitongli/Developer/slime/examples/on_policy_distillation/README.md:24) |
| `--rm-url` | 外部 teacher/reward 服务地址。 | OPD-sglang 常与 health-check 配套。 | server 未就绪会导致训练启动失败。 | [来源](/Users/qitongli/Developer/slime/examples/on_policy_distillation/run-qwen3-8B-opd.sh:76) [来源](/Users/qitongli/Developer/slime/examples/on_policy_distillation/run-qwen3-8B-opd.sh:26) |
| `--rollout-function-path` | 替换整个 rollout 驱动。 | fully-async 需配 `train_async.py`。 | 入口切错会退回同步行为或直接报错。 | [来源](/Users/qitongli/Developer/slime/examples/fully_async/README.md:37) [来源](/Users/qitongli/Developer/slime/examples/fully_async/run-qwen3-4b-fully_async.sh:41) |
| `--sglang-enable-deterministic-inference` | 推理侧确定性模式。 | 与 `--true-on-policy-mode`、`--deterministic-mode` 联动。 | 缺失该参数时 `train_rollout_logprob_abs_diff` 往往不为 0。 | [来源](/Users/qitongli/Developer/slime/examples/true_on_policy/run_simple.py:104) [来源](/Users/qitongli/Developer/slime/examples/true_on_policy/README.md:31) |
| `--sglang-rl-on-policy-target` | 指定 on-policy 对齐目标训练后端。 | 常设为 `fsdp`，需与训练后端一致。 | 目标不一致会导致对齐链路无效。 | [来源](/Users/qitongli/Developer/slime/examples/true_on_policy/run_simple.py:105) |
| `--true-on-policy-mode` | 开启严格训推 logprob 一致性路径。 | 需要 deterministic 组合与环境变量配套。 | 单独开启常无法达到 bitwise 级对齐。 | [来源](/Users/qitongli/Developer/slime/examples/true_on_policy/run_simple.py:109) |
| `--use-opd` | 开启 on-policy distillation。 | 必须配 `--opd-type`。 | 参数不完整会触发显式报错。 | [来源](/Users/qitongli/Developer/slime/examples/on_policy_distillation/README.md:16) [来源](/Users/qitongli/Developer/slime/examples/on_policy_distillation/README.md:117) |
| `--use-rollout-correction` | 开启 rollout correction（IS/RS）。 | 与 `--use-tis`、MIS 配置联动。 | 权重裁剪/拒绝策略过激会影响样本利用率。 | [来源](/Users/qitongli/Developer/slime/examples/train_infer_mismatch_helper/README.md:99) [来源](/Users/qitongli/Developer/slime/examples/train_infer_mismatch_helper/README.md:109) |
| `--use-rollout-logprobs` | 训练侧直接使用 rollout logprobs。 | 常与 mismatch 监控或 correction 联动。 | 若 token/logprob 对不齐，会直接污染 loss。 | [来源](/Users/qitongli/Developer/slime/examples/train_infer_mismatch_helper/README.md:20) |

## 3. 参数进阶专题与配置技巧

### 3.1 公式与联动

批量关系公式不是建议而是约束；一旦不守恒，会造成训练消费与采样供给失配，常见表现是步数/数据量认知错误。建议每次改 `rollout_batch_size`、`n_samples_per_prompt`、`num_steps_per_rollout` 时同步重算。 [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/quick_start.md:170)

### 3.2 OOM 调优顺序（可执行）

推荐顺序：
1. 先降 `--max-tokens-per-gpu`（仅在动态 batch 打开时生效）。
2. 再看是否需要增大 `--context-parallel-size` 承接长序列。
3. 训推一体时调低 `--sglang-mem-fraction-static` 给 Megatron 留空间。
4. 若是 server 侧拥塞/卡死，调 `--sglang-server-concurrency` 与 `--sglang-cuda-graph-bs`。
[来源](/Users/qitongli/Developer/slime/docs/zh/get_started/qa.md:24) [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/qa.md:26) [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:153) [来源](/Users/qitongli/Developer/slime/docs/zh/examples/qwen3-4B.md:281)

### 3.3 Dynamic Sampling + Partial Rollout

dynamic sampling 用 `--over-sampling-batch-size` + `--dynamic-sampling-filter-path` 做先过采样再筛选；`over_sampling_batch_size` 必须大于 `rollout_batch_size`。 [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/quick_start.md:342) [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/quick_start.md:350)

开启 `--partial-rollout` 后，可回收被 abort 的半成品样本，降低浪费；但要注意 `sample.metadata` 中保存了首次 rollout id，二次过滤逻辑必须兼容这点。 [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/quick_start.md:372) [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/quick_start.md:387)

### 3.4 训推一体 vs 训推分离 vs 异步训练

- 训推一体：省卡，但显存资源互相压制，必须精细调 mem-fraction；
- 训推分离：拓扑更稳，适合明确容量规划；
- 异步训练：提升资源利用率，但日志会混杂，需降低/分层日志级别。
[来源](/Users/qitongli/Developer/slime/docs/zh/get_started/quick_start.md:321) [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/quick_start.md:308) [来源](/Users/qitongli/Developer/slime/docs/zh/examples/qwen3-4B.md:295)

### 3.5 FP8/INT4 与 CPU Adam/deepep/checkpoint 约束

FP8 推理与训练路径要求 `config.json` 内量化配置正确；`--fp8-param-gather` 当前与 CPU Adam 冲突，是明确已知限制。 [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/low-precision.md:19) [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/low-precision.md:65)

INT4 QAT 依赖环境变量 `OPEN_TRAINING_INT4_FAKE_QAT_FLAG` 与 `OPEN_TRAINING_INT4_GROUP_SIZE`；多机需额外做好 Ray 侧编排。 [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/low-precision.md:87) [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/low-precision.md:120)

大型 MoE（如 Qwen3-30B/GLM4.5/DeepSeek-R1）文档普遍采用 CPU Adam + deepep + 大 EP 配置来平衡显存与吞吐。 [来源](/Users/qitongli/Developer/slime/docs/zh/examples/qwen3-30B-A3B.md:33) [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4.5-355B-A32B.md:147) [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:170)

### 3.6 examples 驱动的调参顺序

1. 真正 on-policy 对齐：先锁定确定性参数与环境变量组合（`--true-on-policy-mode`、`--sglang-enable-deterministic-inference`、`--deterministic-mode` + `NCCL_ALGO`/`NVTE_ALLOW_NONDETERMINISTIC_ALGO`/`CUBLAS_WORKSPACE_CONFIG`），再看 `train/train_rollout_logprob_abs_diff` 是否归零。 [来源](/Users/qitongli/Developer/slime/examples/true_on_policy/run_simple.py:103) [来源](/Users/qitongli/Developer/slime/examples/true_on_policy/run_simple.py:114) [来源](/Users/qitongli/Developer/slime/examples/true_on_policy/README.md:31)
2. OOM 顺序细化：先降 `--max-tokens-per-gpu`，再加 `--context-parallel-size`，再调 `--sglang-mem-fraction-static`，最后再控 `--sglang-server-concurrency`/`--sglang-cuda-graph-bs`。 [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/qa.md:24) [来源](/Users/qitongli/Developer/slime/examples/train_infer_mismatch_helper/run-qwen3-4b-mis.sh:67) [来源](/Users/qitongli/Developer/slime/examples/strands_sglang/strands_qwen3_8b.sh:112)
3. 多轮/tool 场景先保 token/logprob 对齐：`return_logprob=True` 时禁用文本后处理，避免重分词造成 token-logprob 错位。 [来源](/Users/qitongli/Developer/slime/examples/search-r1/generate_with_search.py:93) [来源](/Users/qitongli/Developer/slime/examples/search-r1/generate_with_search.py:179) [来源](/Users/qitongli/Developer/slime/examples/search-r1/generate_with_search.py:224)
4. OPD 模式选择：teacher 架构异构或显存放不下，走 `sglang` 外部 teacher；与 policy/ref 同构时，走 `megatron` teacher 更直接。 [来源](/Users/qitongli/Developer/slime/examples/on_policy_distillation/README.md:24) [来源](/Users/qitongli/Developer/slime/examples/on_policy_distillation/README.md:27)

## 4. 性能监控、追踪与排查闭环（重点）

### 4.1 监控对象（先看什么）

建议最小监控集合：
- 推理吞吐上限指标：`perf/longest_sample_tokens_per_sec`；
- 训练稳定性：KL、grad_norm；
- 资源风险：OOM 迹象、KV cache 压力、router/worker 健康。
[来源](/Users/qitongli/Developer/slime/docs/zh/blogs/release_v0.1.0.md:51) [来源](/Users/qitongli/Developer/slime/docs/zh/developer_guide/debug.md:19) [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/fault-tolerance.md:9)

### 4.2 Profiling 标准流程

标准链路：
1. 用 `--rollout-function-path slime.rollout.sleep_rollout.sleep` 把 rollout 挂起。
2. 查 router `/workers` 获取活跃引擎。
3. `tools/profile_rollout.py --action start` 启动采集。
4. 负载压测。
5. `--action stop` 或自动停止后，拿 trace 到 Perfetto/Chrome 分析。
[来源](/Users/qitongli/Developer/slime/docs/zh/developer_guide/profiling.md:13) [来源](/Users/qitongli/Developer/slime/docs/zh/developer_guide/profiling.md:21) [来源](/Users/qitongli/Developer/slime/docs/zh/developer_guide/profiling.md:42) [来源](/Users/qitongli/Developer/slime/docs/zh/developer_guide/profiling.md:66)

### 4.3 Debug 标准流程

先做精度对齐：第一步看 rollout 可读性、`log_probs` 与 `ref_log_probs` 是否一致、`num_steps_per_rollout==1` 时 KL 是否为 0；不一致再按 kernel/non-determinism、参数映射、特殊 buffer 释放等方向拆。 [来源](/Users/qitongli/Developer/slime/docs/zh/developer_guide/debug.md:7) [来源](/Users/qitongli/Developer/slime/docs/zh/developer_guide/debug.md:13) [来源](/Users/qitongli/Developer/slime/docs/zh/developer_guide/debug.md:19)

分而治之：用 `--debug-rollout-only` / `--debug-train-only`，配合 `--save-debug-rollout-data` 与 `--load-debug-rollout-data` 固定训练输入，快速定位是推理链路还是训练链路问题。 [来源](/Users/qitongli/Developer/slime/docs/zh/developer_guide/debug.md:35) [来源](/Users/qitongli/Developer/slime/docs/zh/developer_guide/debug.md:47)

IMA（illegal memory access）排查优先序：`CUDA_LAUNCH_BLOCKING=1` -> 关闭 speculative/cuda graph -> 关闭 deepep -> CUDA core dump 锁定 kernel。 [来源](/Users/qitongli/Developer/slime/docs/zh/developer_guide/debug.md:55) [来源](/Users/qitongli/Developer/slime/docs/zh/developer_guide/debug.md:57) [来源](/Users/qitongli/Developer/slime/docs/zh/developer_guide/debug.md:61)

### 4.4 容灾与可复现

容灾：`--use-fault-tolerance` + 三个 health-check 参数（first-wait/interval/timeout），在 rollout 中做心跳与失效隔离、轮后重启更新。 [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/fault-tolerance.md:5) [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/fault-tolerance.md:11)

可复现：SGLang deterministic + Megatron deterministic + 指定 NCCL/NVTE/CUBLAS 环境变量；并要求去掉 flash_attn_3。 [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/reproducibility.md:5) [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/reproducibility.md:15)

### 4.5 一条完整“现象->指标->命令->定位->修复”示例

现象：训练中途 OOM 或推理长时间无输出。 [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/qa.md:22) [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/qa.md:54)

指标：检查 `max_tokens_per_gpu` 与实际响应长度、是否触发 stop token、server 并发是否超默认图容量。 [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/qa.md:24) [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/qa.md:56) [来源](/Users/qitongli/Developer/slime/docs/zh/examples/qwen3-4B.md:281)

命令：
- 先 `tools/profile_rollout.py` 采样；
- 必要时切 `--debug-rollout-only` 复现推理瓶颈；
- 再切 `--debug-train-only` + `--load-debug-rollout-data` 复现训练瓶颈。  
[来源](/Users/qitongli/Developer/slime/docs/zh/developer_guide/profiling.md:42) [来源](/Users/qitongli/Developer/slime/docs/zh/developer_guide/debug.md:35) [来源](/Users/qitongli/Developer/slime/docs/zh/developer_guide/debug.md:49)

定位与修复：
- OOM：先降 `--max-tokens-per-gpu`，再开/调 CP，再降 `--sglang-mem-fraction-static`；
- 卡死：设置 stop token（`--rollout-stop` / `--rollout-stop-token-ids`），必要时限流并发；
- IMA：按 debug 指南逐级关特性与 core dump。  
[来源](/Users/qitongli/Developer/slime/docs/zh/get_started/qa.md:24) [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/qa.md:56) [来源](/Users/qitongli/Developer/slime/docs/zh/developer_guide/debug.md:53)

### 4.6 examples 监控与排障闭环补充

- 现象：`train/train_rollout_logprob_abs_diff` 持续非 0。  
  指标：`train/train_rollout_logprob_abs_diff`。  
  命令：开启 true-on-policy 参数组并固定确定性环境变量。  
  定位：训推 kernel/数值路径没对齐。  
  修复：`--true-on-policy-mode` + `--sglang-enable-deterministic-inference` + `--deterministic-mode` + 对应环境变量组合。  
  [来源](/Users/qitongli/Developer/slime/examples/true_on_policy/README.md:31) [来源](/Users/qitongli/Developer/slime/examples/true_on_policy/run_simple.py:103) [来源](/Users/qitongli/Developer/slime/examples/true_on_policy/run_simple.py:114)

- 现象：异步训练吞吐掉速或阶段性“卡住”。  
  指标：worker queue size、`No progress` 告警。  
  命令：检查异步 worker 日志与输出队列状态。  
  定位：在飞任务并发/队列回压配置不匹配。  
  修复：回调 `rollout_batch_size` 与并发参数，减少积压。  
  [来源](/Users/qitongli/Developer/slime/examples/fully_async/fully_async_rollout.py:57) [来源](/Users/qitongli/Developer/slime/examples/fully_async/fully_async_rollout.py:145) [来源](/Users/qitongli/Developer/slime/examples/fully_async/fully_async_rollout.py:227)

- 现象：OPD 训练刚启动就失败。  
  指标：teacher `/health_generate` 不可用。  
  命令：`curl -sf http://$TEACHER_IP:$TEACHER_PORT/health_generate`，并检查 `get_model_info`。  
  定位：teacher server 未就绪或地址配置错误。  
  修复：先等待 teacher 通过 health-check，再提交 Ray 训练任务。  
  [来源](/Users/qitongli/Developer/slime/examples/on_policy_distillation/run-qwen3-8B-opd.sh:26) [来源](/Users/qitongli/Developer/slime/examples/on_policy_distillation/run-qwen3-8B-opd.sh:32)

- 现象：agentic 工具调用不稳定、失败率高。  
  指标：tool iteration/tool call count、外部 API 并发限流错误。  
  命令：降低并发，限制工具回合与迭代上限。  
  定位：工具执行链路过长或并发过高。  
  修复：设置迭代上限（如 `ToolIterationLimiter`）并收缩并发参数。  
  [来源](/Users/qitongli/Developer/slime/examples/strands_sglang/generate_with_strands.py:62) [来源](/Users/qitongli/Developer/slime/examples/strands_sglang/generate_with_strands.py:91) [来源](/Users/qitongli/Developer/slime/examples/tau-bench/run_qwen3_4B.sh:102)

## 5. 典型训练场景模板（Dense/MoE/SFT/多机）

### 5.1 单机 Dense 基线模板

Qwen3-4B / GLM4-9B 是最完整的“标准模板”：MODEL/CKPT/ROLLOUT/EVAL/PERF/GRPO/OPT/SGLANG 分组清晰，适合做参数变更基线。 [来源](/Users/qitongli/Developer/slime/docs/zh/examples/qwen3-4B.md:50) [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:50)

### 5.2 中大规模 MoE 模板（Qwen3-30B/GLM4.5/DeepSeek-R1）

三者共性：大 EP、CPU Adam、deepep、动态 batch、多机 Ray；差异在并行切分（tp/pp/cp/ep）、SGLang EP/DP 配置与并发设置。 [来源](/Users/qitongli/Developer/slime/docs/zh/examples/qwen3-30B-A3B.md:44) [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4.5-355B-A32B.md:74) [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:111)

### 5.3 SFT 模式模板

SFT 复用 rollout 机制，但切换到 `slime.rollout.sft_rollout.generate_rollout`，并配 `--loss-type sft_loss`、`--calculate-per-token-loss`、`--disable-compute-advantages-and-returns`、`--debug-train-only`。 [来源](/Users/qitongli/Developer/slime/docs/zh/examples/qwen3-4b-base-openhermes.md:60) [来源](/Users/qitongli/Developer/slime/docs/zh/examples/qwen3-4b-base-openhermes.md:68)

### 5.4 Qwen3-Next 特殊依赖与编译补丁

Qwen3-next-80B-A3B 文档明确给出额外依赖（flash-linear-attention、causal-conv1d）与 Triton/Blackwell 编译补丁路径；这类依赖不应混入通用模板，而应单独做环境层前置检查。 [来源](/Users/qitongli/Developer/slime/docs/zh/examples/qwen3-next-80B-A3B.md:20) [来源](/Users/qitongli/Developer/slime/docs/zh/examples/qwen3-next-80B-A3B.md:31)

### 5.5 Agentic 多轮模板（Search-R1 / Tau / Retool / Strands / Multi-Agent）

统一模式是“custom generate 作为环境驱动器 + custom rm 作为任务奖励器”，并按场景选择是否保留 rollout logprobs。 [来源](/Users/qitongli/Developer/slime/examples/search-r1/README_zh.md:168) [来源](/Users/qitongli/Developer/slime/examples/retool/retool_qwen3_4b_rl.sh:123) [来源](/Users/qitongli/Developer/slime/examples/strands_sglang/strands_qwen3_8b.sh:129)

Search-R1/Tau 偏“外部检索或用户模拟”环境，Retool/Strands 偏“工具执行轨迹内生到样本”；Multi-Agent 则是 solver/rewriter/selector 多角色并行，当前示例不支持 eval。 [来源](/Users/qitongli/Developer/slime/examples/search-r1/generate_with_search.py:160) [来源](/Users/qitongli/Developer/slime/examples/tau-bench/generate_with_tau.py:127) [来源](/Users/qitongli/Developer/slime/examples/multi_agent/agent_system.py:187) [来源](/Users/qitongli/Developer/slime/examples/multi_agent/run-qwen3-30B-A3B-multi-agent.sh:57)

### 5.6 True On-Policy 与 OPD 专项模板

True-on-policy 模板重点是“严格对齐训推 logprob 并接受适度性能损失”；验证指标是 `train/train_rollout_logprob_abs_diff == 0`。 [来源](/Users/qitongli/Developer/slime/examples/true_on_policy/README.md:31) [来源](/Users/qitongli/Developer/slime/examples/true_on_policy/README.md:40)

OPD 模板重点是“在 on-policy 数据上叠加 teacher KL 约束”，可选外部 sglang teacher 或内部 megatron teacher，两者选型按 teacher 规模与架构兼容性决定。 [来源](/Users/qitongli/Developer/slime/examples/on_policy_distillation/README.md:7) [来源](/Users/qitongli/Developer/slime/examples/on_policy_distillation/README.md:24)

## 6. 平台与硬件特例（AMD）

AMD 文档是英文特有内容，当前限定 MI300/MI325，给出了 ROCm 镜像与运行命令模板。 [来源](/Users/qitongli/Developer/slime/docs/en/platform_support/amd_tutorial.md:8) [来源](/Users/qitongli/Developer/slime/docs/en/platform_support/amd_tutorial.md:34)

ROCm 差异点：
- 需要 `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1` 与 `HIP_VISIBLE_DEVICES`。
- 当前需 `--no-gradient-accumulation-fusion`（apex/ROCm 限制）。
- 提供 CPU-only/Gloo 的转换路径绕过硬件问题。  
[来源](/Users/qitongli/Developer/slime/docs/en/platform_support/amd_tutorial.md:113) [来源](/Users/qitongli/Developer/slime/docs/en/platform_support/amd_tutorial.md:111) [来源](/Users/qitongli/Developer/slime/docs/en/platform_support/amd_tutorial.md:94)

与 NVIDIA 路径的共性是训练主参数分组几乎一致；差异主要在 runtime 环境变量和部分 fused 功能可用性。 [来源](/Users/qitongli/Developer/slime/docs/en/platform_support/amd_tutorial.md:156) [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/quick_start.md:107)

## 7. 已知风险与高频坑点

- 乱码/精度异常：多由 ckpt 加载、参数映射、并行层号转换问题触发；先做首步 logprob 对齐检查。 [来源](/Users/qitongli/Developer/slime/docs/zh/developer_guide/debug.md:9)
- 任务卡在 Ray 提交页：大概率是资源预算与 colocate/分离模式不一致。 [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/qa.md:9)
- OOM：`max_tokens_per_gpu` 过高、CP 未启用或自定义多轮生成长度失控。 [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/qa.md:24)
- sglang 连接/端口问题：多 server 端口冲突，临时缓解是减少单机 server 数（例如提高 tp）。 [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/qa.md:46)
- torch inductor `JSONDecodeError`：cache 竞争，需在 env_var 禁用相关 cache。 [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/qa.md:62)
- fp8 参数存储与 CPU Adam 冲突、训练保存 ckpt 仍是原精度形态。 [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/low-precision.md:50) [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/low-precision.md:65)
- Qwen3Next 快速适配路径存在 TP 限制：被替换模块自身暂不支持 TP。 [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/arch-support-beyond-megatron.md:32)
- 非二值 reward（如 0/0.9）在 fp32 下可能引入伪梯度，导致“同组同奖但 advantage 不为 0”的训练噪声。 [来源](/Users/qitongli/Developer/slime/examples/geo3k_vlm/README.md:104)
- 当 `return_logprob=True` 时做响应后处理，会破坏 token-logprob 对齐并污染损失计算。 [来源](/Users/qitongli/Developer/slime/examples/search-r1/generate_with_search.py:93)
- fully-async 示例目前不支持 eval，且错误处理是最小实现；multi-agent 示例当前也未支持 eval。 [来源](/Users/qitongli/Developer/slime/examples/fully_async/README.md:29) [来源](/Users/qitongli/Developer/slime/examples/multi_agent/run-qwen3-30B-A3B-multi-agent.sh:57)
- 示例中的激进清理命令（如 `pkill -9 python`）适合隔离容器，不适合共享机器或多任务环境。 [来源](/Users/qitongli/Developer/slime/examples/fully_async/run-qwen3-4b-fully_async.sh:8) [来源](/Users/qitongli/Developer/slime/examples/retool/retool_qwen3_4b_rl.sh:8)

## 8. 补充建议（非原文）

补充（非原文）: 建议把参数管理拆成三层配置（`base.yaml` / `hardware.yaml` / `experiment.yaml`），最终渲染成一份可追溯 CLI；这样可以明显降低“脚本复制改名”带来的参数漂移。

补充（非原文）: 建议固定一个“首步守门 CI”：启动后只跑 1 rollout + 1 train step，强制校验 KL、logprob 对齐、OOM 安全阈值，再放开长跑。

补充（非原文）: 对 `--sglang-*` 与原生 SGLang 参数建立映射清单并做单测（尤其 `tp/ep/dp/context/mem-fraction`），能避免跨版本升级时的 silent regression。

## 9. 来自 examples 的实战经验补充（排除 holdem_game）

### 模式 A：同步/异步训练驱动差异

同步模板通常走 `train.py`，而 fully-async 通过 `train_async.py` + 持久后台 worker 实现“训练消费与 rollout 生产并行”。 [来源](/Users/qitongli/Developer/slime/examples/fully_async/README.md:37) [来源](/Users/qitongli/Developer/slime/examples/fully_async/fully_async_rollout.py:52) [来源](/Users/qitongli/Developer/slime/examples/on_policy_distillation/run-qwen3-8B-opd.sh:160)

### 模式 B：多轮环境接入（custom generate + custom config + env API）

多轮 VLM/agentic 路径本质是：`--custom-generate-function-path` 接管轨迹生成，`--custom-config-path` 注入环境参数，再由 env API（build/reset/step/format）维护状态机。 [来源](/Users/qitongli/Developer/slime/examples/geo3k_vlm_multi_turn/run_geo3k_vlm_multi_turn.py:63) [来源](/Users/qitongli/Developer/slime/examples/geo3k_vlm_multi_turn/README.md:15)

### 模式 C：训练-推理一致性路径（true-on-policy vs MIS/TIS vs OPD）

三条路径各有定位：true-on-policy 追求严格训推一致；MIS/TIS 修正 mismatch 并给出监控指标；OPD 用 teacher KL 注入额外学习信号。 [来源](/Users/qitongli/Developer/slime/examples/true_on_policy/README.md:1) [来源](/Users/qitongli/Developer/slime/examples/train_infer_mismatch_helper/README.md:173) [来源](/Users/qitongli/Developer/slime/examples/on_policy_distillation/README.md:7)

### 模式 D：评估配置化（multi-task eval config）

多任务评估更稳妥的工程实践是把数据集与默认采样策略外置到 `eval-config`，避免训练脚本内硬编码多个 `--eval-*` 组合。 [来源](/Users/qitongli/Developer/slime/examples/eval_multi_task/README.md:4) [来源](/Users/qitongli/Developer/slime/examples/eval_multi_task/multi_task.sh:59)

## 附录A：全量参数索引

说明：本附录按“文档字面出现的 `--xxx` 参数”全量收录（含少量宿主命令参数、前缀说明项与文中占位符如 `--eval-*` / `--xxx`），用于参数差集校验与检索。

| 参数名 | 作用 | 关键联动/约束 | 风险/坑点 | 主要出处 |
| --- | --- | --- | --- | --- |
| `--accumulate-allreduce-grads-in-fp32` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/qwen3-next-80B-A3B.md:88) |
| `--action` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/developer_guide/profiling.md:42) |
| `--activities` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/developer_guide/profiling.md:49) |
| `--actor-num-gpus-per-node` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:214) |
| `--actor-num-nodes` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:213) |
| `--adam-beta1` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:189) |
| `--adam-beta2` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:190) |
| `--address` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:63) |
| `--advantage-estimator` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:144) |
| `--apply-chat-template` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:96) |
| `--attention-backend` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/developer_guide/debug.md:15) |
| `--attention-dropout` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/en/platform_support/amd_tutorial.md:239) |
| `--attention-softmax-in-fp32` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/en/platform_support/amd_tutorial.md:243) |
| `--balance-abs-threshold` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:154) |
| `--balance-data` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:118) |
| `--block-size` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/low-precision.md:15) |
| `--buffer-filter-path` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:268) |
| `--calculate-per-token-loss` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/qwen3-4b-base-openhermes.md:69) |
| `--cap-add` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/en/platform_support/amd_tutorial.md:39) |
| `--ckpt-format` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:110) |
| `--ckpt-step` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/qa.md:7) |
| `--colocate` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/blogs/introducing_slime.md:39) |
| `--context-length` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:347) |
| `--context-parallel-size` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:124) |
| `--critic-load` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:255) |
| `--critic-lr` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:257) |
| `--critic-lr-warmup-iters` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:258) |
| `--critic-num-gpus-per-node` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:233) |
| `--critic-num-nodes` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:233) |
| `--critic-save` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:256) |
| `--cuda-graph-bs` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/qwen3-4B.md:289) |
| `--custom-convert-samples-to-train-data-path` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/customization.md:23) |
| `--custom-eval-rollout-log-function-path` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/customization.md:25) |
| `--custom-generate-function-path` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/customization.md:12) |
| `--custom-loss-function-path` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/customization.md:19) |
| `--custom-megatron-before-log-prob-hook-path` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/customization.md:29) |
| `--custom-megatron-before-train-step-hook-path` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/customization.md:30) |
| `--custom-megatron-init-path` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/customization.md:28) |
| `--custom-pg-loss-reducer-function-path` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/customization.md:21) |
| `--custom-reward-post-process-path` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/customization.md:22) |
| `--custom-rm-path` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:101) |
| `--custom-rollout-log-function-path` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/customization.md:24) |
| `--custom-tis-function-path` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/customization.md:20) |
| `--dashboard-host` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/en/platform_support/amd_tutorial.md:256) |
| `--dashboard-port` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/en/platform_support/amd_tutorial.md:256) |
| `--data-dir` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/low-precision.md:81) |
| `--data-source-path` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/customization.md:26) |
| `--debug-rollout-only` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/blogs/introducing_slime.md:55) |
| `--debug-train-only` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/blogs/introducing_slime.md:63) |
| `--decoder-first-pipeline-num-layers` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:43) |
| `--decoder-last-pipeline-num-layers` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:44) |
| `--deterministic-mode` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/reproducibility.md:12) |
| `--device` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/en/platform_support/amd_tutorial.md:35) |
| `--disable-bias-linear` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:66) |
| `--disable-compute-advantages-and-returns` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/qwen3-4b-base-openhermes.md:70) |
| `--disable-usage-stats` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:63) |
| `--dp-size` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:348) |
| `--dynamic-sampling-filter-path` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:242) |
| `--enable-dp-attention` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:348) |
| `--enable-mtp-training` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/speculative-decoding.md:32) |
| `--enable-return-routed-experts` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/slime-router.md:57) |
| `--entropy-coef` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:148) |
| `--ep-size` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:348) |
| `--eps-clip` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:149) |
| `--eps-clip-high` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:150) |
| `--eval-function-path` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/customization.md:27) |
| `--eval-interval` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:128) |
| `--eval-max-response-len` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:131) |
| `--eval-prompt-data` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:129) |
| `--eval-top-p` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:132) |
| `--expert-model-parallel-size` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:42) |
| `--expert-tensor-parallel-size` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:41) |
| `--ffn-hidden-size` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:63) |
| `--force` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:75) |
| `--fp8-format` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/low-precision.md:30) |
| `--fp8-param-gather` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/low-precision.md:32) |
| `--fp8-recipe` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/low-precision.md:31) |
| `--git` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/qwen3-next-80B-A3B.md:40) |
| `--global-batch-size` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/qwen3-4b-base-openhermes.md:66) |
| `--gpus` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/quick_start.md:34) |
| `--grad-reduce-in-bf16` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/qwen3-next-80B-A3B.md:88) |
| `--group-add` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/en/platform_support/amd_tutorial.md:38) |
| `--group-query-attention` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:69) |
| `--group-rm` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/customization.md:86) |
| `--head` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:66) |
| `--hf-checkpoint` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/low-precision.md:9) |
| `--hidden-dropout` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/en/platform_support/amd_tutorial.md:240) |
| `--hidden-size` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:62) |
| `--input-dir` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/low-precision.md:79) |
| `--input-fp8-hf-path` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:25) |
| `--input-key` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:92) |
| `--ipc` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/quick_start.md:34) |
| `--kl-coef` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:154) |
| `--kl-loss-coef` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:146) |
| `--kl-loss-type` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:147) |
| `--kv-channels` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:71) |
| `--label-key` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:93) |
| `--load` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/developer_guide/debug.md:47) |
| `--load-debug-rollout-data` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/developer_guide/debug.md:47) |
| `--local-dir` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/reproducibility.md:32) |
| `--log-level` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/quick_start.md:293) |
| `--loss-type` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/qwen3-4b-base-openhermes.md:68) |
| `--lr` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:186) |
| `--lr-decay-style` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:187) |
| `--master-addr` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:35) |
| `--master-port` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:35) |
| `--max-positional-embedding` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:87) |
| `--max-tokens-per-gpu` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:109) |
| `--max-workers` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/low-precision.md:16) |
| `--mem-fraction` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/blogs/release_v0.1.0.md:64) |
| `--mem-fraction-static` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:346) |
| `--metadata-key` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/quick_start.md:467) |
| `--micro-batch-size` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:159) |
| `--model-dir` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/low-precision.md:13) |
| `--model-name` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/qa.md:30) |
| `--model-path` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:353) |
| `--moe-a2a-backend` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:348) |
| `--moe-enable-deepep` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:206) |
| `--moe-permute-fusion` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/developer_guide/debug.md:21) |
| `--moe-token-dispatcher-type` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:207) |
| `--mtp-loss-scaling-factor` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/speculative-decoding.md:33) |
| `--mtp-num-layers` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/speculative-decoding.md:31) |
| `--n-samples-per-eval-prompt` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:130) |
| `--n-samples-per-prompt` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:110) |
| `--name` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/en/platform_support/amd_tutorial.md:45) |
| `--nnodes` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:36) |
| `--no-check-for-nan-in-loss-and-grad` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/qa.md:68) |
| `--no-deps` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:11) |
| `--no-gradient-accumulation-fusion` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/en/platform_support/amd_tutorial.md:89) |
| `--node-ip-address` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:63) |
| `--node-rank` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:36) |
| `--norm-epsilon` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:75) |
| `--normalization` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:74) |
| `--normalize-advantages` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:217) |
| `--nproc-per-node` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:34) |
| `--num-attention-heads` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:68) |
| `--num-critic-only-steps` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:259) |
| `--num-epoch` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/qwen3-4b-base-openhermes.md:64) |
| `--num-gpus` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:63) |
| `--num-layers` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:61) |
| `--num-query-groups` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:70) |
| `--num-rollout` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:105) |
| `--num-steps` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/developer_guide/profiling.md:42) |
| `--num-steps-per-rollout` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:116) |
| `--optimizer` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:164) |
| `--optimizer-cpu-offload` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:164) |
| `--origin-hf-dir` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/quick_start.md:102) |
| `--output-bf16-hf-path` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:25) |
| `--output-dir` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/low-precision.md:80) |
| `--over-sampling-batch-size` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:241) |
| `--overlap-cpu-optimizer-d2h-h2d` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:165) |
| `--partial-rollout` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:266) |
| `--pipeline-model-parallel-size` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:40) |
| `--prefill-num-servers` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/pd-disaggregation.md:5) |
| `--privileged` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/en/platform_support/amd_tutorial.md:41) |
| `--profile-by-stage` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/developer_guide/profiling.md:50) |
| `--prompt-data` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:91) |
| `--qk-layernorm` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:72) |
| `--recompute-granularity` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:129) |
| `--recompute-method` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:130) |
| `--recompute-num-layers` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:131) |
| `--ref-load` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:97) |
| `--repo-type` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/reproducibility.md:32) |
| `--rm` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:102) |
| `--rm-type` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:102) |
| `--rm-url` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/customization.md:103) |
| `--rollout-all-samples-process-path` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/customization.md:17) |
| `--rollout-batch-size` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:107) |
| `--rollout-data-postprocess-path` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/customization.md:18) |
| `--rollout-function-path` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/developer_guide/profiling.md:13) |
| `--rollout-health-check-first-wait` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/fault-tolerance.md:11) |
| `--rollout-health-check-interval` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/fault-tolerance.md:12) |
| `--rollout-health-check-timeout` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/fault-tolerance.md:13) |
| `--rollout-max-response-len` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:112) |
| `--rollout-num-gpus` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:172) |
| `--rollout-num-gpus-per-engine` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:172) |
| `--rollout-sample-filter-path` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/customization.md:16) |
| `--rollout-shuffle` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:98) |
| `--rollout-stop` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/qa.md:56) |
| `--rollout-stop-token-ids` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/qa.md:56) |
| `--rollout-temperature` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:113) |
| `--rotary-base` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:63) |
| `--router-balance-abs-threshold` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:154) |
| `--router-url` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/developer_guide/profiling.md:42) |
| `--runtime-env-json` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/quick_start.md:564) |
| `--save` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/low-precision.md:14) |
| `--save-debug-rollout-data` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/developer_guide/debug.md:43) |
| `--save-dir` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/low-precision.md:14) |
| `--save-interval` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:101) |
| `--security-opt` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/en/platform_support/amd_tutorial.md:40) |
| `--seq-length` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:87) |
| `--sequence-parallel` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:122) |
| `--sglang` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/reproducibility.md:8) |
| `--sglang-` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/reproducibility.md:8) |
| `--sglang-attention-backend` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/reproducibility.md:9) |
| `--sglang-context-length` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:152) |
| `--sglang-cuda-graph-bs` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4.5-355B-A32B.md:185) |
| `--sglang-deepep-mode` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:190) |
| `--sglang-dp-size` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:184) |
| `--sglang-enable-deepep-moe` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/blogs/introducing_slime.md:54) |
| `--sglang-enable-deterministic-inference` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/reproducibility.md:8) |
| `--sglang-enable-dp-attention` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/blogs/introducing_slime.md:54) |
| `--sglang-enable-dp-lm-head` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:186) |
| `--sglang-enable-ep-moe` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/blogs/introducing_slime.md:54) |
| `--sglang-ep-num-redundant-experts` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/qwen3-30B-A3B.md:104) |
| `--sglang-ep-size` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:180) |
| `--sglang-log-level` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/qwen3-4B.md:301) |
| `--sglang-mem-fraction-static` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:179) |
| `--sglang-moe-a2a-backend` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:189) |
| `--sglang-moe-dense-tp-size` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:185) |
| `--sglang-router-ip` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/slime-router.md:21) |
| `--sglang-router-policy` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/slime-router.md:123) |
| `--sglang-router-port` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:359) |
| `--sglang-server-concurrency` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:174) |
| `--sglang-speculative-algorithm` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/speculative-decoding.md:10) |
| `--sglang-speculative-draft-model-path` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/speculative-decoding.md:19) |
| `--sglang-speculative-eagle-topk` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/speculative-decoding.md:12) |
| `--sglang-speculative-num-draft-tokens` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/speculative-decoding.md:13) |
| `--sglang-speculative-num-steps` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/speculative-decoding.md:11) |
| `--sglang-tp-size` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:26) |
| `--shm-size` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/quick_start.md:34) |
| `--slime-router-middleware-paths` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/slime-router.md:15) |
| `--strategy` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/low-precision.md:15) |
| `--swiglu` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:64) |
| `--tensor-model-parallel-size` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:39) |
| `--tp-size` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:352) |
| `--train-backend` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:38) |
| `--true-on-policy-mode` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:196) |
| `--ulimit` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/quick_start.md:35) |
| `--use-dynamic-batch-size` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:109) |
| `--use-fault-tolerance` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/fault-tolerance.md:5) |
| `--use-kl-loss` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:145) |
| `--use-precision-aware-optimizer` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:166) |
| `--use-rollout-routing-replay` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/slime-router.md:66) |
| `--use-rotary-position-embeddings` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:77) |
| `--use-routing-replay` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/customization.md:422) |
| `--use-slime-router` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/advanced/slime-router.md:23) |
| `--use-tis` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/quick_start.md:274) |
| `--use-wandb` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/en/platform_support/amd_tutorial.md:226) |
| `--value-clip` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:261) |
| `--vocab-size` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/get_started/quick_start.md:105) |
| `--wandb-group` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/en/platform_support/amd_tutorial.md:228) |
| `--wandb-key` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/en/platform_support/amd_tutorial.md:229) |
| `--wandb-project` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/en/platform_support/amd_tutorial.md:227) |
| `--weight-decay` | 文档中出现的 CLI 参数或参数前缀标记。 | 详见正文第2章分层说明；涉及 SGLang 时注意前缀映射。 | 常见风险是同名参数跨组件语义不同，或被资源模式（colocate/分离）改变效果。 | [来源](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:188) |

### 附录A.1：examples 参数增量索引（关键训练参数）

说明：本小节只收录 examples 新增或更强调的训练/推理/评估核心参数；下载命令参数和系统管理参数（如 `--local-dir`、`--head`）不纳入此索引。

| 参数名 | 作用 | 关键联动/约束 | 风险/坑点 | 主要出处 |
| --- | --- | --- | --- | --- |
| `--custom-config-path` | 注入多轮/算法配置。 | 与 `--custom-generate-function-path` 或 `--custom-tis-function-path` 联动。 | 配置对象与代码约定不一致会直接失效。 | [来源](/Users/qitongli/Developer/slime/examples/geo3k_vlm_multi_turn/run_geo3k_vlm_multi_turn.py:64) |
| `--custom-reward-post-process-path` | 奖励后处理（如写入 teacher logprobs）。 | OPD-sglang 常与 `--rm-url` 配套。 | 对齐处理缺失会引入蒸馏噪声。 | [来源](/Users/qitongli/Developer/slime/examples/on_policy_distillation/run-qwen3-8B-opd.sh:75) |
| `--custom-tis-function-path` | 指定 MIS/TIS 权重函数。 | 与 `--use-tis`、`mis.yaml` 联动。 | 权重边界配置不当会高方差。 | [来源](/Users/qitongli/Developer/slime/examples/train_infer_mismatch_helper/run-qwen3-4b-mis.sh:125) |
| `--deterministic-mode` | 训练确定性模式。 | 需配合推理确定性参数。 | 单独开启无法保证训推一致。 | [来源](/Users/qitongli/Developer/slime/examples/true_on_policy/run_simple.py:108) |
| `--eval-config` | 配置化多任务评估。 | 与 `multi_task.yaml` 配套。 | 数据集字段缺失会导致评估缺项。 | [来源](/Users/qitongli/Developer/slime/examples/eval_multi_task/multi_task.sh:59) |
| `--get-mismatch-metrics` | 只监控 mismatch，不修正 loss。 | 与 `rollout_log_probs` 搭配收益最大。 | 容易误当成“已修复”。 | [来源](/Users/qitongli/Developer/slime/examples/train_infer_mismatch_helper/README.md:22) |
| `--opd-kl-coef` | OPD KL 系数。 | `--use-opd` 开启后生效。 | 过大压制任务学习。 | [来源](/Users/qitongli/Developer/slime/examples/on_policy_distillation/README.md:18) |
| `--opd-teacher-ckpt-step` | teacher step 选择。 | megatron teacher 场景常用。 | 版本漂移导致目标不稳。 | [来源](/Users/qitongli/Developer/slime/examples/on_policy_distillation/README.md:20) |
| `--opd-teacher-load` | teacher checkpoint 路径。 | 仅 `megatron` 模式可用。 | 模式冲突会报错。 | [来源](/Users/qitongli/Developer/slime/examples/on_policy_distillation/README.md:19) |
| `--opd-type` | OPD 模式选择。 | `sglang`/`megatron` 二选一。 | 模式选择与资源不匹配会失败。 | [来源](/Users/qitongli/Developer/slime/examples/on_policy_distillation/README.md:17) |
| `--rm-url` | 外部 teacher/reward 地址。 | 需要 health-check。 | 服务未就绪导致训练中断。 | [来源](/Users/qitongli/Developer/slime/examples/on_policy_distillation/run-qwen3-8B-opd.sh:76) |
| `--rollout-function-path` | 自定义 rollout 驱动入口。 | fully-async 与 `train_async.py` 联动。 | 路径错误会直接降级或报错。 | [来源](/Users/qitongli/Developer/slime/examples/fully_async/README.md:40) |
| `--sglang-enable-deterministic-inference` | 推理确定性模式。 | 与 true-on-policy 参数组联动。 | 缺失时 logprob diff 常不归零。 | [来源](/Users/qitongli/Developer/slime/examples/true_on_policy/run_simple.py:104) |
| `--sglang-rl-on-policy-target` | 指定 on-policy 对齐目标后端。 | 需与训练后端一致。 | 不一致会破坏一致性假设。 | [来源](/Users/qitongli/Developer/slime/examples/true_on_policy/run_simple.py:105) |
| `--true-on-policy-mode` | 开启严格训推一致性。 | 需 deterministic 参数和环境变量配套。 | 只开 flag 不足以达成目标。 | [来源](/Users/qitongli/Developer/slime/examples/true_on_policy/run_simple.py:109) |
| `--use-opd` | 开启 OPD。 | 必须配 `--opd-type`。 | 参数不全会启动失败。 | [来源](/Users/qitongli/Developer/slime/examples/on_policy_distillation/README.md:16) |
| `--use-rollout-correction` | 开启 IS/RS 修正。 | 与 `--use-tis`、MIS 配置联动。 | 过强过滤降低样本有效率。 | [来源](/Users/qitongli/Developer/slime/examples/train_infer_mismatch_helper/README.md:99) |
| `--use-rollout-logprobs` | 直接使用 rollout logprobs 计算损失。 | 要求 token/logprob 对齐。 | 对齐错误会污染梯度。 | [来源](/Users/qitongli/Developer/slime/examples/train_infer_mismatch_helper/README.md:20) |

## 附录B：文档来源导航

> 目标：保证你能快速定位“这段总结对应原文哪里”。每个文档至少出现一次。

| 文档 | 主要用途 | 在本文对应章节 |
| --- | --- | --- |
| [docs/zh/index.rst](/Users/qitongli/Developer/slime/docs/zh/index.rst:1) | 总体目录与能力边界 | 1 |
| [docs/zh/get_started/quick_start.md](/Users/qitongli/Developer/slime/docs/zh/get_started/quick_start.md:107) | 快速上手、参数分组、特性入口 | 2/3/5 |
| [docs/zh/get_started/usage.md](/Users/qitongli/Developer/slime/docs/zh/get_started/usage.md:14) | 参数语义与机制说明主文档 | 2/3 |
| [docs/zh/get_started/customization.md](/Users/qitongli/Developer/slime/docs/zh/get_started/customization.md:5) | 自定义接口总表与函数签名 | 2 |
| [docs/zh/get_started/qa.md](/Users/qitongli/Developer/slime/docs/zh/get_started/qa.md:1) | 高频问题与一线排障经验 | 3/4/7 |
| [docs/zh/developer_guide/debug.md](/Users/qitongli/Developer/slime/docs/zh/developer_guide/debug.md:1) | 精度对齐、分离调试、IMA 定位 | 4/7 |
| [docs/zh/developer_guide/profiling.md](/Users/qitongli/Developer/slime/docs/zh/developer_guide/profiling.md:1) | rollout profiling 标准流程 | 4 |
| [docs/zh/advanced/slime-router.md](/Users/qitongli/Developer/slime/docs/zh/advanced/slime-router.md:1) | SlimeRouter、R3、会话亲和 | 2/3 |
| [docs/zh/advanced/speculative-decoding.md](/Users/qitongli/Developer/slime/docs/zh/advanced/speculative-decoding.md:1) | 投机采样与 MTP 在线训练 | 3 |
| [docs/zh/advanced/low-precision.md](/Users/qitongli/Developer/slime/docs/zh/advanced/low-precision.md:1) | FP8/INT4 训练与推理约束 | 3/7 |
| [docs/zh/advanced/reproducibility.md](/Users/qitongli/Developer/slime/docs/zh/advanced/reproducibility.md:1) | 全确定性训练配置 | 4 |
| [docs/zh/advanced/fault-tolerance.md](/Users/qitongli/Developer/slime/docs/zh/advanced/fault-tolerance.md:1) | rollout 容灾机制与健康检查参数 | 4 |
| [docs/zh/advanced/pd-disaggregation.md](/Users/qitongli/Developer/slime/docs/zh/advanced/pd-disaggregation.md:1) | Prefill/Decode 分离部署 | 2 |
| [docs/zh/advanced/arch-support-beyond-megatron.md](/Users/qitongli/Developer/slime/docs/zh/advanced/arch-support-beyond-megatron.md:1) | 新架构快速接入与限制 | 7 |
| [docs/zh/examples/qwen3-4B.md](/Users/qitongli/Developer/slime/docs/zh/examples/qwen3-4B.md:1) | Dense 基线样板（含动态采样/异步） | 5 |
| [docs/zh/examples/glm4-9B.md](/Users/qitongli/Developer/slime/docs/zh/examples/glm4-9B.md:1) | Dense 基线样板（GLM 口径） | 5 |
| [docs/zh/examples/qwen3-30B-A3B.md](/Users/qitongli/Developer/slime/docs/zh/examples/qwen3-30B-A3B.md:1) | 中型 MoE 样板 | 5 |
| [docs/zh/examples/glm4.5-355B-A32B.md](/Users/qitongli/Developer/slime/docs/zh/examples/glm4.5-355B-A32B.md:1) | 大规模 MoE 多机样板 | 5 |
| [docs/zh/examples/deepseek-r1.md](/Users/qitongli/Developer/slime/docs/zh/examples/deepseek-r1.md:1) | 超大规模 MoE 与长序列样板 | 5 |
| [docs/zh/examples/qwen3-4b-base-openhermes.md](/Users/qitongli/Developer/slime/docs/zh/examples/qwen3-4b-base-openhermes.md:1) | SFT 模式样板 | 5 |
| [docs/zh/examples/qwen3-next-80B-A3B.md](/Users/qitongli/Developer/slime/docs/zh/examples/qwen3-next-80B-A3B.md:1) | Qwen3-next 依赖与补丁特例 | 5 |
| [docs/zh/blogs/introducing_slime.md](/Users/qitongli/Developer/slime/docs/zh/blogs/introducing_slime.md:1) | 架构哲学与设计取向 | 1/2 |
| [docs/zh/blogs/release_v0.1.0.md](/Users/qitongli/Developer/slime/docs/zh/blogs/release_v0.1.0.md:1) | 性能优化思路、checklist、CI 正确性 | 3/4/7 |
| [examples/README.md](/Users/qitongli/Developer/slime/examples/README.md:1) | examples 全景索引与能力边界 | 9 |
| [examples/eval_multi_task/README.md](/Users/qitongli/Developer/slime/examples/eval_multi_task/README.md:1) | 多任务评估配置化（`--eval-config`） | 2/5/9 |
| [examples/fully_async/README.md](/Users/qitongli/Developer/slime/examples/fully_async/README.md:1) | fully-async 训练驱动与限制 | 2/3/4/9 |
| [examples/geo3k_vlm/README.md](/Users/qitongli/Developer/slime/examples/geo3k_vlm/README.md:1) | VLM 单轮双后端与数值稳定性 | 5/7 |
| [examples/geo3k_vlm_multi_turn/README.md](/Users/qitongli/Developer/slime/examples/geo3k_vlm_multi_turn/README.md:1) | VLM 多轮环境 API 与配置接入 | 2/5/9 |
| [examples/multi_agent/README.md](/Users/qitongli/Developer/slime/examples/multi_agent/README.md:1) | 多角色并行 agent 模板 | 5/7/9 |
| [examples/on_policy_distillation/README.md](/Users/qitongli/Developer/slime/examples/on_policy_distillation/README.md:1) | OPD 参数、teacher 模式与约束 | 2/3/4/5/9 |
| [examples/retool/README.md](/Users/qitongli/Developer/slime/examples/retool/README.md:1) | 工具调用 + SFT->RL 流水线 | 5/9 |
| [examples/search-r1/README_zh.md](/Users/qitongli/Developer/slime/examples/search-r1/README_zh.md:1) | 检索型多轮 tool-calling 与 TIS 联动 | 3/5/7/9 |
| [examples/strands_sglang/README.md](/Users/qitongli/Developer/slime/examples/strands_sglang/README.md:1) | Strands-SGLang 集成与 TITO 轨迹 | 4/5/9 |
| [examples/tau-bench/README.md](/Users/qitongli/Developer/slime/examples/tau-bench/README.md:1) | 用户模拟环境中的 agentic RL | 4/5/9 |
| [examples/train_infer_mismatch_helper/README.md](/Users/qitongli/Developer/slime/examples/train_infer_mismatch_helper/README.md:1) | MIS/TIS 算法与 mismatch 指标体系 | 2/3/4/7/9 |
| [examples/true_on_policy/README.md](/Users/qitongli/Developer/slime/examples/true_on_policy/README.md:1) | 训推严格对齐与验证指标 | 2/3/4/5/9 |
| [examples/true_on_policy_vlm/README.md](/Users/qitongli/Developer/slime/examples/true_on_policy_vlm/README.md:1) | VLM true-on-policy 特殊约束 | 2/5/9 |
| [docs/en/platform_support/amd_tutorial.md](/Users/qitongli/Developer/slime/docs/en/platform_support/amd_tutorial.md:1) | AMD/ROCm 特有配置与限制 | 6 |
