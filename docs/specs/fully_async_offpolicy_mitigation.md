# Fully Async Off-Policy Mitigation 与正确性验证指南

## 1. 这份文档解决什么问题

这份文档有两个目的：

1. 记录这次对 `examples/fully_async` 的改造，说明现在到底补了什么。
2. 给出一套可以在面试中直接复述的、完整的 fully async 正确性验证方法。

重点先说清楚一件事：

**不能只看 sync 和 fully async 的 reward 曲线。**

原因很简单。reward 曲线只能告诉你“最后分数长什么样”，但它不能告诉你：

- 样本是不是已经 stale 到离谱了
- mismatch 是否正在失控
- 吞吐提升是不是靠“吃了更多脏样本”换来的
- async 版本是不是只是 wall-clock 看起来更快，但 equal-token / equal-update 其实更差

所以，验证 fully async 是否“正确”，必须同时看：

- 链路正确性
- off-policy 健康度
- 学习效果
- 系统收益

## 2. 当前 fully async 的问题是什么

原始 `fully_async_rollout.py` 是一个极简示例：

- 后台常驻 worker 持续从 data buffer 取 prompt group
- 异步生成完成后塞进 output queue
- 训练侧只负责 drain 已完成结果

这个形态确实让 rollout 和 train 重叠了，但也天然引入了 off-policy 风险：

- 一个 group 可能跨多个权重版本才被训练侧消费
- `Sample.weight_versions` 虽然已经被 rollout 记录，但原示例根本没消费
- 原示例绕开了标准 `sglang_rollout` 已有的 hook 面
- 原示例只停留在“能跑”，没有任何 version-aware staleness control

## 3. 这次改了什么

这次只做 example-first，不做 core-first 重构。目标不是把 fully async 扶成框架一等公民，而是先把它从“危险示例”修成“有护栏的实验系统”。

### 3.1 补回标准 rollout hook 面

现在 `examples/fully_async/fully_async_rollout.py` 已经接回标准 rollout 的三个 hook：

- `dynamic_sampling_filter_path`
- `rollout_sample_filter_path`
- `rollout_all_samples_process_path`

这意味着 fully async 示例不再绕开标准 rollout 能力，而是和同步路径保持基本一致的 hook 语义。

### 3.2 增加 version-aware stale filtering

现在 fully async 示例会对每个 group 做版本治理：

- 读取该 group 所有 `sample.weight_versions`
- 取其中最大、且能解析成整数的版本作为 `group_weight_version`
- worker 持续维护 `latest_seen_weight_version`
- 如果满足以下任一条件，则直接丢弃该 group：
  - `group_weight_version` 不可用，且 `fully_async_drop_unknown_version=true`
  - `latest_seen_weight_version - group_weight_version > fully_async_max_staleness`

注意：这里默认是 **drop，不 requeue**。原因很现实，旧样本 requeue 以后也不会自己变新。

### 3.3 示例脚本切到 `MIS + keep-old-actor`

示例脚本现在默认开启：

- `--keep-old-actor`
- `--custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp`
- `--custom-config-path examples/fully_async/fully_async_offpolicy.yaml`

这意味着 fully async 示例默认不再只用最原始的 `vanilla_tis`，而是进入更接近真实实验配置的 mismatch-correction 路径。

## 4. 为什么“只看 reward 曲线”不够

只看 reward 曲线，至少会漏掉 4 类问题。

### 4.1 你不知道样本是不是 stale

reward 曲线可能还在涨，但训练实际上已经在吃很旧的样本。  
这时你看到的是“模型还能学”，不是“fully async 是正确的”。

### 4.2 你不知道 mismatch 有没有爆炸

async 训练里最危险的情况是：

- rollout policy 和 training policy 偏差越来越大
- importance ratio 被大量裁剪
- 训练并没有立即崩，但已经进入高偏差 / 高方差区

reward 曲线在这个阶段可能还没来得及反映问题。

### 4.3 你不知道 wall-clock 的提升是不是偷来的

fully async 经常会出现一种假象：

- wall-clock 上更快
- 但它在相同时间里喂了更多样本
- 或者用了更多 stale 样本

如果你不按 equal-token / equal-update 对齐，就没法证明“训练更对”，只能证明“系统更忙”。

### 4.4 你不知道问题来自算法，还是来自系统编排

sync vs fully async 的差异，可能来自：

- driver 不同
- sample 新鲜度不同
- mismatch correction 不同
- 资源切分不同

如果只画两条 reward 曲线，根本分不清因果。

## 5. 我们如何定义 fully async 的“正确性”

我建议把正确性拆成 4 层。面试时就按这 4 层说。

### 5.1 链路正确性

问题是：系统有没有真的按你设计的方式工作？

要确认：

- fully async worker 确实持续生成
- stale gate 确实在工作
- dynamic filter / sample filter / all-samples hook 确实被调用
- 没有空 batch、NaN、死循环、持续无法收敛的 queue

### 5.2 Off-policy 健康度

问题是：虽然是 async，但 off-policy 偏差有没有被控制在合理范围？

要确认：

- rollout 和 training 的 logprob 差异没有持续恶化
- importance sampling 权重没有长期大面积被裁剪或屏蔽
- stale 样本比例可控

### 5.3 学习效果

问题是：equal-update / equal-token 下，fully async 学出来的模型是否不比 sync 差？

要确认：

- holdout eval 指标接近 sync baseline
- pass@k / eval reward 没有系统性退化
- 多个 seed 下趋势一致，而不是单次运气好

### 5.4 系统收益

问题是：正确性不坏的前提下，fully async 有没有带来吞吐收益？

要确认：

- `step_time` 降低
- `wait_time_ratio` 降低
- train 和 rollout 的重叠确实增加

只有这四层同时满足，我才会说 fully async 是“正确且有工程价值”的。

## 6. 推荐实验矩阵

不要只做两组。最少做四组。

### S0: Sync Baseline

- `train.py`
- 标准同步 rollout

作用：

- 作为学习效果与系统表现的基线

### A0: Async Driver Baseline

- `train_async.py`
- 但不用 fully async worker

作用：

- 分离“async driver 带来的收益”和“fully async worker 带来的收益”

### A1: Raw Fully Async

- fully async worker
- 不开 stale gate
- 不接 MIS helper

作用：

- 故意暴露 fully async 的原始 off-policy 问题
- 给 A2 提供反例对照

### A2: Mitigated Fully Async

- fully async worker
- stale gate
- MIS helper
- `keep-old-actor`

作用：

- 这是你真正要证明“可用”的版本

## 7. 对比实验必须锁死什么变量

这一步非常关键。否则结论没意义。

- 相同初始 checkpoint
- 相同训练数据和评测数据
- 相同 reward model
- 相同 `global_batch_size`
- 相同 `n_samples_per_prompt`
- 相同训练步数
- 相同 GPU 配置
- 至少 3 个随机 seed

比较时至少要看 3 个横轴：

- 按 optimizer updates
- 按 consumed/generated tokens
- 按 wall-clock

面试里可以直接说：

> “我不会只按时间对比。我会同时按 update budget、token budget 和 wall-clock 对齐，因为 fully async 很容易在其中一个坐标系里看起来更好，但那不一定代表训练更正确。”

## 8. 指标说明：每个指标到底代表什么

下面这部分最重要。你说“没看懂指标代表什么”，问题就在这里。

### 8.1 学习结果指标

#### `eval/<dataset>`

代表什么：

- holdout eval 数据集上的平均得分

为什么看它：

- 这是最接近“模型最后学得怎么样”的指标

怎么看：

- fully async 是否在 equal-update / equal-token 条件下接近 sync baseline

危险信号：

- wall-clock 上更快，但 equal-token 下 eval 明显更差

#### `passrate/pass@k`

代表什么：

- 对同一题生成多个样本时，至少有一个正确的概率

为什么看它：

- 对 reasoning / math 任务，比单点 reward 更稳

怎么看：

- fully async 的 pass@k 不应系统性低于 sync

危险信号：

- rollout reward 还行，但 pass@k 掉得很厉害

这通常说明训练分布在漂，模型只是学会了某些投机行为。

### 8.2 rollout 侧链路指标

#### `rollout/truncated_ratio`

代表什么：

- rollout 中被截断的样本比例

为什么看它：

- 如果 async 后截断比例突然升高，可能说明并发、长度分布或队列时延出了问题

怎么看：

- fully async 与 sync 相比不应出现异常跳升

危险信号：

- 截断比例持续明显高于 baseline

#### `rollout/response_len/{mean,median,max,min}`

代表什么：

- 生成长度分布

为什么看它：

- 它帮助你判断吞吐变化究竟是来自系统重叠，还是来自样本长度分布变化

怎么看：

- 各组长度分布应该可比

危险信号：

- fully async 的平均长度显著变短，但 reward 看起来更高

这可能只是任务难度被悄悄改了，不能算正确性提升。

#### `rollout/repetition_frac`

代表什么：

- 输出出现明显重复模式的比例

为什么看它：

- 训练出问题时，经常先从重复输出表现出来

危险信号：

- reward 没立刻坏，但 repetition 持续上升

### 8.3 系统吞吐指标

#### `perf/train_wait_time`

代表什么：

- 训练 actor 不在训练、而是在等待其他阶段的时间

为什么看它：

- 它是 async 优化最直接要打掉的“空等时间”

怎么看：

- fully async 成功的话，这个值应该下降

#### `perf/rollout_time`

代表什么：

- rollout 阶段总耗时

为什么看它：

- 帮你判断瓶颈到底是不是 rollout 自己慢

怎么看：

- 它下降是好事，但不是充分条件

危险信号：

- `rollout_time` 不变甚至更长，但 `step_time` 降了

这说明收益可能来自 overlap，而不是 rollout 本身变快。这不是坏事，但你要讲清楚。

#### `perf/step_time`

代表什么：

- 一个训练 step 的总耗时

为什么看它：

- 这是系统端最直观的吞吐指标

怎么看：

- fully async 的目标之一就是降低它

#### `perf/wait_time_ratio`

代表什么：

- `train_wait_time / step_time`

为什么看它：

- 它告诉你一个 step 里有多大比例时间训练 actor 在“没干活”

怎么看：

- fully async 做对了，这个比例应该下降

危险信号：

- reward 差不多，但 `wait_time_ratio` 没降

这说明系统收益并不成立。

### 8.4 mismatch / off-policy 指标

这部分才是 fully async 正确性验证的核心。

#### `train_rollout_logprob_abs_diff`

代表什么：

- 训练侧 old/proximal logprob 和 rollout logprob 的绝对差

为什么看它：

- 这是最直接的 train-vs-rollout mismatch 指标

怎么看：

- 它在 A1 里通常会比 S0 高，这是正常现象
- A2 应该比 A1 更稳，至少不要持续单边恶化

危险信号：

- 这个值一路升，不回落

这通常说明：

- stale 样本过多
- 或 update cadence 太快
- 或 MIS 根本没压住 mismatch

#### `ois`

代表什么：

- 当前训练 policy 与 old/proximal policy 之间的 on-policy importance ratio

为什么看它：

- 它帮助区分“训练内部更新过猛”和“rollout mismatch 太大”这两类问题

怎么看：

- 一般希望它稳定，不要剧烈抖动

危险信号：

- `ois` 本身就非常不稳定

说明 PPO 这边的更新半径都开始失控了。

#### `mis_tis_weight_before_bound`

代表什么：

- rollout/training 原始 importance ratio，还没裁剪之前的大小

为什么看它：

- 它反映 mismatch 的原始强度

怎么看：

- 可以比 S0 大，但不能长期离谱

危险信号：

- 长期偏大，且方差很高

说明 fully async 的 stale 问题在积累。

#### `mis_tis_clip_fraction_low` / `mis_tis_clip_fraction_high`

代表什么：

- 有多少 importance weight 触发了下界或上界裁剪

为什么看它：

- 这是“off-policy correction 被迫出手多频繁”的直接证据

怎么看：

- 少量、稳定的裁剪是正常的
- 如果大面积、长期裁剪，说明 mismatch 已经很重

危险信号：

- 这两个值持续很高

这表示训练大量依赖 clipping 才没炸，正确性很可疑。

#### `mis_is_ratio_mean_final`

代表什么：

- 经过 MIS 处理后的最终 importance ratio 均值

为什么看它：

- 帮你看“修正后的分布”是否还算健康

怎么看：

- 如果开启了 batch normalize，通常会更稳定

危险信号：

- 最终 ratio 仍然剧烈波动

说明 correction 不够，或者 staleness threshold 太松。

### 8.5 fully async 新增版本治理指标

#### `fully_async/stale_drop_count`

代表什么：

- 因为 version 过旧而被丢弃的 group 数量

为什么看它：

- 这是 stale gate 是否真的生效的最直接证据

怎么看：

- 在有明确 staleness 限制的配置下，它不应该永远为 0

危险信号：

- `max_staleness=0` 仍然长期是 0

这通常说明 gate 没生效，或者系统实际上没发生跨版本积压。

#### `fully_async/stale_drop_ratio`

代表什么：

- 被检查的 group 里，有多少比例因为 stale 被丢弃

为什么看它：

- 它告诉你 fully async 到底有多依赖“吃旧样本”

怎么看：

- 这个值不是越低越好，也不是越高越好
- 太低，可能 gate 没起作用
- 太高，说明系统在拿大量样本利用率换正确性

经验判断：

- 稳定、小幅存在通常是健康的
- 持续高比例则说明配置不对

#### `fully_async/unknown_version_drop_count`

代表什么：

- 因为版本信息缺失而被丢弃的 group 数量

为什么看它：

- 它能告诉你 rollout 版本链路有没有断

危险信号：

- 这个值持续增长

说明 version metadata 采集或传递有问题。

#### `fully_async/max_seen_weight_version`

代表什么：

- worker 迄今为止见过的最大 rollout 权重版本

为什么看它：

- 用来判断系统是否真的跨版本运行

#### `fully_async/min_accepted_weight_version`

代表什么：

- 当前接受进训练的 group 里，最老的版本

为什么看它：

- 帮你衡量训练集里的“最老样本年龄”

怎么看：

- 它和 `max_seen_weight_version` 的差越大，说明 accepted 数据越 stale

## 9. 正确的判定流程

我建议按下面顺序判断。

### 第一步：先判链路

确认：

- fully async run 没崩
- stale metrics 有值
- dynamic filter / sample filter metrics 正常

如果连这一步都不过，别去看 reward。

### 第二步：再判 off-policy 健康度

重点看：

- `train_rollout_logprob_abs_diff`
- `mis_tis_weight_before_bound`
- `mis_tis_clip_fraction_low/high`
- `fully_async/stale_drop_ratio`

结论方式：

- A1 比 S0 更差，是正常的
- A2 如果没有明显好于 A1，说明 mitigation 不成立

### 第三步：再判学习效果

重点看：

- `eval/<dataset>`
- `passrate/pass@k`

结论方式：

- equal-update 下，A2 最终效果应接近 S0
- equal-token 下，A2 不应系统性差于 S0

面试里可以用这个表达：

> “如果 fully async 只是 wall-clock 更快，但 equal-token 下最终 eval 更差，我不会说它正确，只会说它更激进。”

### 第四步：最后判系统收益

重点看：

- `perf/step_time`
- `perf/train_wait_time`
- `perf/wait_time_ratio`

结论方式：

- A2 相比 S0，如果 step_time 和 wait_time_ratio 都下降，同时学习效果不坏，才算真正有工程价值

## 10. 面试里怎么讲

### 10.1 30 秒版本

> “我不会只拿 sync 和 fully async 的 reward 曲线做结论，因为那只能看结果，不能证明训练过程是对的。我会把 fully async 的正确性拆成四层：链路正确性、off-policy 健康度、学习效果和系统收益。实验上至少做 sync baseline、async driver、原始 fully async 和 mitigated fully async 四组，然后同时比较 eval 指标、`train_rollout_logprob_abs_diff` / `mis_*` / stale-drop 指标，以及 `step_time` / `wait_time_ratio`。只有当 fully async 在 equal-update 和 equal-token 下不明显差于 sync，并且 mismatch 被控制住、吞吐确实提升，我才认为它是正确的。” 

### 10.2 2 分钟版本

> “fully async 的核心风险不是 reward 曲线不好看，而是它可能在你看不到的地方 silently 变成 off-policy 训练。所以我验证它的时候不会只看 reward。我先验证链路：worker 是否持续生成，stale gate 和 hook 是否真的触发。然后看 off-policy 健康度，重点是 `train_rollout_logprob_abs_diff`、`ois`、`mis_tis_weight_before_bound`、裁剪比例，以及我们自己加的 `fully_async/stale_drop_ratio`。这些指标告诉我 mismatch 有没有被压住。再往上才是学习效果，我会在 equal-update 和 equal-token 条件下比较 holdout eval 和 pass@k，要求 fully async mitigated 至少接近 sync baseline。最后才是系统收益，看 `step_time` 和 `wait_time_ratio` 是否下降。这样我就能把‘更快’和‘没把训练搞坏’同时证明出来。” 

## 11. 这套方法的核心结论

一句话总结：

**fully async 的正确性，不是“reward 曲线能不能涨”，而是“在可控 mismatch 下，equal-budget 学习效果不坏，同时系统吞吐确实更好”。**

这也是你在面试里最应该强调的点。

## 12. 后续还没做的事

这次仍然只是 example-first。

还没做的核心扩展包括：

- 把 `rollout_weight_version` 透传进标准 train batch
- 把 staleness 参数升为框架级 CLI
- 在训练侧增加 version-aware 监控与拒绝策略
- 把 fully async 从 example 提升成框架一等 rollout 模式

这几项如果继续做，验证方法本身不变，只是链路会更标准、更容易自动化。
