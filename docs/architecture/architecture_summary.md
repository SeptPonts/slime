# Slime 训练架构分析

> "差劲的程序员担心代码，优秀的程序员担心数据结构及其关系。" — Linus Torvalds

本文档对 `slime` 项目的训练架构进行拆解分析，重点关注关键数据流路径、Rollout 与 Training 之间的关注点分离，以及基于 Ray 的任务编排机制。

## 1. 核心架构概览

系统围绕基于 Ray 的**客户端-服务端（Client-Server）**模型设计，将**数据生成（Rollout）**与**模型优化（Training）**解耦分离。

```mermaid
flowchart TD
    subgraph Orchestrator [主进程 (train.py)]
        Loop[训练循环]
        PG[Placement Groups 配置]
    end

    subgraph DataGeneration [Rollout Manager]
        RM[RolloutManager Actor]
        SGL[SGLang Engines xN]
        RM -->|管理| SGL
        RM -->|生成| Sample[List[Sample]]
        Sample -->|转换| Batch[RolloutBatch]
    end

    subgraph Training [Training Group]
        TG[RayTrainGroup]
        W[Worker Actors xN]
        TG -->|管理| W
        W -->|训练| Weights[模型权重]
    end

    Loop -->|1. 生成| RM
    RM -->|2. 数据引用| Loop
    Loop -->|3. Train(数据引用)| TG
    TG -->|4. 更新权重| Loop
    Loop -->|5. 同步权重| RM
```

## 2. 关键数据结构

系统依赖两种主要数据表示形式，两者之间的转换发生在 `RolloutManager` 内部。

### A. 原子单元（Atomic Unit）：`Sample`（`slime.utils.types.Sample`）
这是推理引擎的原始输出，是一个包含所有元数据的富对象。
- **归属**：由 `RolloutManager` 内部的 `SGLangEngine` 创建。
- **关键字段**：
    - `tokens`：生成的 token ID 序列。
    - `response_length`：生成响应的长度。
    - `reward`：原始奖励值。
    - `status`：生命周期状态（Lifecycle State）（PENDING、COMPLETED、TRUNCATED）。
    - `log_probs` / `routed_experts`：PPO/GRPO 的策略元数据（Policy Metadata）。

### B. 训练载荷（Training Payload）：`RolloutBatch`
这是为训练循环优化的张量/列表字典。
- **归属**：由 `RolloutManager._convert_samples_to_train_data` 生成。
- **结构**：
    ```python
    {
        "tokens": List[List[int]],       # 输入 ID
        "loss_masks": List[List[int]],   # 损失计算掩码
        "rewards": List[float],          # 归一化奖励
        "raw_reward": List[float],       # 原始奖励
        "truncation": List[int],         # 状态标志
        # ... 后端特定字段
    }
    ```

## 3. 组件深度剖析

### 编排器（Orchestrator）（`train.py` / `train_async.py`）
- **职责**：资源分配与流程排序。
- **关键机制**：
    - **资源分配（Resource Allocation）**：使用 `slime.ray.placement_group.create_placement_groups` 严格为 Rollout 与 Training 预留 GPU，通过隔离（Isolation）防止 OOM。
    - **异步与同步（Async vs Sync）**：
        - `train.py`：阻塞调用（Blocking Call）（`ray.get`）确保严格顺序执行，易于调试。
        - `train_async.py`：流水线化（Pipeline）生成与训练。`Step N` 训练期间，`Step N+1` 同时生成。使用 `ray.ObjectRef` 传递未来数据，无需阻塞。

### Rollout 管理器（`slime.ray.rollout.RolloutManager`）
- **性质**：推理管理的"上帝对象（God Object）"。
- **核心职责**：
    - **Actor 管理**：创建并监控 `SGLangEngine` Actor。
    - **数据转换（Data Transformation）**：将高度结构化的 `Sample` 对象转换为扁平的 `RolloutBatch` 字典（`_convert_samples_to_train_data`）。
    - **容错处理（Fault Tolerance）**：包含模拟和处理 Worker 崩溃的逻辑（`_try_ci_fault_injection`）。
    - **权重同步（Weight Sync）**：处理 `onload_weights`，将训练 Actor 的最新模型参数拉取到推理引擎中。

### Actor 组（`slime.ray.actor_group.RayTrainGroup`）
- **性质**：分布式进程组的门面（Facade）。
- **核心职责**：
    - 对 `Megatron` 或 `FSDP` 后端的统一抽象。
    - **并行执行（Parallel Execution）**：`async_train` 等方法将调用广播至所有 Worker Actor，并返回 `ObjectRef` 列表。
    - **拓扑感知（Topology Awareness）**：根据第一个 Actor 的 IP，为 `torch.distributed` 初始化设置 `MASTER_ADDR`/`PORT`。

## 4. 关键设计决策与权衡

### Separated（分离式）vs Colocated（共置式）
- **Separated（分离式，默认）**：Rollout 与 Training 在不同 GPU 上运行。
    - ✅ **优点**：零内存竞争（Zero Memory Contention），训练不会导致推理崩溃。
    - ❌ **缺点**：硬件成本更高，权重同步（Weight Sync）存在网络开销。
- **Colocated（共置式）**：`train_async.py` 中未完全支持（有显式断言）。

### 异步流水线（Async Pipelining）
- **机制**：`train_async.py` 在 `rollout_id` 训练完成前就启动 `rollout_id + 1` 的生成。
- **复杂性**：需要精确管理"是哪个版本的权重生成了这批数据"。
    - *风险*：若权重更新发生在生成完成之前，会引发 Off-policy（离策略）问题。代码通过在 `update_weights` *之前*同步生成来处理此问题。

### ObjectRef 作为数据管道（Data Pipe）
- 系统在 Rollout 与 Training 之间传递 `ray.ObjectRef`（数据引用），而非数据本身。
- 这为编排器带来了"零拷贝（Zero-Copy）"效果——实际的重量级数据传输直接发生在 Rollout Worker 节点与 Training Worker 节点之间（通过 Ray 的对象存储（Object Store））。

## 5. Linus 点评

### 优点（Good Taste）
- **数据中心化流程（Data-Centric Flow）**：`Sample`（逻辑单元）与 `RolloutBatch`（计算单元）的清晰区分是扎实的设计。
- **显式资源管理（Explicit Resource Management）**：`placement_group` 逻辑具有确定性（`sort_key`），避免了分布式系统中常见的"随机卡死"问题。

### 缺点（Code Smells）
- **RolloutManager 过于复杂**：`RolloutManager` 类承担了过多职责——故障注入（Fault Injection）、指标计算（`_compute_perf_metrics`）、数据转换、Actor 管理，800+ 行代码混杂多种关注点（Mixed Concerns）。
    - *建议*：将数据转换逻辑提取为无状态工具函数，将指标逻辑提取为独立的 `MetricsCollector`。
- **特殊情况处理**：`train.py` 中存在 `if args.offload_rollout` 这类判断，说明架构在内存约束上力不从心，用条件逻辑打补丁，而非采用统一的内存模型（Unified Memory Model）。

### 隐患（Risks）
- **SGLang 强依赖（Tight Coupling）**：`RolloutManager` 与 `SGLangEngine` 紧密耦合。若要替换推理后端（如换成 vLLM），`RolloutManager` 需要大规模重写。
- **权重校验**：`check_weights` 逻辑暗示对分布式同步机制缺乏信任。"信任但验证（Trust but Verify）"是好习惯，但若主循环中必须做此校验，说明基础设施的可信度偏低。
