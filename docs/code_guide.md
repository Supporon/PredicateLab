# PredicateLab 代码详解与二次开发指南（V1.1）

> 目标读者：第一次接手本项目、需要读懂代码并继续修改的人。
>
> 读完本文你应该能回答四个问题：
>
> 1. 这个项目到底在做什么？
> 2. 一条命令从配置走到报告，中间经过了哪些模块？
> 3. 结果里的 `status` / `state` / `p` / `q` / `effect` 分别是什么意思？
> 4. 我要加一个谓词、改一个阈值、加一个数据域或改数据库表，应该动哪里？

本文以当前仓库 V1.1 代码为准；规范性协议细节见
[V1.1 评估协议](v1.1_evaluation_protocol.md)，数据域扩展清单见
[扩展新数据域](extending_domains.md)。

---

## 0. 先建立全局印象

PredicateLab 不是一个“训练模型”的项目，而是一个**谓词发现与证据验证流水线**：

```
候选谓词（JSON 数据）
   │  生成器提出
   ▼
结构校验 → 绑定数据快照 → 探针（训练折）→ 去重
   ▼
筛选折（screen）→ 多折 × 多 seed 完整验证 → 统计检验
   ▼
证据入库（成功/失败/不确定都保留）
   ▼
冻结候选族 → 独立确认（更晚的数据）→ 一次性留出集终评
```

它刻意把“发现”和“确认”分开：

- **发现阶段**：在搜索折上找看起来有效的谓词；
- **确认阶段**：在冻结候选族之后，用更晚/独立的数据复核；
- **终评阶段**：一次性读取留出集，读取即永久消耗。

因此，代码里最重要的不是某个模型，而是**身份（identity）、状态机（state machine）、
访问控制（holdout guard/ledger）和统计口径**。

---

## 1. 目录地图

```
predicatelab/
├── cli.py                  # 命令行入口：discover / freeze-family / confirm / finalize / report / export ...
├── core/                   # 与领域无关的核心
│   ├── schema.py           # 所有 Pydantic 数据模型（模板、候选、折、证据、试次、终评...）
│   ├── adapter.py          # DomainAdapter / CandidateGenerator 抽象 + TrialRequest + 试次身份哈希
│   ├── engine.py           # 生命周期调度、筛选/验证/聚合/分类、报告、导出
│   ├── store.py            # SQLite 状态存储、迁移、缓存校验、holdout 账本封装
│   ├── protocol.py         # purge/embargo、折协议静态校验、HoldoutGuard
│   ├── label_window.py     # 标签未来依赖解析 + 按真实交易日裁剪各分区尾部
│   ├── holdout.py          # 全局留出集账本（原子 claim、consumed 语义）
│   ├── identity.py         # 代码/环境/数据内容身份、严格 JSON、原子写
│   ├── stats.py            # Newey-West、block bootstrap、paired effect、BH、stability
│   ├── data.py             # 文件清单/snapshot 辅助 + 数据适用性聚合
│   └── hashing.py          # canonical JSON + SHA-256 / short id
├── domains/
│   ├── __init__.py         # create_adapter / executable_domains / reserved_domains
│   ├── tabular/            # 可执行的 sklearn 乳腺数据集适配器（验证核心与领域解耦）
│   ├── tsil/               # 可执行的 qlib + TSIL LSTM/LUSI 适配器
│   ├── mineral/            # 仅契约，执行路径抛 DomainNotImplemented
│   └── image/              # 仅契约，执行路径抛 DomainNotImplemented
configs/                    # 实验配置
tests/                      # 128 个测试（单元 + 对抗性协议测试 + 端到端 + V1.1 缺陷回归）
docs/                       # 协议、架构、结果、扩展说明
TSIL-main/                  # 上游 TSIL 代码（数据、模型、损失、原生谓词）
artifacts/                  # 运行产物（.gitignore，不提交）
```

依赖方向非常严格：

```
domains/*  ──依赖──►  core/*        （领域依赖核心）
core/*     ──绝不依赖──►  domains/* 或 qlib/torch/sklearn
```

核心只通过 `DomainAdapter` 接口与领域交互。发现 `core/` 里出现 `import torch`、
`import qlib`、`import sklearn`，基本就是设计被破坏了。

---

## 2. 三个关键模型

### 2.1 三层候选身份（防止“拟合参数混进谓词定义”）

[schema.py:116](../predicatelab/core/schema.py#L116) 起定义了三种对象：

| 层级 | 模型 | 含义 | 身份哈希 |
|---|---|---|---|
| 模板 | `PredicateTemplate` | “这个谓词说的是什么”，与数据无关 | `sha256(domain, kind, descriptor, schema_version)` |
| 绑定 | `BoundCandidate` | “这个谓词作用于哪个不可变数据快照” | `sha256(template_hash, dataset_id, snapshot_hash, bound_scope)` |
| 拟合 | `FittedInstance` | “在某个训练折内拟合出的阈值/分位数” | `sha256(candidate_id, fold_id, fitted_params, fit_data_hash)` |

关键纪律：

- 阈值、分位数、标准化参考值**不能**写进 `PredicateTemplate` / `BoundCandidate`；
- 它们必须在 `run_trial` 里只用该折训练集拟合，并通过 `fitted_params` 返回；
- `FittedInstance.fit_data_hash` 让“跨折复用拟合参数”这件事变得可审计。

例子：`threshold(feature 0, q=0.5)` 只定义一次；5 折 CV 会产生 5 个不同的
`FittedInstance`，各自记录自己的训练行哈希。

### 2.2 候选种类与证据状态

`CandidateKind`（[schema.py:42](../predicatelab/core/schema.py#L42)）只是分类标签，
核心不解释 descriptor 内容：

`native / statistical / soft / conditional / relational / constraint / composite`

`EvidenceStatus`（[schema.py:60](../predicatelab/core/schema.py#L60)）是**一次验证的机械结果**：

`validated / rejected / insufficient_evidence / screened_out / duplicate /
invalid_structure / execution_failed / blocked`

`CandidateState`（[schema.py:80](../predicatelab/core/schema.py#L80)）是**允许对外声称的科学状态**：

| state | 含义 |
|---|---|
| `executable` | 能跑，但没有测到有用效果 |
| `exploratory_positive` | 在发现数据上为正；**不能**说成独立确认 |
| `resampling_confirmed` | 在同一数据的其他重采样上确认（不是新数据） |
| `independently_confirmed` | 在更晚、未参与选择的数据上确认 |
| `insufficient` | 证据不足 |
| `rejected` | 显著反向 |
| `legacy_unknown` | 历史行，阶段不可证明 |

**最重要的读表规则：`status=validated` 只等于 `state=exploratory_positive`。**
只有 `confirm` 之后才可能出现 `independently_confirmed` / `resampling_confirmed`。

### 2.3 三阶段生命周期

```
discover ──► freeze-family ──► confirm ──► finalize
（搜索折）    （冻结候选族）    （确认折）    （留出集，仅一次）
```

- `discover`：生成、筛选、完整验证、写证据、写反馈；
- `freeze-family`：把发现阶段 `validated` 的候选固化成不可变清单；
- `confirm`：只评估清单成员，在确认折上运行，结果**不会**反馈给生成器；
- `finalize`：先写冻结 manifest，再向全局账本 claim 留出集，评估一次。

留出集是**全局消耗型资源**，不是“换个 run_id 就能再读一次”的折。它的账本身份
只由数据决定：

```
holdout_key = H(dataset_id, snapshot_hash, data_manifest_hash,
                 label_spec_hash, test_start, test_end)
```

见 [holdout.py:54](../predicatelab/core/holdout.py#L54)。`run_id`、候选、协议时间戳都
不在 key 里。

---

## 3. 一条命令的完整时序

### 3.1 CLI 入口

[cli.py](../predicatelab/cli.py) 很薄，只做 YAML 读取、构造 `PredicateLabEngine`、
调用对应方法、打印 JSON。所有科学逻辑都在 `core/engine.py`。

```bash
export PYTHONPATH=.
PY=/home/dmz-ai/miniconda3/envs/zzr_py312/bin/python

$PY -m predicatelab.cli domains
$PY -m predicatelab.cli validate-config --config configs/tsil_lstm_smoke.yaml
$PY -m predicatelab.cli discover      --config configs/tsil_lstm_smoke.yaml --rounds 1
$PY -m predicatelab.cli freeze-family --config configs/tsil_lstm_smoke.yaml --run-id run_...
$PY -m predicatelab.cli confirm       --config configs/tsil_lstm_smoke.yaml --run-id run_...
$PY -m predicatelab.cli finalize      --config configs/tsil_lstm_smoke.yaml \
    --run-id run_... --candidate-id cand_...
```

### 3.2 `_build_context()`：每次命令的共同前置

[engine.py:209](../predicatelab/core/engine.py#L209)。无论 discover、report 还是
finalize，都会先构造上下文：

1. `create_adapter(config)` 创建领域适配器；
2. `build_protocol` / `data_contract` / `snapshot` / `suitability`；
3. 数据适用性 **fail closed**：有 blocker 直接拒绝；有 warning 必须显式确认
   （`run.warning_acknowledgements` + `run.warning_ack_reason`）；
4. `_resolve_label_dependency`：从标签表达式解析未来交易日依赖；
5. `validate_protocol`：折顺序、purge/embargo、确认折晚于发现折、holdout 不重叠；
6. 如果有未来依赖：用真实交易日历 `validate_and_isolate` 裁剪各分区尾部；
7. 构造 `SearchPolicy`，补默认 screen/validation 折；
8. 计算 `protocol_hash`、代码版本、环境版本、标签规格哈希、`context_hash`；
9. 构造 `HoldoutGuard`；
10. 新 run 建行；已有 run 校验 protocol/snapshot 没变，变了直接报错；
11. 原子写 `context.json`。

`context.json` 是复现实验的第一入口：里面有协议、有效折（裁剪后）、数据快照、
标签隔离报告、代码/环境身份、`identity_hash`。

### 3.3 `discover` 内部

[engine.run()](../predicatelab/core/engine.py#L682) →
[_run_round()](../predicatelab/core/engine.py#L714) →
[_run_round_body()](../predicatelab/core/engine.py#L776)。

每轮：

```
generator.propose
   → validate_descriptor            （结构非法：invalid_structure 证据）
   → add_template / add_candidate   （内容哈希去重）
   → probe（只用 screen 折训练数据） （探针失败/退化：invalid_structure）
   → 行为去重（probe signature 相关性 ≥ search.behavioural_dedup_corr）
   → baseline 在 screen 折跑一次
   → 每个候选在 screen 折跑一次，paired effect
   → 按方向排序，取 top-k 进入完整验证
   → 未晋级：screened_out 证据
   → 完整验证：validation_folds × seeds，每个候选与 baseline 同数据/同预算/同 seed 配对
   → 每个 seed 聚合 → 跨 seed 聚合 → 分类
   → BH 多重检验校正
   → 写 round summary、feedback；round 置 completed
```

几个容易忽略的点：

- **试次缓存**：`_trial_config` 先算 `config_hash`（见
  [adapter.py:160](../predicatelab/core/adapter.py#L160)），`run_trial` 命中校验通过的
  缓存就直接返回。缓存键是 `(candidate_id, fold_id, protocol_hash, config_hash)`。
- **缓存校验**：`get_cached_trial` 会检查 `status=success`、`result_hash` 是否匹配、
  `metric_series` 非空、预测文件是否存在且哈希一致；不通过就当未命中并重跑
  （[store.py:801](../predicatelab/core/store.py#L801)）。
- **失败也入库**：`invalid_structure`、`duplicate`、`screened_out`、
  `execution_failed` 都写 evidence，不是只留赢家。
- **NaN 不是成功**：`sanitise_json` 把 NaN/Inf 换成 `null` 并记录 `null_paths`；
  如果 worker 声称 success 但含非有限值，engine 会降级为 failed
  （[engine.py:484](../predicatelab/core/engine.py#L484)）。
- **断点续跑**：round 的 `summary.proposals` 会保存本轮提案；中断后恢复时复用同一批
  提案，不会偷偷换一批候选。`rounds.status='completed'` 才算完成。

### 3.4 `freeze-family`

[engine.freeze_family()](../predicatelab/core/engine.py#L1233)。

前置条件：

- run 状态必须是 `search_complete` / `confirmation_complete` / `final_complete`；
- 所有 round 都 `completed`；
- 只允许发现阶段 `status=validated` 的候选进入；
- 显式 `--candidate-id` 时逐个校验；不指定时默认取所有 positive（可 `--top-k`）。

产物 `frozen_family.json` 记录候选 ID、descriptor、发现证据、协议/数据/标签/代码/
环境身份、多重检验族、冻结时间，并计算 `manifest_hash`。

### 3.5 `confirm`

[engine.confirm()](../predicatelab/core/engine.py#L1319)。

- 没有确认折直接拒绝；
- 没有冻结族就先调用 `freeze_family`；
- 校验冻结族身份仍匹配当前 protocol/snapshot/data manifest；
- `HoldoutGuard.unlock(manifest_hash)` 后才允许读确认折；
- 对每个成员在确认折 × seeds 上跑候选与 baseline，聚合分类；
- `confirmation_kind=time_forward` → `independently_confirmed`；
  `resampling` → `resampling_confirmed`；
- 再做 BH；不通过的降为 `insufficient`；
- 写 `confirmation.json`，run 置 `confirmation_complete`。

确认阶段的证据**不会**进入生成器的 feedback。

### 3.6 `finalize`

[engine.finalize()](../predicatelab/core/engine.py#L1472)。

守卫顺序（任何一条失败都会拒绝读留出集）：

1. 已有成功终评 → 幂等返回；
2. 本 run 已选过别的终评候选 → 拒绝（一个 run 只有一个终评候选）；
3. run 未完成、round 未完成 → 拒绝；
4. 候选不存在 / 绑定的 snapshot 不一致 → 拒绝；
5. 候选发现证据不是 `validated` → 拒绝；
6. 候选族未冻结，或候选不在族内，或族身份不匹配 → 拒绝；
7. 默认要求已独立确认（`run.require_confirmation: true`）；否则只能做
   **exploratory final**，不能声称 confirmatory；
8. 写 `frozen_manifest.json`；
9. 向全局账本 `claim_holdout`；若已被别的 study 消耗：
   - 非 exploratory → 抛 `HoldoutConsumed`；
   - exploratory 且显式允许 → 记录 `exploratory_reuse` 行，仍标记为非确认；
10. 执行最终评估，`mark_holdout` 状态随成功/失败/中断变化；
11. 写 `final_evaluation.json`、`final_evals` 行，run 置 `final_complete`。

注意：**`failed` / `interrupted` 也算已消耗**，因为数据可能已经被读过。只有完全相同的
`(run_id, candidate_id, manifest_hash)` 重试会复用同一 claim。

### 3.7 产物清单

`artifacts/<run_id>/`：

| 文件 | 何时产生 | 内容 |
|---|---|---|
| `context.json` | 任意命令首次建上下文 | 协议、有效折、快照、标签隔离、代码/环境身份 |
| `report.json` / `report.md` | discover/finalize 结束 | 证据统计、轮次表、终评表、身份 |
| `evidence.jsonl` | 同上 | 每行一条 evidence |
| `frozen_family.json` | freeze-family | 冻结候选族与 manifest hash |
| `confirmation.json` | confirm | 确认结果与确认折 |
| `frozen_manifest.json` | finalize | 终评前冻结的完整清单 |
| `final_evaluation.json` | finalize | 终评结果、账面 access_id、exploratory 标记 |
| `trials/<phase>/<fold>/<candidate>/<variant>/seed_<n>/<hash>/` | 每个试次 | `result.json`、worker 日志/任务 |
| `probe/<candidate>/...` | 每个探针 | 探针中间产物 |
| `bundle/`（`export` 输出） | export | 证据副本、预测、`checksums.sha256`、`bundle_manifest.json` |

---

## 4. 核心模块逐个讲

### 4.1 `core/schema.py` — 数据契约

重点模型：

- `FoldSpec`（[schema.py:217](../predicatelab/core/schema.py#L217)）：一折的 train/valid/test
  日期、`role`（search / confirmation / holdout）、`purge_days`、`embargo_days`、`metadata`。
  非时间折用 `metadata` 存索引（tabular 就是这样做）。
- `ExperimentProtocol`（[schema.py:249](../predicatelab/core/schema.py#L249)）：主指标、方向、
  发现折、确认折、holdout、baseline/control、seeds、model_spec、data_config、
  确认类型。`content_hash` 只包含影响科学结果的字段。
- `SearchPolicy`（[schema.py:296](../predicatelab/core/schema.py#L296)）：筛选取 top-k、
  显著性阈值、FDR、bootstrap 参数、seed、是否保存预测。
- `TrialRecord` / `EvidenceRecord` / `RoundRecord` / `FinalEvaluation`：持久化记录；
  每个都有稳定的 `ensure_id()`，重试不会制造重复行。

### 4.2 `core/adapter.py` — 领域接口

`DomainAdapter`（[adapter.py:86](../predicatelab/core/adapter.py#L86)）必须实现：

| 方法 | 作用 |
|---|---|
| `data_contract(config)` | 数据语义、标签定义、风险说明 |
| `snapshot(config)` | 不可变数据身份 |
| `suitability(snapshot, config)` | 数据是否适合该协议（ok/warn/block） |
| `trading_calendar(config)` | 真实交易日；有未来标签依赖时必须提供 |
| `build_protocol(config)` | 折、holdout、指标、seeds |
| `evaluation` | `EvaluationSpec` |
| `validate_descriptor(kind, descriptor)` | 静态校验，拒绝一切编译器不接受的输入 |
| `probe(descriptor, fold, snapshot, config, workdir)` | 只用训练折的数值冒烟测试，返回 `signature` 供行为去重 |
| `make_generator(config, policy)` | 候选生成器 |
| `run_trial(request)` | 执行一个（候选/基线/对照, 折, seed）试次 |

`run_trial` 返回约定：

```python
{
  "status": "success" | "failed",
  "metrics": {metric: value},
  "metric_series": {metric: {unit_key: value}},   # 配对统计的核心
  "fitted_params": {...},                          # 折内拟合参数
  "fit_data_hash": "...",                          # 拟合用的行身份
  "predictions_path": "...",
  "duration_s": 1.2,
  "error": "",
  "worker": {...},
}
```

`trial_config_hash`（[adapter.py:160](../predicatelab/core/adapter.py#L160)）决定缓存身份，
包含 domain、variant、control、descriptor、折、phase、seed、训练预算、模型规格、
snapshot/data manifest、标签表达式/偏移、预处理版本、代码 fingerprint/commit、
环境指纹。**故意不包含 `attempt`**：同一科学配置的重试必须命中同一缓存键。

### 4.3 `core/engine.py` — 生命周期与统计调度

建议按以下顺序读：

1. `_build_context`（209）——身份与协议；
2. `_trial_config`（396）与 `run_trial`（439）——试次执行与缓存；
3. `_run_round` / `_run_round_body`（714 / 776）——发现主循环；
4. `_seed_aggregate` / `_aggregate`（585 / 620）——跨折/跨 seed 聚合；
5. `_classify`（1104）——何时算 validated/rejected/insufficient；
6. `freeze_family` / `confirm` / `finalize`（1233 / 1319 / 1472）；
7. `_write_report` / `export`（1755 / 1861）。

统计口径（与 `core/stats.py` 对应）：

- 每个折、每个 seed：候选与 baseline 按共同 unit key 配对，得到差值序列；
- `effect = mean(candidate - baseline)`；
- 标准误用 Newey-West HAC（处理日度 IC 序列自相关）；
- 置信区间用 moving-block bootstrap；
- 跨 seed：effect 取均值，CI 取并集（min low / max high），p 取最差（max）；
- 跨折：`stability` = 效应方向与预期一致的折占比；
- `fold_coverage` = 实际成功折数 / 期望折数；
- `seed_agreement` = 方向一致 seed 占比；
- 每轮发现族内做 Benjamini-Hochberg；确认族单独做 BH。

分类门槛（[engine.py:1104](../predicatelab/core/engine.py#L1104)）：

```
coverage < min_fold_coverage           → insufficient
n_folds < min_folds_for_validation     → insufficient
seed_agreement < min_seed_agreement    → insufficient
无 p 值 / 不可推断                     → insufficient

direction * effect >= min_effect
  且 stability >= min_stability
  且 p <= fdr_alpha                    → validated

direction * effect <= -min_effect
  且 p <= fdr_alpha                    → rejected
否则                                   → insufficient
```

注意：这里的 `p <= fdr_alpha` 是**原始 p 值**门槛，随后 BH 再把不通过的 `validated`
降级。最终看 `q_value` 更稳妥。

### 4.4 `core/store.py` — SQLite 状态

[store.py](../predicatelab/core/store.py) 是一个带迁移的状态存储。表：

| 表 | 作用 |
|---|---|
| `tasks` / `snapshots` | 研究任务、数据快照 |
| `templates` / `candidates` | 三层身份的前两层 |
| `fitted_instances` | 折内拟合参数 |
| `runs` / `rounds` | 运行/轮次状态机 |
| `trials` / `trial_attempts` | 规范试次 + 每次尝试的不可变记录 |
| `evidence` | 每条候选每次阶段的证据 |
| `feedback` | 每轮给下一轮生成器的反馈 |
| `final_evals` | 终评 |
| `frozen_families` | 冻结候选族 |
| `holdout_access` / `holdout_access_events` | 全局留出集账本及事件 |

关键机制：

- `record_trial` 用 `ON CONFLICT` 更新规范行；**成功行不会被失败行覆盖**；
- `get_cached_trial` 校验结果哈希/指标序列/预测哈希，损坏行标记 `corrupt`；
- `trial_attempts` 记录每次尝试（包括 running 状态），便于崩溃审计；
- 迁移是加列/加索引式的（`_ADDED_COLUMNS`），历史行标 `legacy_*`，绝不把历史结果
  改写成“已确认”；
- 当前 `SCHEMA_VERSION = 3`，迁移历史见 [store.py:43](../predicatelab/core/store.py#L43)。

### 4.5 `core/protocol.py` — 折协议与访问控制

- `effective_time_fold`（[protocol.py:41](../predicatelab/core/protocol.py#L41)）按真实交易日
  对训练尾部 purge、验证头部 embargo；
- `validate_protocol`（[protocol.py:70](../predicatelab/core/protocol.py#L70)）拒绝：
  - 重复 fold_id、角色不对；
  - `purge_days < 1`，或小于标签未来依赖；
  - `train_end >= valid_start`；
  - 验证期互相重叠；
  - 确认折不晚于所有发现折；
  - 发现/确认折伸进 holdout；
- `HoldoutGuard`（[protocol.py:164](../predicatelab/core/protocol.py#L164)）按 purpose
  （search / confirmation / finalize）限制可读折；确认折需要已冻结 manifest，holdout
  需要已冻结 manifest。

### 4.6 `core/label_window.py` — 标签真正什么时候可用

这是 V1.1 最容易被忽略但很重要的模块。

- `parse_future_offset` 解析 qlib 表达式里最大的 `Ref(..., -N)`；
- `resolve_future_offset` 与数据契约里的 `label_window.uses_future_days` 交叉校验，
  不一致直接报错；
- `LabelCalendar.availability(date)` 用真实交易日历算“样本标签在哪天可用”；
- `isolation_report` 裁剪：
  - 训练尾：标签必须在 `valid_start` 之前可用；
  - 验证尾：标签必须在 `valid_end` 之前可用；
  - holdout test 尾：未来价格不存在的样本丢弃并记录 `tail_labels_unavailable`；
- `count_label_leak_samples` 可以按 instrument 文件数出具体泄漏样本数（审计用）。

所以配置里的日期是**名义日期**，真正执行用的是 `effective_folds`（在 `context.json`
和 `validate-config` 输出里可见）。

### 4.7 `core/holdout.py` — 一次性留出集

- `claim_holdout` 在 `BEGIN IMMEDIATE` 中查重并插入，防并发双吃；
- 同 key 的活跃行只允许一条（`ux_holdout_active` 部分唯一索引）；
- 完全相同的 run/candidate/manifest 重试 → `retry=True`，不新开测试；
- 同数据集、日期窗口重叠的另一个 study → `HoldoutConsumed`；
- `record_exploratory_reuse` 记录“明知已消耗仍复用”的行为，状态永远不会是
  confirmatory。

### 4.8 `core/identity.py` — 让缓存可追溯

- `code_version`：commit + `git status --porcelain` + `git diff HEAD` 的哈希；
  **任何未提交修改都会改变 fingerprint**，从而让所有试次缓存失效；
- `environment_versions`：python/platform/numpy/pandas/pydantic/sklearn/torch/qlib/scipy；
- `full_content_manifest`：逐文件 SHA-256 + Merkle root，带可自校验缓存；
  缓存只有在 size/mtime 仍匹配时才复用，`verify=True` 强制全量重算；
- `sanitise_json`：NaN/Inf → `null` 并记录路径；
- `atomic_write_text`：同目录临时文件 + fsync + `os.replace`，崩溃不留半文件。

### 4.9 `core/stats.py` — 统计口径

- `newey_west_se`：Bartlett 核 HAC 标准误；
- `block_bootstrap_ci`：移动块 bootstrap，保留短期相关；
- `paired_effect`：只取候选与基线共同 unit key 的差值；
- `benjamini_hochberg`：输入顺序返回 q 值，NaN 保留；
- `pool_fold_series`：跨折池化；同一 unit key 重复会报错（协议 bug 不能静默覆盖）；
- `stability`：效应符号与预期一致的折占比。

全部函数给定 seed 后确定性。

### 4.10 `domains/tabular` — 架构验证用适配器

[tabular/adapter.py](../predicatelab/domains/tabular/adapter.py)：

- 数据：`sklearn.datasets.load_breast_cancer`（569×30）；
- 固定 80%/20% 分层切分；20% holdout 只通过 finalize 访问；
- 发现折：80% pool 上的重复分层 5 折 CV；前 `discovery_repeats` 轮为 search，
  其余轮为 resampling confirmation；
- 模型：`StandardScaler + LogisticRegression`；
- 候选谓词作为额外一列拼到特征上；
- `metric_series` 的 unit key 是 `cv_key`（如 `r0_f3`），不是日期；
- `fitted_params` 记录 ratio 的分母尺度、sigmoid 的中位数/IQR、threshold 的阈值等。

这个适配器是“核心不依赖 TSIL”的证明，也是写新域适配器时最好的参照。

### 4.11 `domains/tsil` — 真实时序场景

组成：

| 文件 | 作用 |
|---|---|
| `adapter.py` | qlib 数据契约、全量内容 manifest、折协议、探针、子进程试次调度 |
| `dsl.py` | 受信任的 descriptor 编译器：7 个 DSL family + 23 个原生谓词 + 组合 |
| `seeds.py` | 第一轮种子谓词（paper6 / exploratory / DSL） |
| `generator.py` | 规则变异生成器 + `llm_json` 外部提案生成器 |
| `worker.py` | 唯一 import qlib/torch 的进程：建数据集、注入 `phi_fn`、训练、算指标 |

执行链：

```
engine
  → TSILAdapter.run_trial
    → 写 job.json
    → subprocess: python -m predicatelab.domains.tsil.worker job.json
      → qlib.init
      → MTSDatasetH(...)
      → QniverseModel(phi_fn=compile_descriptor(...))
      → fit → 逐日横截面 IC/RankIC/MSE/MAE
      → 原子写 result.json
    → 读取 result.json，校验返回码/有限性
```

安全模型：descriptor 是 JSON 数据，`dsl.compile_descriptor` 只映射白名单 family/
参数到本地纯 torch 函数；**从不 `eval`/`exec`**。LLM 只能写提案 JSON，仍要过静态
校验、探针和编译器。

原生谓词分派（[dsl.py:304](../predicatelab/domains/tsil/dsl.py#L304)）：

- `phi_timestamp` → `src.temporal_predicates`，必须提供 timestamp 特征；
- 无参数的 `phi_cycle/phase/momentum/volatility/return_dist` → temporal_predicates；
- 其他 → `src.custom_phi_constructors.get_custom_I_P`。

---

## 5. 配置详解

一个配置的顶层结构：

```yaml
run:        # 域、目录、数据库、轮数、是否要求确认、warning 确认
task:       # 研究问题、标题、假设（进入 ResearchTask 身份）
seeds:      # 全局 seed 列表（不是 search.seed）
data:       # 领域数据参数（TSIL 的 qlib 路径、market、label...）
model:      # TSIL 模型结构
training:   # 基础训练预算 + screen/full 覆盖
protocol:   # 折、holdout、purge/embargo、confirmation_kind
evaluation: # 主指标、方向、次要指标
search:     # 生成器、筛选/验证折、top-k、显著性阈值、bootstrap
execution:  # workers、GPU、超时
```

### 5.1 `run`

| 字段 | 说明 |
|---|---|
| `domain` | `tsil` / `tabular` / `mineral` / `image` |
| `artifacts_dir` | 产物根目录 |
| `db` | SQLite 路径 |
| `rounds` | discover 默认轮数 |
| `require_confirmation` | finalize 是否必须已独立确认，默认 true |
| `warning_acknowledgements` | 适用性 warning 的哈希列表，或 `true`/`"all"` |
| `warning_ack_reason` | 接受 warning 的书面理由（必填） |

### 5.2 `search` 阈值

| 字段 | 影响 |
|---|---|
| `screen_folds` / `validation_folds` | 必须是发现折 ID；默认 screen 取最后一个发现折，validation 取全部 |
| `max_candidates_per_round` | 每轮生成上限 |
| `full_validation_top_k` | screen 晋级名额 |
| `min_screen_effect` | screen 晋级的最小方向化效应 |
| `min_effect` | 完整验证的效应门槛 |
| `min_stability` | 折符号一致率门槛 |
| `fdr_alpha` | 分类的原始 p 门槛 + BH 的 alpha |
| `min_fold_coverage` | 成功折覆盖率门槛（默认 0.75） |
| `min_seed_agreement` | seed 方向一致率门槛（默认 0.5） |
| `bootstrap_samples` / `bootstrap_block` | CI 精度与块长（时间序列要 >1） |
| `nw_lag` | Newey-West 滞后阶数 |
| `behavioural_dedup_corr` | 探针签名的行为去重阈值（默认 0.995） |
| `min_folds_for_validation` | 完整验证最少折数（默认 1） |
| `save_predictions` | 是否让 adapter 落预测文件 |

`generator: rule` 使用确定性规则；`generator: llm_json` 需要
`generator_config.proposals_path`，文件内容只当数据。

### 5.3 `training` 的 profile 合并规则

[engine._training_config()](../predicatelab/core/engine.py#L375)：

```python
base = config["training"] 去掉键 "screen" 和 "full"
profile_key = "full" if phase == "final" else phase
merged = {**base, **training[profile_key]}
```

phase 实际取值：`screen`、`full`、`confirmation`、`final`。

- `screen` 用 `training.screen`；
- 完整验证用 `training.full`；
- finalize 也用 `training.full`；
- 确认阶段用 `training.confirmation`（没写就只用 base）。

这是 TSIL 省钱的主要开关：筛选 3 epoch、完整验证 8/50 epoch。

### 5.4 `protocol`

```yaml
protocol:
  purge_days: 2          # 训练尾丢弃的交易日数，必须 >= 标签未来依赖
  embargo_days: 3        # 验证头丢弃的交易日数
  folds: [...]           # role=search
  confirmation_folds: [...]  # role=confirmation，必须晚于所有发现折
  confirmation_kind: time_forward | resampling
  holdout:
    train/valid/test ... # 只通过 finalize 访问
```

TSIL 的 `_fold` 会先调用 `effective_time_fold` 应用 purge/embargo，再由 engine 按标签
可用性裁剪。**配置里写的是名义日期，`context.json.effective_protocol` 才是实际日期。**

### 5.5 标签改动

TSIL 的标签依赖在两处声明：

1. `data.label` 表达式，例如 `Ref($close, -2) / Ref($close, -1) - 1`；
2. `TSILAdapter.data_contract` 里的 `label_window.uses_future_days` / 
   `label_dependency.future_offset_trading_days`（当前硬编码为 2）。

如果你把标签改成 `-5`，只改 `data.label` 会在 `resolve_future_offset` 处因“表达式
声明 5、契约声明 2”而失败。正确做法是同时把契约里的 2 改成 5，并保证
`purge_days >= 5`。这是刻意的 fail-closed 设计，不是 bug。

---

## 6. 结果怎么读

### 6.1 先看身份，再看数字

```
context.json:
  identity_hash        —— 任务/协议/快照/标签/有效折/代码/环境/model 的综合身份
  protocol_hash        —— 协议内容哈希
  snapshot.manifest_hash
  data_manifest_hash   —— 全量内容 Merkle root（TSIL）
  label_expression / label_offset_trading_days
  label_isolation      —— 每个折被裁剪了多少
  effective_protocol   —— 实际使用的折
```

如果两次运行 `identity_hash` 不同，数字不可直接横向比较。

### 6.2 `status` vs `state` vs `stage`

- `stage`：`discovery` / `confirmation` / `final` / `legacy`；
- `status`：这次执行的机械结果；
- `state`：允许声称的科学结论。

报告按这三个维度分别计数。看到 `validated` 先确认 `stage=discovery` 且
`state=exploratory_positive`；看到 `independently_confirmed` 才是独立确认。

### 6.3 统计字段

| 字段 | 读法 |
|---|---|
| `effect` | 候选减去基线的主指标均值差（方向由协议决定） |
| `ci_low/high` | block bootstrap 区间；`null` 表示不可估计 |
| `p_value` | Newey-West HAC 下的双侧 p；单折/单点常为 `null` |
| `q_value` | 该阶段族内 BH 校正后的 q |
| `stability` | 折效应符号一致率 |
| `fold_coverage` | 成功折占比 |
| `seed_agreement` | seed 方向一致率 |
| `null_paths` | 被替换成 null 的字段路径；不是 0，是“无法估计” |

### 6.4 数据库怎么查

```bash
sqlite3 artifacts/predicatelab.sqlite \
  "SELECT stage,state,status,count(*) FROM evidence GROUP BY 1,2,3;"
sqlite3 artifacts/predicatelab.sqlite \
  "SELECT candidate_id,status,effect,p_value,q_value,stability FROM evidence
   WHERE run_id='run_...' ORDER BY created_at;"
```

或：

```bash
$PY -m predicatelab.cli evidence --config configs/tsil_lstm_smoke.yaml --run-id run_...
$PY -m predicatelab.cli holdout-audit --db artifacts/predicatelab.sqlite
```

---

## 7. 二次开发手册

### 7.1 加一个 TSIL DSL 谓词族

以新增 `range_position`（窗口内价格分位）为例：

1. 在 [dsl.py](../predicatelab/domains/tsil/dsl.py) 的 `OP_PARAMS` 加：
   ```python
   "range_position": {"window": "int"},
   ```
2. 在 `raw_phi` 里加分支，返回 `[B, d]` 的未归一化描述子（必须是纯 torch、
   有界、无副作用）：
   ```python
   if family == "range_position":
       lo = xw.amin(dim=1)
       hi = xw.amax(dim=1)
       pos = (xw[:, -1, :] - lo) / (hi - lo).clamp_min(1e-6)
       return torch.stack([pos.mean(dim=1), pos.std(dim=1)], dim=1)
   ```
3. 静态校验自动生效（`window` 必填且范围受 `seq_len` 限制）；如果参数更多，
   在 `validate_descriptor` 里补类型/范围检查。
4. 在 [generator.py](../predicatelab/domains/tsil/generator.py) 的 `_exploration` /
   `_mutate_one` 里加入该 family；或在 [seeds.py](../predicatelab/domains/tsil/seeds.py)
   的 DSL_SEEDS 里加种子。
5. 在 `tests/test_tsil_dsl.py` 加：合法 descriptor 通过、非法参数被拒、输出形状/
   有限性正确。

**不要**在 descriptor 里放代码字符串；不要把 family 参数映射到动态 import。

### 7.2 加一个原生谓词

1. 在 `NATIVE_PREDICATES`（[dsl.py:31](../predicatelab/domains/tsil/dsl.py#L31)）注册名称、
   参数 schema、是否需要 timestamp；
2. 确认上游 `TSIL-main/src/temporal_predicates.py` 或
   `custom_phi_constructors.py` 存在对应实现；
3. 在 `_native_I_P` 里决定分派路径：
   - 无参数 canonical 实现 → temporal_predicates；
   - 带参数实现 → custom_phi_constructors；
4. 如需要 timestamp，确保 `needs_timestamp` 返回 True，worker 的 dataset
   `phi_type=phi_timestamp` 才会带 timestamp；
5. 加测试。

### 7.3 改生成器

- 规则生成器：实现 `CandidateGenerator.propose(round_index, n, feedback, known_template_hashes)`；
- 生成器**只能**看到发现阶段 feedback；不要读取 confirmation/final 证据；
- 新生成器在 `make_generator` 注册；
- 生成器输出必须是 `PredicateTemplate` 数据，不能直接执行；
- 保证确定性（同输入同输出），否则“可复现”无从谈起。

### 7.4 改统计或阈值

- 统计算法改 [stats.py](../predicatelab/core/stats.py)；保持确定性、单测覆盖；
- 阈值改配置 `search.*`；若新增策略字段，加到 `SearchPolicy` 并在
  `_classify` / `_run_round_body` 使用；
- 改分类规则时，同时更新：
  - `docs/v1.1_evaluation_protocol.md`；
  - `tests/test_stats.py` / 端到端测试；
  - 报告里的 reason 文案（`_reason_for`）。

### 7.5 新增一个可执行领域适配器

以 `foo` 为例：

1. 新建 `predicatelab/domains/foo/adapter.py`；
2. 实现 `DomainAdapter` 的全部抽象方法（可参考 tabular）；
3. 数据契约必须写清：实体键、时间键、标签窗口/未来依赖、特征语义、已知风险；
4. `snapshot` 必须给出稳定、可验证的数据身份；
5. `suitability` 在数据不支持协议时返回 `block`；
6. `validate_descriptor` 必须是封闭白名单；`run_trial` 里的任何拟合只用该折训练数据；
7. 在 [domains/__init__.py](../predicatelab/domains/__init__.py) 的 `create_adapter`
   注册，并把名称加入 `executable_domains()`（只有真的能跑才加）；
8. 在 `tests/` 加：descriptor 拒绝、折隔离、一个端到端 round；
9. 在配置里 `run.domain: foo`。

最小骨架：

```python
from predicatelab.core.adapter import CandidateGenerator, DomainAdapter, TrialRequest
from predicatelab.core.schema import (DataContract, DataSnapshot, EvaluationSpec,
                                      ExperimentProtocol, FoldSpec)

class FooAdapter(DomainAdapter):
    name = "foo"

    def data_contract(self, config): ...
    def snapshot(self, config): ...
    def build_protocol(self, config): ...
    @property
    def evaluation(self): ...
    def validate_descriptor(self, kind, descriptor): ...
    def probe(self, descriptor, fold, snapshot, config, workdir): ...
    def make_generator(self, config, policy): ...
    def run_trial(self, request: TrialRequest): ...
```

### 7.6 改折或标签

- 时间序列：holdout 永远放 `protocol.holdout_fold`，**不要**放进 `folds`；
- 确认折必须晚于所有发现折的 `valid_end`；
- `purge_days >= 标签未来交易日数`；
- 分组数据（图像/空间）：按 scene/patient/block 分组切分，不能随机切相邻样本；
- 改完跑 `validate-config`，检查 `effective_folds` 与 `label_isolation`；
- 跑 `tests/test_adversarial_label_isolation.py` 与 `test_protocol_guard.py`。

### 7.7 改数据库 schema

[store.py](../predicatelab/core/store.py) 的迁移是“只增不减”的：

1. 新表/新索引：加到 `_SCHEMA`（`CREATE TABLE IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS`）；
2. 新列：同时加到 `_SCHEMA` 和 `_ADDED_COLUMNS`，旧库靠 `ALTER TABLE ADD COLUMN`；
3. 提升 `SCHEMA_VERSION`，在 `MIGRATION_HISTORY` 加说明；
4. 历史行缺字段一律标 `legacy_unknown`，**不要**猜测确认状态；
5. 破坏性变更（改类型/删列/改唯一约束）需要显式迁移函数，不能只改 `_SCHEMA`；
6. 加迁移测试（参考 `tests/test_migration.py`），验证旧库可升级且行数不丢。

注意：`_ADDED_COLUMNS` 只在 `_migrate()` 里执行；`_SCHEMA` 对已有表不会补列。
两处必须同步。

### 7.8 改 CLI / 导出

- CLI 只做参数解析与调用，不放科学逻辑；
- 新命令在 `build_parser()` 注册，处理函数返回 0/非 0；
- 导出新文件时，同时更新 `export()` 的文件列表、`checksums.sha256` 和
  `bundle_manifest.json`，保证 bundle 可校验。

### 7.9 测试怎么写

现有测试分三类：

| 类型 | 文件 | 目的 |
|---|---|---|
| 单元 | `test_stats.py`、`test_schema_identity.py`、`test_tsil_dsl.py` | 统计/哈希/DSL 语法 |
| 状态与迁移 | `test_store_resume.py`、`test_migration.py` | 缓存、恢复、旧库升级 |
| 对抗性协议 | `test_adversarial_*.py` | 试次身份、holdout、标签隔离、worker 失败、导出 |
| 端到端 | `test_engine_tabular.py` | 完整 discover → freeze → confirm → finalize |

运行：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. \
  /home/dmz-ai/miniconda3/envs/zzr_py312/bin/python -m pytest -q
```

新增功能至少覆盖：正常路径、拒绝路径（非法 descriptor/越权折/坏缓存）、
以及“失败不会变成成功”。

---

## 8. 调试与排错

| 症状 | 原因 | 处理 |
|---|---|---|
| `data suitability ... unacknowledged warnings` | 适用性检查有 warning | 把打印的哈希加到 `run.warning_acknowledgements`，并写 `run.warning_ack_reason` |
| `purge_days=... is not allowed` | 时间折没有 purge 或小于标签依赖 | 提高 `protocol.purge_days` |
| `protocol ... exceeds label ... future trading-day dependency` | purge 小于标签偏移 | 同上，按 `label_offset_trading_days` 设置 |
| `run ... protocol ... current config hashes ... refusing to mix` | 改了配置却复用 run_id | 换新 run_id，或恢复配置 |
| `snapshot changed ... cached baselines are invalid` | 数据内容/元数据变了 | 换新 run_id；旧缓存不可信 |
| `holdout ... already consumed` | 该数据窗口已被别的 study 终评 | 不能再做 confirmatory final；只能显式 exploratory reuse |
| `candidate ... is not validated` / `not in the frozen family` | finalize 前置不满足 | 先完成 discover、freeze-family；必要时确认 |
| `candidate ... is not independently confirmed` | 默认要求确认 | 先 `confirm`；或设 `run.require_confirmation: false` 做 exploratory final |
| 改了代码后所有试次都重跑 | `code_version` 把 dirty diff 纳入身份 | 这是设计；提交代码或接受重算 |
| 缓存没命中 | data/label/fold/training/model/环境变化 | 对比 `context.json` 与 `trials/*/config_hash` |
| TSIL worker 失败 | 看 `worker.log` 与 `result.json` | 常见：qlib 数据路径、CUDA、OOM、标签无有效样本 |
| `constant predicate` | 探针在训练折上方差为 0 | 换参数/特征子集；该候选不可用 |
| 指标是 `null` | 统计量不可估计（单折/单日/退化） | 看 `null_paths`，不要当 0 |
| `validate-config` 很慢 | TSIL 每次对全量数据做内容哈希，且临时目录不复用缓存 | 属正常开销；可参考第 9 节改进 |

---

## 9. 已知限制与建议优先修改项

以下是阅读代码时能明确看到的、适合后续迭代的点（按影响排序）：

1. **`freeze_family(top_k=...)` 的排序没有考虑指标方向**
   [engine.py:1265](../predicatelab/core/engine.py#L1265) 用 `-(effect)` 排序。对
   `direction="lower"`（如 tabular 的 log_loss）会优先选到最差的候选。当前默认
   `top_k=None` 时选全部 positive，不受影响；一旦用 `--top-k` 就会暴露。
   建议按 `direction * effect` 排序（并优先用 q 值/稳定性），并加回归测试。

2. **冻结族在确认/终评后仍可被重新冻结**
   `freeze_family` 只检查 run 状态，不检查是否已经 confirm/finalize。重新冻结会
   改变 `manifest_hash`，而确认证据并不绑定族哈希。建议：一旦存在确认证据或
   final_evals，就禁止重新冻结，或把 `family manifest_hash` 写入确认证据并校验。

3. **`execution.workers` 目前没有真正并行**
   `_max_workers()` 被定义了但未使用；试次是串行执行的。需要并行时，建议在
   `run_trial` 调度层加线程/进程池，同时注意 SQLite 写入串行化与 GPU 显存。

4. **`validate-config` 每次都对 TSIL 全量数据重新哈希**
   [cli.py:48](../predicatelab/cli.py#L48) 把 `artifacts_dir` 指到临时目录，导致
   `_manifests` 缓存不可复用。可增加 `run.manifest_cache_dir`，或让 validate 复用
   正常 artifacts 下的缓存。

5. **TSIL 标签未来依赖在契约里硬编码为 2**
   [tsil/adapter.py:90](../predicatelab/domains/tsil/adapter.py#L90)。改标签时容易只改
   `data.label`。建议从表达式解析后写入契约，或在适配器初始化时交叉校验并给出
   更明确的错误。

6. **`__version__` 仍是 `1.0.0`**
   [predicatelab/\_\_init\_\_.py](../predicatelab/__init__.py) 与部分 docstring 仍提
   V1.0。建议升到 `1.1.x` 并统一术语。

7. **`freeze_family` 的 multiple-testing 假设**
   冻结清单把“所有发现阳性”作为 BH 族；如果显式只选一部分候选，族定义会与
   `policy.fdr_alpha` 的解释不完全一致。建议在清单里区分“发现族”和“进入确认的
   子族”，或明确记录选择规则对 FDR 的影响。

8. **TSIL probe 与 trial 使用不同 batch**
   探针用 `batch_size=64` 的单个训练 batch，trial 用训练配置的 batch；探针只做
   数值冒烟/行为去重，不参与统计，因此可接受，但行为去重的阈值应理解为
   “近似”而非严格等价。

9. **历史结果不能重算**
   V1.0 的 TSIL holdout 已被 V1.1 账本标记为 consumed；这是不可逆的科研记录，
   不是代码缺陷。

---

## 10. 推荐阅读顺序

第一次读代码：

1. 本文第 0–3 节（建立全局图）；
2. [core/schema.py](../predicatelab/core/schema.py)（先认识数据模型）；
3. [core/adapter.py](../predicatelab/core/adapter.py)（领域边界）；
4. [core/engine.py](../predicatelab/core/engine.py) 的 `_build_context` → `run_trial` →
   `_run_round_body` → `freeze_family` → `confirm` → `finalize`；
5. [core/store.py](../predicatelab/core/store.py)（状态如何落库）；
6. [domains/tabular/adapter.py](../predicatelab/domains/tabular/adapter.py)（最小可执行域）；
7. [domains/tsil/adapter.py](../predicatelab/domains/tsil/adapter.py) +
   [dsl.py](../predicatelab/domains/tsil/dsl.py) + [worker.py](../predicatelab/domains/tsil/worker.py)
   （真实场景与子进程边界）；
8. `tests/test_engine_tabular.py`（一条完整生命周期的可执行说明）；
9. [V1.1 评估协议](v1.1_evaluation_protocol.md)（规范层面为什么这样设计）。

要改代码时，先回答三个问题：

- 这个改动属于 **核心协议**、**领域实现**，还是 **纯展示/工具**？
- 它会不会改变 `protocol_hash` / `config_hash` / `context_hash` / `holdout_key`？
- 它会不会让“失败/不确定”被误读成“确认”？

如果第三个问题的答案是“可能”，先停下来改设计，再写代码。
