# PredicateLab：Coding Agent 增量开发与科学验证总提示词（2026-11-06）

## 0. 你的角色、任务和交付标准

你是同时具备统计学习理论/LUSI、时间序列机器学习、自动科学发现、统计推断、Agent/Harness 与科研软件工程能力的高级开发与独立审计团队。工作对象是**现有 PredicateLab 仓库**，不是从零搭建演示项目。

目标：在兼容已有 V1.1 实现和全部历史证据的前提下，把 PredicateLab 发展成**自动化科学假设与谓词发现、反证、独立验证、知识积累和有界自主研究系统**。研究首个完整场景是 TSIL 时间序列预测；Tabular 是跨领域接口与隔离的工程对照；找矿预测和图像分类目前只保留清晰的 DomainAdapter / DataContract / PredicateSpec 扩展契约，不得冒称已实现领域能力。

必须执行代码审计、必要的增量修改、可运行测试、真实或明确受阻的集成检验、证据导出和独立验收报告。**仅输出设计文档、伪代码、模拟数据的漂亮截图或宣称功能完成，不算交付。**

判断成功的核心不是发现尽可能多的正效果，而是：
- 发现过程能够提出合法、非重复、可证伪的候选假设与谓词；
- 能够自动识别无效、等价、泄漏、退化和仅由通用正则化解释的候选；
- 能保护未暴露数据，保留阴性结论，按预注册协议升级科学证据；
- 可以由独立脚本在相同输入下重算关键结论；
- 在有限预算与明确权限下持续推进，而不依赖人逐轮盯守。

## 1. 开发前必须承认的事实和硬边界

### 1.1 理论与科学事实

Vapnik 与 Izmailov 的 LUSI 理论是理论起点，不是“任何自动生成谓词都能提升预测”的保证。对平方损失下的 Bayes 回归函数以及预测时可测、可积的 `phi(Z)`，`E[phi(Z)(Y-f*(Z))]=0` 不等于该谓词具有独立预测价值。必须区分：新的信息、新的归纳偏置、通用正则化、样本权重变化、方差下降、优化预算优势。

历史 TSIL 研究已有正反结果：PF 硬投影、Parameter-space、Learned Support、TSIL-Multi 都存在机制边界或缺乏稳定专属增益；不得把已有被否定/待验证的机制重新包装为已确立优势。历史 PredicateLab V1.0 最佳探索 IC 增益约 `+0.00378`，最终 holdout 增益 `-0.001026`、`p≈0.3861`，**未支持样本外增益**。保留此阴性结论。

### 1.2 代码与数据基线

- 优先在实际仓库核对 `Predicate_Gen_v1.1_fixed_20261102_061339.tar.gz` 对应版本与工作树。历史指南所述结构：`predicatelab/core/`、`predicatelab/domains/tsil/`、`predicatelab/domains/tabular/`、`TSIL-main/`、`tests/`、`configs/`、`docs/`。若本地仓库不同，以**真实文件审查结果**为准并记录差异，不得臆造存在的接口。
- V1.1 已有 `Engine / DomainAdapter / PredicateTemplate / BoundCandidate / FittedInstance / Trial / Evidence / SQLite / holdout ledger / CLI`；应复用并扩展。既有 TSIL `phi_fn → Φ/P → WeightedMSELoss → QniverseModel.fit` 必须保持真实调用链，不能替换为特征拼接或 mock。
- 既有 TSIL 主配置使用固定 `tau=0.5`；早停验证从训练内部切分，当前 fold 报告段应独立；Screen/Full 分折。核对**实际运行代码**是否满足，不能只看说明。
- 历史 V1.1 数据库曾迁移至 schema v3，并保留过 `12 runs / 17 rounds / 519 trials / 206 evidence / 2 final_evals / 15 feedback`。这些是历史基线记录，不能假定当前数据库计数不变；执行前对真实数据库备份、记录表计数与摘要，迁移后逐表核对，不做 `reset/clean`，不覆盖未提交文件，不删除原始证据。
- **真实 TSIL 2024-01-01 至 2025-08-31 final holdout 已消费；禁止本次重新读取该区间执行科研终评/重新作为独立确认；不得调用其真实 `finalize`。**历史 2 条终评与 `backfill_legacy` 记录必须保留。
- `f2023` 曾出现在历史搜索/反馈中，当前研究轨迹里可用于开发演练、机械验证与历史结果复现，必须标注 `historically_exposed / exploratory`，**不可作为新的独立确认**。新数据库、新 run、新名字、新日期别名不能改变历史暴露状态。
- 历史某些配置使用 `f2022` Screen、`f2020/f2021` Full、`f2023` Confirmation：它们不能自动解释为时间前向独立验证。新增科学协议应重新定义按时序可解释的滚动验证；历史运行照原样保留。
- 在缺失源码、数据、依赖、GPU 或访问权限时报告 `NOT_VERIFIED`/`BLOCKED_RESOURCE`，不能用 mock 冒充真实 TSIL，也不能虚构执行结果。

### 1.3 权限与安全

- 任何 Agent/Worker 不得修改历史数据库记录、已冻结实验协议、证据阈值、未暴露验证数据的访问控制和专家决定；变更必须产生新版本与新审批记录。
- LLM 只能提交经过 JSON Schema/Pydantic 验证的结构化假设或 DSL 描述符；禁止将生成文本直接 `eval/exec` 或任意导入执行。所有 Worker 用最小权限、固定资源配额与沙箱；不允许自动发布研究结论、上传敏感数据或对外发送消息。
- 修改 `TSIL-main` 原生训练逻辑必须附差分数值测试、行为回归和独立审核；优先通过 Adapter/新模块实现。
- 计算预算与数据访问权限先检查后执行。不得抢占其他用户 GPU、不得自行启动大规模训练；资源不足应可恢复地挂起，并记录阻塞原因。

## 2. 研究问题与形式化科学契约

任何研究运行必须由 `ResearchHypothesis` 和冻结的 `ExperimentProtocol` 驱动。不能仅以“搜索最优 IC”作为未定义的研究目标。

### 2.1 ResearchHypothesis 必须包含

`id/version`、研究问题、目标任务和数据范围、主张 `claim`、原假设/备择假设、对照基线、主要 estimand/metric、最小实际有意义效应、可用信息集合、谓词作用机制及替代解释、成立前提、预期适用边界、明确反证条件、关联参考与历史证据、提出者、创建时间、状态。

必须区分四类研究命题：**谓词表达是否正确、统计约束/知识是否可用、加入模型是否有增益、增益是否来自谓词特有机制**。四类命题分别验证，不能因一种通过推断另外三种通过。

### 2.2 ExperimentProtocol 必须冻结

数据快照与数据来源、时间可见性、样本/交易日对齐、标签依赖 horizon、完整开发/筛选/内部模型选择/确认/终评划分、purge 与 embargo、模型/骨干/超参、固定 τ 或其他损失机制、配对随机种子、主要和次要指标、统计单位、显著性/置信区间方法、多重比较范围和方法、负/正对照、最低效果/稳定性标准、失败处理、提前停止准则、预算上限、所需人工审批、代码与环境身份。

**禁止使用 Confirmation 标签执行早停、epoch/checkpoint 选择、参数拟合、搜索反馈；禁止用 Final 结果更新搜索。**对滚动标签窗口实施 point-in-time 检查，并输出实际日期与预测时点信息集。

### 2.3 人工研究 Gate（必须独立实现）

- `Gate A / Research Charter`：研究目标、主假设/反证、数据权利与历史暴露、开发期数据范围、主要指标、最大计算预算获批准后，方可启动新的研究家族。
- `Gate B / Freeze & Confirm`：候选族、搜索历史、协议/阈值、验证代码与数据访问权限锁定；只在存在真正未暴露且授权的数据时允许一次合规独立确认。**不能因为研究人员批准而把已暴露数据变成新的独立证据。**
- `Gate C / Knowledge & Release`：独立审核完整正反证据、适用边界、冲突解释和复现包后，批准知识等级升级或对外发布。发布权限与科研结果权限分开。

审批应是带 ID、时间、原始决策、权限范围及到期条件的可审计记录；人工介入只用于本质性科学与安全决定，不能让每个 Trial 都来找人。

## 3. 系统架构：科学控制面优先，Super-agent 只承担执行

沿用现有 Core/Adapter/Store，在其上增量建立三层逻辑：

1. **Research Control Plane**：确定性的生命周期/协议控制、访问权限、Research Router、任务分派、事件去重、预算与 Gate 管理；必要时用轻量 LLM 理解研究意图，但科研规则由代码强制。
2. **Isolated Research Workers**：候选生成、结构/语义探针、TSIL 真实 Trial、对照实验、反证、统计计算、独立复核。Worker 不直接调整验收标准、不直接向专家发散消息；Worker 与“专业 Agent”是不同概念，可以由多个角色在一个任务内部协作。
3. **Evidence & Knowledge Plane**：持久化数据库、内容寻址 artifacts、全局数据暴露账本、科学证据图谱、实验历史/否定知识、可重放证明、专家审批记录与报告。

只实现**逻辑上的** Router/Worker、异步队列和单实例 Watchdog 作为基础。默认单机队列/进程池、现有 SQLite 与原子文件写入；按压测证据再决定是否引入独立消息中间件、分布式数据库、常驻云端模型、A2A、多 Agent 框架或复杂前端。不要为复刻 Super-agent 而破坏科学控制面。

### 3.1 事件与任务规范

统一事件字段：`event_id, event_type, source, research_id, hypothesis_id, task_id, correlation_id, occurred_at, payload_ref, payload_hash, schema_version, priority`；事件只作为意图/状态变化，不作为无校验的代码指令。

至少支持：`run.created, candidate.proposed, trial.completed, trial.failed, evidence.updated, falsification.requested, protocol.freeze_requested, approval.decided, resource.available, watchdog.alert, knowledge.review_due`。

先用确定性 Fast Path 做权限、重复、资源、协议检查；已知归属的事件直接送关联 Worker，跨任务决策交 Research Router。必须有幂等键、原子 claim/lease、心跳、到期重试、任务取消、断点续跑、重复/乱序事件测试；禁止因重试产生重复 Trial/Evidence 或重复消费受保护数据。

### 3.2 双重 Watchdog

`ExecutionWatchdog`：检测存活、死循环、无进度、GPU OOM、超时、重试风暴、孤儿任务、存储损坏。

`ScientificWatchdog`：检测标签泄漏、历史数据暴露、指标/样本数不一致、阈值被修改、模型选择用错数据、fold/seed 缺失、负对照异常、统计上不合格却晋级等。ScientificWatchdog 应优先执行确定性守卫；LLM 分析仅提出调查建议，不能覆写守卫结论。

## 4. 核心数据模型与不可变证据

在已有对象上最小兼容扩展，优先增加：

- `ResearchHypothesis`：目标、零/备择、作用机制、反证、适用范围、假设版本。
- `FalsificationCase`：反证类型、目标替代解释、输入/对照、运行前预期、结果、结论。
- `DataExposure`：原始源身份/内容哈希、切片与标签覆盖、研究家族、研究者/执行者、访问目的、实际读写/反馈、时间、被污染状态、不可降级的暴露等级。
- `KnowledgeClaim`：精确定义、理论前提、适用域、正反证据、证据等级、冲突与失效条件、当前状态、版本与复核日期。
- `ExperimentProtocolVersion`、`DecisionRecord`、`ReplayCertificate`：冻结协议、专家决策原文/日期/范围、可重算证据与校验和。

继续维持 `PredicateTemplate → BoundCandidate → FittedInstance → Trial → Evidence` 的身份层次；`FittedInstance` 必须绑定 fold 与参数来源。提供迁移脚本，先备份再增量迁移，保存外键、历史身份、run/trial/evidence/final 记录；旧输出解析仍须兼容。不得把工程执行状态 `success` 或筛选结果 `validated` 默认为科学确认。

为运行和科学结论分设独立状态：
- 执行状态如 `PENDING/RUNNING/SUCCEEDED/FAILED/BLOCKED/CANCELLED`；
- 科学状态如 `UNTESTED/EXPLORATORY/REFUTED/INCONCLUSIVE/INDEPENDENTLY_CONFIRMED/EXTERNALLY_REPLICATED`。

科学状态升级必须由程序化 Evidence Policy 决定，依据完整证据链与访问记录；不得由 LLM、单一 p 值、放宽门槛测试或专家主观口头指令直接伪造。专家可以批准研究方向/冻结/发布，也须保留独立证据标准。

全局数据暴露账本必须由独立于单次 `run.db` 的权威注册表与受控评价入口管理（可先用独立受保护 SQLite/服务实现），在多 run、多库、新 dataset_id、文件重命名、镜像数据等情形下按稳定来源内容身份识别历史使用；无法确认同源时默认禁止独立确认并标明原因。保留访问拒绝日志及人为例外审批记录，不能暗中改为未暴露。

证据记录包括：源码 commit/脏工作树指纹、配置、依赖、硬件、数据哈希、时间区间、生成器及搜索策略版本、全部候选（含被淘汰）、所有种子/折、日志、预测 `(instrument,date,score,label)`、效果序列、失败与恢复、统计程序版本、结论与理由。导出 `manifest + checksums + machine-readable JSON/NPZ + human-readable report`；独立复核无需相信原先聚合器。

## 5. 完整研究闭环（必须真正连通）

```mermaid
flowchart LR
    problem["研究问题与原始证据"] --> hypothesis["假设与反证条件"]
    hypothesis --> protocol["冻结实验协议"]
    protocol --> proposal["结构化谓词提案"]
    proposal --> probe["语义验证与去重"]
    probe --> screen["低预算 Screen"]
    screen --> full["配对 Full 验证"]
    full --> falsify["机制反证与负对照"]
    falsify --> evidence["统计证据与独立重算"]
    evidence --> decision{"科学 Gate"}
    decision -->|"继续探索"| proposal
    decision -->|"冻结"| freeze["候选族冻结"]
    freeze --> confirm["未暴露数据独立确认"]
    confirm --> knowledge["知识/否定结论入库"]
    decision -->|"无效或资源耗尽"| knowledge
```

文字说明：`Screen/Full` 均是研究开发范围内的搜索与验证；只有事先隔离、未暴露并经授权的数据才可能进入独立 Confirmation；已暴露历史 TSIL 区间不得走新的 Confirm/Final。需要人工授权的节点可以挂起，不能自行越过。

对每一步实现运行契约、状态变更、异常处理、测试和落库。候选发现应覆盖：已知原生模板、受限 DSL 组合、逻辑/条件/软谓词、统计描述子、基于反馈的变异；可选以 PySR/Optuna/LLM JSON 提案做 **Generator 插件**，必须通过同一个安全编译、预算和科学评估接口。

搜索反馈只允许来自对应 Discovery 开发分区与明确可用的历史结果。记录**完整搜索族**，不仅记录 top-k。用多样性/复杂度约束与机制先验减小无效搜索空间，识别表达式等价、训练折参数污染、无效常数矩阵、对角化退化、异常高阶复杂度、future/label information leak。

**停止规则**：达到资源预算、发现无法优于复杂度匹配安慰剂、跨折/跨 seed 缺乏稳定性、关键机制反证失败、系统性数据质量缺陷、统计精度不足或长期无有效假设增量时应停止并生成负结果，而不是无限继续 mutation。

## 6. TSIL 科学实验设计（首个完整领域）

### 6.1 有比较意义的基线与反证

同一数据快照、标签、骨干、训练资源、配对 seeds/folds 下至少实现：

- `B0`：普通 MSE，无谓词。
- `B1`：经过历史使用的原生/人工 TSIL 谓词。
- `B2`：系统自动生成候选。
- `B3`：复杂度、矩阵谱/强度或训练代价尽可能匹配的随机/打乱谓词。
- `B4`：`Phi_ones`、常数、错误时点/错误条件谓词等负对照。
- `B5`：仅增加输入特征、一般加权损失、通用正则化/方差控制等替代机制（按假设选择可比项）。

建立“机制归因矩阵”：若 B2 仅胜 B0、不胜 B3/B5，只能报告探索性性能变化，不得宣称谓词独有的统计知识。对 PF、Parameter-space、Support、Multi 等既有阴性机制保留复现实验入口与历史结论，**不默认要求重新全量运行**。

### 6.2 时间与统计

- 优先使用有前向意义的 rolling-origin/nested time evaluation；按真实交易日和标签 horizon 进行 purge/embargo，标注每段精确日期与可见信息时点。
- 训练内 early stopping 与被报告/确认的折分开；所有预处理、归一化、谓词阈值和拟合参数只能见对应训练折。
- 预声明主要指标（TSIL 可用每日横截面 IC，具体以冻结协议为准），同时报告 MSE/RankIC 等次要指标；明确方向、统计单位、缺失日与股票池口径，核对成分股生存偏差、复权与行情修订、停牌、信息发布时点。
- 使用同一日期/种子配对差异；HAC 或合适的 block bootstrap 给置信区间，不能将股票×日期×seed 全部当作独立样本。对跨 regime 的时间稳定性、骨干稳定性、效果分布与负尾风险进行记录。
- 对完整候选搜索与阶段晋级保留选择历史，按预声明 family 做多重比较；可使用 BH/BY、White Reality Check、Hansen SPA、嵌套选择后独立确认等合适方法。**仅对晋级 top-k 做 BH 不能自动消除全部 adaptive-search bias。**若方法前提不满足，降低证据等级而非输出伪精确 p 值。
- 报告效果量及不确定性，与最低实际有意义增益比较；没有显著性或数据功效不足应输出 `INCONCLUSIVE`，而不是武断成功/失败；**未拒绝原假设不等于证明谓词无效**。

### 6.3 自动发现系统自身的 PredicateBench

创建小而可重复的**科学系统单元测试集**，至少包含：

1. 有隐藏时间/条件规律的正控（实现只知道可访问的训练信息，评估者另持真值）；可测试能否恢复已知规律及正确时点条件。
2. 完全无效或标签打乱的负控；在多次独立复制中评估虚假发现率，要求报告区间和样本数，不得只展示一次没发现。
3. 只有通用正则化有帮助、无额外谓词信息的对照；阻止误认新知识。
4. 不稳定分布、制度切换、错误信息时点、重叠标签、极端缺失和退化矩阵。
5. 规则表达等价、相同来源跨路径重命名、源数据同大小/mtime 但内容变化。

`PredicateBench` 应输出：真正例恢复、假阳性、复杂度/成本、跨分布鲁棒性、反证命中率、数据泄漏拦截率、可复现性与人工介入数量。最低门槛从小规模 smoke 开始，充分统计功效的版本在预算批准后执行；不允许把不足样本的校准结果称作 FDR 已证实。

## 7. 分阶段执行，不跳步

### Stage A — 只读事实审计与基线冻结

1. 找到实际仓库与上游代码，输出版本、`git status`、差异、文件/依赖/配置清单，列出哪些历史文档可核验，哪些缺失；先备份数据库和重要 artifacts。
2. 从原始论文、上游实现、原始数据/日志提出至少 5 个现有需求未覆盖但可证伪的系统性风险，给出对应反例测试；再追溯真实执行链 `CLI → Engine → Adapter → Worker → Dataset → QniverseModel → WeightedMSELoss → NPZ → Evidence`，逐项标注 `VERIFIED/CODE_ONLY/NOT_VERIFIED/FAILED`。
3. 独立检查已知高危点：f2023 历史暴露、early stopping/confirmation、全局 ledger 跨库、源数据哈希、freeze/confirm/finalize 状态、缓存与身份、失败折汇总、试次重试、预测日期、真实 tau 与 P 梯度。
4. 在隔离环境运行已有回归与低风险 Tabular demo；输出 `docs/research_baseline_20261106.md` 与缺陷优先级清单。若源码内容与历史指南冲突，保持事实证据优先。

### Stage B — 先修科学隔离和可靠性 P0

在保持现有 CLI 与历史结果兼容的前提下：修复或验证全局暴露/权限、训练验证边界、候选与试次身份、不可变协议、幂等恢复、源数据内容校验、科学与工程状态解耦、失败 trial 不得误晋级。新增数据库迁移、`pytest` 回归、属性/变形测试（可选 Python `hypothesis` 库）和攻击性测试；模型与环境非确定性应记录复现实验的容差及不可控因素。任何 P0 阻断存在时不进入新的真实科研确认。

### Stage C — 可证伪科学对象与独立统计

实现 Hypothesis/Protocol/Falsification/KnowledgeClaim、效果评估和机制对照、全搜索族记录、PredicateBench、独立 ReplayCertificate。先用确定性 Tabular/合成数据证明生命周期和判定规则可复核。

### Stage D — 真实 TSIL 最小闭环

仅在 Stage A/B Gate 通过、数据/资源许可时，在 2018–2023 历史开发范围内运行：
- `Smoke A`：一个合法谓词 + MSE + Phi_ones，单 seed、单开发折、极小训练预算；核对真实 P/梯度/预测、日期索引、日志及数值。
- `Smoke B`：一轮、2–3 候选、3 个不同 seeds、受限 Screen/Full；验证去重、失败处理、反馈、证据和可恢复性。可在已暴露历史数据上机械演练 Freeze/Confirm，但**必须标记 exploratory/integration**，不进入真实 Final。

未取得真实 qlib、GPU、历史研究限制不允许或资源不足时给出 CODE_ONLY 或 BLOCKED_RESOURCE，并给出一条可复核的阻断原因；禁止生成伪造指标。

### Stage E — 有界自主执行与知识积累

Stage C/D 实证可行后引入事件队列、逻辑 Research Router/Worker、Watchdog、统一低打扰审批入口。只自动运行低风险、预算内且开发数据允许的任务。对冻结协议变更、新确认集消费、阈值修改、对外发布、删除数据或超预算必须产生待审 DecisionRecord。把通过、失败、不确定三类结果都写入可检索知识库；支持知识冲突、证据失效、过期复核与重新开启新版本研究。

### Stage F — 独立验收、文档与交付

输出架构/数据流/任务状态 Mermaid 图、操作指南、配置项与默认风险级别、测试矩阵、真实执行日志和证据包、未完成项及下一批研究建议；若需要前端，优先在现有 CLI/报告可用后再实现研究工作台，不将前端完成当作核心验收。

## 8. 强制验收矩阵（必须逐项报告）

| ID | 验收项 | 可独立证明的通过条件 |
|---|---|---|
| AC-01 | 历史无损 | 迁移前后各表/引用/历史摘要核对，原始备份可恢复；原有工作树未被破坏 |
| AC-02 | 原生机制 | 真实 TSIL `phi_fn→P→loss→gradient` 数值对照；不是 mock/拼接 |
| AC-03 | 时序隔离 | 构造未来标签、early-stop 窜用、跨期预处理泄漏均被拒绝并留痕 |
| AC-04 | 历史暴露 | 已使用的 2024–2025 holdout 和 f2023 不会因新 run/新 db/重命名恢复独立资格 |
| AC-05 | 完整搜索 | 生成、校验、筛选、全量实验族、多 seed/折、反馈、冻结、报告可追踪 |
| AC-06 | 反证能力 | 安慰剂、信息时点错误、正/负控、通用正则化替代机制可按协议运行并归因 |
| AC-07 | 统计可信 | 效果序列可重算，HAC/bootstrap 与多重比较前提注明；拒绝伪独立样本 |
| AC-08 | 科学状态 | 工程成功/宽松阈值/部分 fold 失败无法升级为独立科学确认 |
| AC-09 | 可恢复性 | 中断、OOM、超时、重复事件、乱序回调、损坏缓存不丢证据或重复计数 |
| AC-10 | 安全性 | LLM 结构化输入不能执行任意代码，任务配额、权限和冻结不可被 Worker 绕过 |
| AC-11 | 真实集成 | 有真实 TSIL smoke 的完整证据，或明确 CODE_ONLY/BLOCKED_RESOURCE，不冒充成功 |
| AC-12 | 独立复核 | 独立脚本从原始预测文件和清单重算效应、置信区间、结论；哈希匹配 |
| AC-13 | 领域扩展 | Tabular 全流程仍可执行；Mineral/Image 只声明契约并明确拒绝未实现执行 |
| AC-14 | 自主性受控 | 事件触发/Watchdog 能推进授权实验，审批门槛不可自主降低，失败知识可查询 |

工程等级与科学证据等级分开汇报：
- 工程：`L1 Core/CI → L2 real TSIL → L3 discovery/falsification → L4 scientific protocol → L5 knowledge/autonomy`；各层必须独立通过。
- 科学：`E0 工程 → E1 机制/正负控 → E2 开发期稳健证据 → E3 新的预注册独立确认 → E4 外部复现 → E5 前瞻持续验证`。

**允许的结论**：`PASS_INTEGRATION / CONDITIONAL_PASS / FAIL / BLOCKED_RESOURCE` 是工程结论；`SUPPORTED/REFUTED/INCONCLUSIVE/NOT_TESTED` 是研究结论。科学阴性结果不使工程验收失败。对没有新独立数据的本轮实现，最高不得声称新达成 E3。

## 9. Coding Agent 执行纪律与必须输出

1. 开始工作时先交付**基于真实源码的差距矩阵**：`需求ID → 现有文件/函数 → 实际状态 → 最小增量实现 → 测试 → 风险/优先级`；随后直接实施 P0 和可安全完成的任务，不因一般性歧义停留在讨论。
2. 所有判断分四种标签：`THEORY_SUPPORTED`、`EXPERIMENT_VERIFIED`、`CODE_VERIFIED`、`UNVERIFIED_OR_PROPOSED`，禁止把规划或旧文档断言当作当前实现。
3. 每项改动按 `contract/test → implementation → regression → adversarial test → documentation` 推进；避免创建重复调度器、重复数据模型或与原有 Core 竞争的第二套真相来源。
4. 所有自动提案、统计判定、人工决定和跨版本知识更新都有输入、产出、依据和不可变关联 ID。独立审计 Worker 只读，不允许审计自己刚改过的验收程序。
5. 遇到科学/权限/资源 Gate 不通过应阻断对应高风险动作，但继续完成可执行的代码、测试和文档任务，不用伪造结果来凑验收率。
6. 交付文件至少包含：`docs/research_baseline_20261106.md`、`docs/research_architecture.md`、`docs/scientific_validation_protocol.md`、`docs/implementation_gap_matrix.md`、`docs/independent_acceptance.md`、`docs/known_limitations.md`；命名可配合已有仓库规范但需提供一一映射。
7. 最终报告按 AC-01 至 AC-14 输出状态、证据路径、可复现命令、测试结果和阻断项；附具体改动文件/迁移、测试统计、真实数据范围、GPU 资源/耗时、未触碰的历史 holdout 声明以及下一步科学确认所缺少的条件。

**现在开始：第一步先审查真实仓库与原始结果，给出可核查的事实基线和差距矩阵，然后按 Gate 顺序实际实施。不要用长篇未来规划替代开发。**

## 10. 参考材料与优先级

先读代码和原始实验，其次读技术指南，再读研究论文与外部工具文档。外部材料是方法候选，不可用“行业流行”替代本项目的实验对照。

- Vapnik & Izmailov, *Complete Statistical Theory of Learning (Learning Using Statistical Invariants)*: https://proceedings.mlr.press/v128/vapnik20a.html
- Dwork et al., *The Reusable Holdout*: https://pubmed.ncbi.nlm.nih.gov/26250683/
- White, *A Reality Check for Data Snooping*: https://doi.org/10.1111/1468-0262.00152
- Hansen, *A Test for Superior Predictive Ability*: https://doi.org/10.1198/073500105000000063
- scikit-learn `TimeSeriesSplit`: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- PySR search-space controls: https://ai.damtp.cam.ac.uk/pysr/api
- tsfresh relevance filtering: https://tsfresh.readthedocs.io/en/stable/api/tsfresh.feature_selection.html
- Optuna reproducibility: https://optuna.readthedocs.io/en/stable/faq.html
- MLflow Tracking: https://mlflow.org/docs/latest/tracking/
- Sakana AI *The AI Scientist* limitations: https://sakana.ai/ai-scientist/
- 用户提供的 `1. Super-agent.md` 与 `kzk_agent.md`：仅将其作为事件驱动/执行调度/Watchdog/人工审批的架构参考，而非科学有效性的证据。
