# Predicate_Lab_v1.1 研究者快速上手与代码架构指南

> **文档定位**：面向即将在 `Predicate_Lab_v1.1` 上开展 **(a) 结构化数据谓词检索与分析** 与 **(b) 量化时序（TSIL）新谓词探索** 的研究人员。本文档串联整体代码运行链路、防泄漏统计协议，以及“在什么情况下需要修改哪几行代码”的精确操作手册。

---

## 1. 平台研究就绪度总览（2026-11-07 验证基线）

在开始正式研究前，`Predicate_Lab_v1.1` 已完成全链路科学可信度审计（`R01`–`R09`）修复与实测验证：

| 验证维度 | 关键指标 / 结果 | 科学含义与保障 |
|---|---|---|
| **回归测试套件** | `137 / 137 passed` | 覆盖 `D-01`–`D-16` 历史审计与 `R01`–`R09` 本轮可信度修复 |
| **时序窗口边界 (`R01`)** | 跨股票切片数 `0 / 71,297` | `_create_ts_slices` 按股票分组截断，彻底消除新股借用前一只股票未来价格的跨品种泄漏 |
| **评估折完整性 (`R02`)** | 尾批次保留率 `100%` | `MTSDatasetH.eval()` 设 `drop_last=False`，消除字典序末尾股票被静默丢弃导致的早停与 IC 偏差 |
| **零假设全流程校准 (`N2`/`N3`)** | Discovery `1.50%–2.25%`；Confirmation `0.00%` (`0/400`)；Holdout `0.00%` (`0/400`) | 在存在种子×折、算子族×年份相关噪声下，全流程累积 BH FDR（`R07`）将假阳性严格压制在 `10%` 名义水平以下 |
| **阳性对照全流程检出 (`P2`)** | Discovery `99.25%`；Confirmation `93.75%`；Holdout `88.75%` (`355/400`) | 注入 `+0.010` 日度 IC（`5×MME`）时端到端检出率达 `88.75%`，且确认与终审阶段真族（`volatility`）精确率均为 `100%` |
| **真实 CSI300 + GPU 验证** | `23/23` 原生谓词满足行置换等变性；`22/23` 原生 + `7/7` DSL 族通过真实探针 | 真实 CSI300 数据（`113,130` 个文件）+ NVIDIA A100 GPU 端到端打通，且 `f2023` 与 `holdout_2024_2025` 保持零消耗 |

---

## 2. 整体代码目录与端到端运行链路

### 2.1 核心代码模块地图

```text
Predicate_Lab_v1.1/
├── configs/
│   ├── tabular_quick.yaml          # 结构化数据（tabular 域）标准实验配置
│   ├── tsil_lstm_smoke.yaml        # TSIL 快速冒烟测试配置
│   ├── tsil_lstm_quick.yaml        # TSIL 中等规模探索配置
│   └── tsil_lstm_full.yaml         # TSIL 正式研究配置（3 种子 × 50 epochs）
├── predicatelab/
│   ├── cli.py                      # 命令行入口（validate-config / discover / freeze-family / confirm / finalize）
│   ├── core/
│   │   ├── engine.py               # 核心状态机与调度引擎（Probe -> Screen -> Full Validation -> BH FDR -> Confirm -> Finalize）
│   │   ├── stats.py                # 成对差异序列、Newey-West HAC t 检验、Moving-Block Bootstrap、Benjamini-Hochberg FDR
│   │   ├── holdout.py              # 一次性 Holdout 与 Confirmation 门控 + 标签前瞻隔离检查
│   │   ├── store.py                # SQLite 实验状态库 + 跨 DB 全局暴露账本（.predicatelab_global_ledger.sqlite）
│   │   └── identity.py             # 无 Git 环境全量源码树 SHA-256 指纹 + 数据全文件 Merkle Root
│   └── domains/
│       ├── tabular/                # 【研究方向 a】结构化表格数据谓词检索域
│       │   ├── adapter.py          # 数据加载、折内参数拟合 fit_transform、探针与基线模型评估
│       │   └── generator.py        # 初始种子对与基于 screen_effect 的跨轮变异器
│       └── tsil/                   # 【研究方向 b】量化时序不变损失（TSIL）谓词探索域
│           ├── adapter.py          # Qlib 数据契约、CPU 探针 5 项数学门控、GPU Worker 子进程调度
│           ├── dsl.py              # 受信任谓词编译器（7 个参数化 DSL 族 + 23 个原生谓词白名单 + Composite 组合）
│           ├── seeds.py            # 第 0 轮初始种子池（paper6 / exploratory / dsl）
│           ├── generator.py        # RuleGenerator（窗口/特征切片/P-mix 变异）与 JsonProposalGenerator
│           └── worker.py           # 独立 GPU 训练子进程（加载 Qlib、注入 phi_fn、输出逐日预测 .npz 与截面 IC）
└── TSIL-main/
    └── src/
        ├── dataset_ori_onehot.py   # Qlib MTSDatasetH 时序切片（含 R01 防跨股截断、R02 eval 全保留、joblib>=1.3 补丁）
        ├── model_backbone_phitp.py # QniverseModel (LSTM/GRU/Transformer) + WeightedMSELoss (tau_hat * I + tau * P)
        ├── temporal_predicates.py  # TSIL 论文 6 大核心时序谓词（含 R05 序列轴 FFT 修复）
        └── custom_phi_constructors.py # 17 个金融领域扩展原生谓词构造器
```

### 2.2 端到端运行链路（`discover` $\to$ `freeze-family` $\to$ `confirm` $\to$ `finalize`）

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ 阶段 0: validate-config (纯静态检查，0 账本消耗)                                         │
│   读取 YAML -> 计算全量源码 SHA-256 与数据 Merkle 根 -> 检查 Purge/Embargo 与日历覆盖     │
└───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ 阶段 1: discover (多轮迭代搜索，仅访问探索折 screen_folds + validation_folds)             │
│                                                                                         │
│   Round r (r = 0, 1, ...):                                                              │
│     1. 候选生成 (Generator.propose)                                                     │
│        ├─ r=0: 从种子池 (seed_templates) 生成初始谓词模板                                 │
│        └─ r>0: 仅按上一轮【screen_effect】排序挑选 Top-K 父代进行窗口/切片/组合变异 (R07) │
│     2. 静态校验与 CPU 探针门控 (Adapter.validate_descriptor + Adapter.probe)             │
│        ├─ 检查有限性、对称性、非退化性、PSD (λ_max ≤ 1)、批次行置换等变性 (R05)           │
│        └─ 提取上三角行为签名 (probe_signature)，若与已有候选余弦距离 ≈ 0 则自动去重       │
│     3. 两阶段评估 — 第一步：Screen 折快速初筛 (screen_folds, 如 f2022, 单种子, 10 epochs) │
│        ├─ 运行 baseline:mse 与 control:ones（按折+种子缓存复用）                          │
│        ├─ 计算每个候选相对 baseline 的成对 ΔIC (screen_effect)                           │
│        └─ 保留 screen_effect ≥ min_screen_effect 的前 full_validation_top_k 个候选       │
│     4. 两阶段评估 — 第二步：Validation 折完整验证 (validation_folds, 如 f2020+f2021, 多种子)│
│        ├─ 跨所有随机种子拼接逐日成对差异 Δ_t = IC_cand(t) - IC_base(t)                   │
│        ├─ 计算 Newey-West HAC 标准误 + Moving-Block Bootstrap 95% 置信区间               │
│        └─ 对【当前 run 所有轮次已进入完整验证的全部候选】执行累积 Benjamini-Hochberg FDR │
│     5. 证据分级：同时满足 effect ≥ min_effect、CI 下界 > 0、q ≤ fdr_alpha、              │
│        stability ≥ min_stability 且击败 control:ones -> 记为 exploratory_positive        │
└───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ 阶段 2: freeze-family & confirm (独立确认折评估，如 f2023)                               │
│   1. freeze-family: 将选定的候选集写入 frozen_families 表并生成不可篡改的 manifest_hash  │
│   2. confirm: 检查全局 confirmation_ledger (R04)：                                       │
│      ├─ 首次访问该确认折: 在 f2023 上运行配对试验 + 家族内 BH 校正 ->                    │
│      │   通过者晋升为 independently_confirmed，并写入全局确认暴露账本                     │
│      └─ 二次/跨库重复访问该确认折: 严格模式下抛出 ConfirmationConsumed；                  │
│          默认模式下自动降级为 exploratory_positive（禁止虚报独立确认）                    │
└───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ 阶段 3: finalize (终极 Holdout 一次性终审，如 holdout_2024_2025)                         │
│   1. 检查候选状态必须为 independently_confirmed                                          │
│   2. 原子抢占本地与全局账本 (.predicatelab_global_ledger.sqlite) 的 holdout_access 锁 (R03)│
│      ├─ 若该数据集的 Holdout 已被任何历史 run 消耗 -> 立即抛出 HoldoutConsumed 拒绝执行  │
│      └─ 首次消耗 -> 在 2024-2025 测试窗上训练并评估唯一候选，写入 final_evals 与报告      │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 速查决策矩阵：在什么情况下，需要改什么代码？

| 编号 | 研究场景 | 是否需改 Python 代码 | 涉及文件与行号 | 改动工作量 |
|---|---|---|---|---|
| **A1** | **结构化数据**：换用自己的 CSV / Parquet 数据集 | **是** | 1. [predicatelab/domains/tabular/adapter.py:122-143](../predicatelab/domains/tabular/adapter.py#L122-L143)<br>2. [predicatelab/domains/tabular/adapter.py:193-267](../predicatelab/domains/tabular/adapter.py#L193-L267)<br>3. [predicatelab/domains/tabular/generator.py:36](../predicatelab/domains/tabular/generator.py#L36) | ~30 行 |
| **A2** | **结构化数据**：新增自定义特征交互/非线性谓词算子 | **是** | 1. [predicatelab/domains/tabular/adapter.py:33-116](../predicatelab/domains/tabular/adapter.py#L33-L116)<br>2. [predicatelab/domains/tabular/adapter.py:274-287](../predicatelab/domains/tabular/adapter.py#L274-L287)<br>3. [predicatelab/domains/tabular/generator.py:20-125](../predicatelab/domains/tabular/generator.py#L20-L125) | ~25 行 |
| **A3** | **结构化数据**：将基线模型从 LogisticRegression 换成 LightGBM / 神经网络 / 回归 | **是** | [predicatelab/domains/tabular/adapter.py:315-345](../predicatelab/domains/tabular/adapter.py#L315-L345) | ~20 行 |
| **B1** | **TSIL 时序**：用手工设计或 LLM 生成的 JSON 提案测试新窗口、特征子集切片、多谓词组合 | **否（0 行代码）** | 新建 `proposals.json` + 修改 YAML 中 `search.generator: llm_json` | 仅配置 |
| **B2** | **TSIL 时序（推荐）**：新增参数化单样本时序算子族（自动保证 $P$ 矩阵对称/半正定/置换等变） | **是** | 1. [predicatelab/domains/tsil/dsl.py:58-66](../predicatelab/domains/tsil/dsl.py#L58-L66) 与 [225-294](../predicatelab/domains/tsil/dsl.py#L225-L294)<br>2. [predicatelab/domains/tsil/seeds.py:47-64](../predicatelab/domains/tsil/seeds.py#L47-L64)<br>3. [predicatelab/domains/tsil/generator.py:136](../predicatelab/domains/tsil/generator.py#L136) 与 [209](../predicatelab/domains/tsil/generator.py#L209) | ~20 行 |
| **B3** | **TSIL 时序**：在 `TSIL-main` 中直接编写自定义 $(I, P)$ 核矩阵构造函数（非内积型 $P$） | **是** | 1. [TSIL-main/src/custom_phi_constructors.py](../TSIL-main/src/custom_phi_constructors.py)<br>2. [predicatelab/domains/tsil/dsl.py:31-55](../predicatelab/domains/tsil/dsl.py#L31-L55) | ~35 行 |
| **B4** | **TSIL 时序**：切换股票池（`csi300` $\to$ `csi500`）、修改预测 Horizon/Label、切换 `GRU`/`Transformer` | **否（0 行代码）** | 修改 [configs/tsil_lstm_full.yaml](../configs/tsil_lstm_full.yaml) 中 `data` / `training` 段 | 仅配置 |

---

## 4. 实战指南 A：换一个结构化数据集开始谓词检索与分析（`tabular` 域）

### 4.1 场景 A1：接入你自己的 CSV / Parquet 数据集

当前 [predicatelab/domains/tabular/adapter.py](../predicatelab/domains/tabular/adapter.py) 默认加载 `sklearn.datasets.load_breast_cancer`。要支持任意 CSV 或 Parquet 数据集，请按以下 3 步修改：

#### 第一步：修改 `predicatelab/domains/tabular/adapter.py` 的数据加载与 `dataset_id`

打开 [predicatelab/domains/tabular/adapter.py:122-143](../predicatelab/domains/tabular/adapter.py#L122-L143)，将 `__init__` 替换为：

```python
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.data_cfg = config.get("data", {})
        data_path = self.data_cfg.get("path")
        if data_path:
            import pandas as pd

            p = Path(data_path)
            df = pd.read_parquet(p) if p.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(p)
            target_col = self.data_cfg["target_col"]
            drop_cols = set(self.data_cfg.get("drop_cols", [])) | {target_col}
            feature_cols = [c for c in df.columns if c not in drop_cols]
            self.X = df[feature_cols].to_numpy(dtype=float)
            self.y = df[target_col].to_numpy()
            self.feature_names = [str(c) for c in feature_cols]
            self.dataset_id = str(self.data_cfg.get("dataset_id", f"tabular-{p.stem}"))
            self.dataset_path = str(p.resolve())
        else:
            from sklearn.datasets import load_breast_cancer

            self.X, self.y = load_breast_cancer(return_X_y=True)
            self.feature_names = list(load_breast_cancer().feature_names)
            self.dataset_id = "sklearn-breast-cancer"
            self.dataset_path = "sklearn.datasets.load_breast_cancer"

        self.n_features = int(self.X.shape[1])
        self.seed = int(config.get("training", {}).get("seed", 2025))
        self._evaluation = EvaluationSpec.model_validate(
            config.get("evaluation", {
                "name": "tabular_cv_logloss",
                "primary_metric": "auc",
                "direction": "higher",
                "secondary_metrics": ["accuracy", "log_loss"],
                "paired": True,
                "multiple_testing": "benjamini-hochberg",
                "notes": "repeated stratified 5-fold cross-validation on the 80% search split",
            })
        )
        self._folds, self._confirmation, self._holdout = self._build_folds()
```

同时将同文件中以下 4 处写死的 `"sklearn-breast-cancer"` 改为 `self.dataset_id`（確保新数据集拥有独立的 Holdout 账本标识）：
- 第 195 行：`dataset_id=self.dataset_id,`
- 第 228 行：`dataset_id=self.dataset_id,`（第 232 行 `path=self.dataset_path,`）
- 第 251 行：`task_id=f"task-{self.dataset_id}",` 与第 264 行 `"dataset": self.dataset_id`
- 第 378 行：`"dataset": self.dataset_id`

#### 第二步：修改 `predicatelab/domains/tabular/generator.py` 使其自适应特征列数

打开 [predicatelab/domains/tabular/generator.py:36](../predicatelab/domains/tabular/generator.py#L36)，将写死的 `self.seed_pairs` 改为按 `n_features` 自动过滤并补充前几列组合：

```python
        default_pairs = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (0, 13), (6, 7), (20, 21)]
        valid_pairs = [(a, b) for a, b in default_pairs if a < n_features and b < n_features]
        if not valid_pairs and n_features >= 2:
            valid_pairs = [(0, 1)]
        self.seed_pairs = valid_pairs
```

#### 第三步：创建你自己的实验配置文件 `configs/tabular_my_study.yaml`

```yaml
run:
  name: tabular_my_study
  domain: tabular
  artifacts_dir: artifacts/my_tabular_study
  db: artifacts/my_tabular_study/predicatelab.sqlite
  rounds: 3
  require_confirmation: true

seeds: [2025, 2026, 2027]
data:
  path: /path/to/your_dataset.csv
  target_col: target
  drop_cols: [id, timestamp]
  dataset_id: credit_risk_v1

protocol:
  cv_splits: 5
  cv_repeats: 5
  discovery_repeats: 4      # 前 4 个 repeat (20 折) 用于搜索与验证，第 5 个 repeat (5 折) 留作确认集
  confirmation_kind: resampling

evaluation:
  name: tabular_cv_logloss
  primary_metric: log_loss  # 或 auc
  direction: lower          # log_loss 选 lower；auc 选 higher
  secondary_metrics: [auc, accuracy]
  paired: true
  multiple_testing: benjamini-hochberg

search:
  generator: rule
  save_predictions: true
  generator_config:
    mutation_top_k: 4
  max_candidates_per_round: 20
  screen_folds: [cv_r0_f0]
  validation_folds:
    - cv_r0_f1
    - cv_r0_f2
    - cv_r0_f3
    - cv_r0_f4
    - cv_r1_f0
    - cv_r1_f1
    - cv_r1_f2
    - cv_r1_f3
    - cv_r1_f4
  full_validation_top_k: 4
  min_screen_effect: 0.0
  fdr_alpha: 0.10
  min_effect: 0.0005
  min_stability: 0.60
  bootstrap_samples: 1000
  bootstrap_block: 1
  nw_lag: 0
  seed: 2025
```

---

### 4.2 场景 A2：在结构化数据中新增自定义谓词算子

假设你想新增一个 **标准化双特征软门控算子 `gated_product`**（当特征 $b$ 高于训练集均值时放大特征 $a$）：

1. **在 [predicatelab/domains/tabular/adapter.py:33](../predicatelab/domains/tabular/adapter.py#L33) 注册算子名与校验规则**：
   ```python
   OPS = {"ratio", "product", "difference", "sigmoid", "threshold", "interaction", "gated_product"}
   ```
   并在第 53 行的双元特征校验集合中加入 `"gated_product"`：
   ```python
   if op in {"ratio", "product", "difference", "interaction", "gated_product"}:
   ```
2. **在 [predicatelab/domains/tabular/adapter.py:67-116](../predicatelab/domains/tabular/adapter.py#L67-L116) 的 `fit_transform` 中实现折内拟合变换**：
   > **防泄漏铁律**：均值 `mb` 和标准差 `sb` **只能从 `X_train` 计算**，绝不能接触 `X_other`！
   ```python
   elif op == "gated_product":
       b = int(desc["b"])
       params["b"] = b
       mb, sb = float(X_train[:, b].mean()), float(X_train[:, b].std() + 1e-8)
       params.update({"mean_b": mb, "std_b": sb})
       gate_tr = 1.0 / (1.0 + np.exp(-(X_train[:, b] - mb) / sb))
       gate_ot = 1.0 / (1.0 + np.exp(-(X_other[:, b] - mb) / sb))
       train = X_train[:, a] * gate_tr
       other = X_other[:, a] * gate_ot
   ```
3. **在 [predicatelab/domains/tabular/adapter.py:277-284](../predicatelab/domains/tabular/adapter.py#L277-L284) 声明其 `CandidateKind`**：
   ```python
   "gated_product": {"soft"},
   ```
4. **在 [predicatelab/domains/tabular/generator.py:20-27](../predicatelab/domains/tabular/generator.py#L20-L27) 的 `KIND` 字典中添加 `"gated_product": CandidateKind.SOFT`**，即可参与自动种子生成与邻近特征变异。

---

## 5. 实战指南 B：在 TSIL 上探索新的时序谓词（`tsil` 域）

### 5.1 TSIL 谓词的数学契约与机制要点

在 `TSIL-main` 中，骨干模型（LSTM / GRU / Transformer）输出预测 $\hat{y} \in \mathbb{R}^B$，残差向量为 $e = \hat{y} - y \in \mathbb{R}^B$。统计不变损失（Statistical Invariant Loss）定义为：

$$\mathcal{L}(e; I, P) = \hat{\tau}\, \frac{1}{B} e^\top I e + \tau\, \frac{1}{B} e^\top P e$$

每个谓词函数 `phi_fn(batch_x, timestamp_x)` 接收：
- `batch_x`：形状 `[B, T, F]`（在 CSI300 默认配置下为 `[800, 20, 158]`，即 800 个样本、过去 20 个交易日、158 个 Alpha158 特征）；
- `timestamp_x`：形状 `[B, T, 3]`（月、日、星期特征）；
- 返回 `(I, P)`，其中 $I = I_B$，$P \in \mathbb{R}^{B \times B}$。

> **关键研究洞察（来自可信度审计 `R05/R06`）**：
> 在默认训练配置 `shuffle: true, batch_size: 800` 下，一个 mini-batch 内的 800 个样本是从整个训练期（2~4 年，约 500~1000 个交易日）随机抽样的，**同一个交易日在同一个 batch 内平均只有约 1~2 只股票**。
> - **应该怎么做**：设计**单样本时序特征算子** $\phi_i = g(X_i) \in \mathbb{R}^d$（只依赖样本 $i$ 自身的 $20 \times 158$ 时序窗口，如波动率结构、回撤形态、量价相关性、上下行非对称性），再通过 $P = \tilde{\phi}\tilde{\phi}^\top$ 让具有相似动力学形态的样本在损失函数中对齐残差。
> - **不要怎么做**：不要在谓词里把 `batch_x.mean(dim=0)` 当作“当日市场指数收益率”，因为随机打乱的 mini-batch 跨度长达数年，跨样本均值并不是同一天的市场截面均值。

---

### 5.2 三种新谓词探索路径（按复杂度由浅入深）

#### 路径 B1：零代码修改 — 用 JSON 提案探索特定特征子空间与多谓词组合

Alpha158 的 158 个特征包含 K 线形态（KBAR）、价格（PRICE）和滚动量价统计（ROLLING）。如果你想探索“在特定特征子集上的谓词”或“多谓词加权投影混合（`P-mix`）”，**无需修改任何 Python 代码**：

1. 创建提案文件 `configs/my_tsil_proposals.json`：
   ```json
   {
     "proposals": [
       {
         "round": 0,
         "kind": "statistical",
         "name": "Tail risk on rolling price-volume slice [60:158] w=15",
         "rationale": "在滚动量价特征子集上度量15日偏度与峰度尾部结构",
         "descriptor": {
           "family": "tail",
           "params": {"window": 15},
           "feature_slice": [60, 158]
         }
       },
       {
         "round": 0,
         "kind": "composite",
         "name": "P-mix(native volatility, DSL drawdown w=20, DSL trend w=10)",
         "rationale": "联合多窗口原生波动率、最大回撤与趋势拟合优度",
         "descriptor": {
           "family": "composite",
           "combine": "mix",
           "parts": [
             {"family": "native", "predicate": "phi_volatility", "params": {"window_sizes": [5, 10, 20]}},
             {"family": "drawdown", "params": {"window": 20}},
             {"family": "trend", "params": {"window": 10}}
           ],
           "weights": [1.5, 1.0, 1.0]
         }
       }
     ]
   }
   ```
2. 在 YAML 配置中启用 `llm_json` 生成器：
   ```yaml
   search:
     generator: llm_json
     generator_config:
       proposals_path: /home/zhengzengrong/PredicateLab/Predicate_Lab_v1.1/configs/my_tsil_proposals.json
       seed_pool: paper6+dsl
   ```

---

#### 路径 B2（最推荐）：在 `predicatelab/domains/tsil/dsl.py` 中新增参数化时序算子族

当你想提出一个新的数学谓词族（例如 **上下行波动非对称性 `asymmetry`** 或 **量价背离 `pv_divergence`**）并让搜索引擎自动对它进行窗口变异、特征切片变异和复合组合时，只需修改以下 3 个文件：

1. **在 [predicatelab/domains/tsil/dsl.py:58-66](../predicatelab/domains/tsil/dsl.py#L58-L66) 的 `OP_PARAMS` 中注册新算子族**：
   ```python
   OP_PARAMS = {
       "momentum": {"window": "int"},
       "volatility": {"window": "int"},
       "dispersion": {"window": "int"},
       "autocorr": {"window": "int"},
       "drawdown": {"window": "int"},
       "tail": {"window": "int"},
       "trend": {"window": "int"},
       "asymmetry": {"window": "int"},  # 新增：上下行波动非对称性算子族
   }
   ```
2. **在 [predicatelab/domains/tsil/dsl.py:225-294](../predicatelab/domains/tsil/dsl.py#L225-L294) 的 `raw_phi(descriptor, batch_x)` 中实现特征提取**：
   > 输入 `batch_x` 已自动完成 `nan_to_num`、`feature_slice` 切片和最后 `k` 步截取（`xw` 形状为 `[B, k, F]`，`m = xw.mean(dim=2)` 形状为 `[B, k]`）。你只需返回形状为 `[B, d]` 的张量，后续列标准化、全局 $L_2$ 归一化和 $P = \tilde{\phi}\tilde{\phi}^\top$ 由 `_dsl_I_P` 自动完成！
   ```python
   if family == "asymmetry":
       # 围绕样本自身时间均值的上下行偏离
       centered = m - m.mean(dim=1, keepdim=True)
       pos = torch.clamp(centered, min=0.0)
       neg = torch.clamp(-centered, min=0.0)
       up_rms = (pos.pow(2).mean(dim=1) + 1e-8).sqrt()
       down_rms = (neg.pow(2).mean(dim=1) + 1e-8).sqrt()
       log_ratio = torch.log(up_rms / down_rms)
       up_freq = (centered > 0).float().mean(dim=1)
       return torch.stack([up_rms - down_rms, log_ratio, up_freq], dim=1)
   ```
3. **在 [predicatelab/domains/tsil/seeds.py:47-64](../predicatelab/domains/tsil/seeds.py#L47-L64) 与 [predicatelab/domains/tsil/generator.py:136,209](../predicatelab/domains/tsil/generator.py#L136) 中加入 `"asymmetry"`**：
   - 在 `seeds.py` 的 `DSL_SEEDS` 列表末尾添加：
     ```python
     (CandidateKind.STATISTICAL, "Up-down asymmetry (window 10)", {"family": "asymmetry", "params": {"window": 10}}),
     ```
   - 在 `generator.py` 第 136 行和第 209 行的算子族集合/列表中加入 `"asymmetry"`。

---

#### 路径 B3：在 `TSIL-main/src/custom_phi_constructors.py` 中编写自定义核矩阵 $P$

如果你研究的谓词不是内积投影形式 $P = \tilde{\phi}\tilde{\phi}^\top$（例如你想直接定义高斯 RBF 核矩阵 $P_{ij} = \exp(-\|\phi_i - \phi_j\|^2 / \sigma^2)$ 并做迹归一化）：

1. **在 [TSIL-main/src/custom_phi_constructors.py](../TSIL-main/src/custom_phi_constructors.py) 中定义函数并在文件底部的 `get_custom_I_P` 字典注册**：
   ```python
   def get_rbf_vol_I_P(batch_x: torch.Tensor,
                       device: Optional[torch.device] = None,
                       **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
       batch_size = batch_x.shape[0]
       if device is None:
           device = batch_x.device
       I = torch.eye(batch_size, device=device)
       x = torch.nan_to_num(batch_x.float(), nan=0.0, posinf=0.0, neginf=0.0)
       # 提取每个样本的时序波动特征 [B, d]
       feat = x.std(dim=1)  # [B, F]
       feat = (feat - feat.mean(dim=0, keepdim=True)) / (feat.std(dim=0, keepdim=True) + 1e-6)
       # 计算成对欧氏距离平方并构造 RBF 核（正定矩阵）
       dist_sq = torch.cdist(feat, feat, p=2.0).pow(2)
       median_sq = torch.median(dist_sq).clamp_min(1e-6)
       K = torch.exp(-dist_sq / (2.0 * median_sq))
       # 【关键】迹归一化使 trace(P) = 1，保证半正定且 lambda_max(P) <= 1
       P = K / (torch.diagonal(K).sum() + 1e-8)
       return I, P
   ```
   并在 `custom_phi_constructors.py` 底部的 `constructors` 映射中加入 `"phi_rbf_vol": get_rbf_vol_I_P`。
2. **在 [predicatelab/domains/tsil/dsl.py:31-55](../predicatelab/domains/tsil/dsl.py#L31-L55) 的 `NATIVE_PREDICATES` 中添加白名单**：
   ```python
   "phi_rbf_vol": {"params": {}},
   ```

---

### 5.3 新谓词 5 秒 CPU 探针快速自检（上 GPU 前必跑）

每次编写完新谓词后，**不要直接启动耗时的 GPU 训练**。先运行下面这段 5 秒钟的 CPU 探针命令，直接在真实 CSI300 探针 batch 上检验 5 大数学门控（有限性 `finite`、对称性 `symmetric`、非退化性 `non_degenerate`、半正定谱界 `psd_lambda_le_one`、批次行置换等变性 `permutation_equivariant`）：

```bash
PYTHONPATH=/home/zhengzengrong/PredicateLab/Predicate_Lab_v1.1 \
/home/zhengzengrong/miniconda3/envs/predicatelab/bin/python -c '
import yaml
from predicatelab.domains.tsil.adapter import TSILAdapter

cfg = yaml.safe_load(open("/home/zhengzengrong/PredicateLab/Predicate_Lab_v1.1/configs/tsil_lstm_smoke.yaml"))
cfg["run"]["artifacts_dir"] = "/home/zhengzengrong/PredicateLab/audit_20261107_credibility/tmpdata/tsil_smoke_verify_v2/artifacts"
adapter = TSILAdapter(cfg)
snap = adapter.snapshot(cfg)
f2022 = [f for f in adapter.build_protocol(cfg).folds if f.fold_id == "f2022"][0]

# 将此处 descriptor 替换为你刚写的谓词描述符：
descriptor = {"family": "volatility", "params": {"window": 10}, "feature_slice": [0, 64]}
res = adapter.probe(descriptor, f2022, snap, cfg, "/tmp/probe_quickcheck")
print("ok =", res["ok"], "| reason =", res["reason"])
for k, v in res.get("checks", {}).items():
    print(f"  {k}: {v}")
'
```
只有当输出 `ok = True` 时，才将该谓词投入 `discover` GPU 训练。

---

## 6. 科学防泄漏协议与账本红线（必读）

1. **探索阶段（`discover`）可无限次运行**：
   - `discover` 只会访问 `screen_folds`（`f2022`）与 `validation_folds`（`f2020`, `f2021`）。
   - 你可以随时调整算子设计、修改超参数、更换 `--run-id` 或 `--db` 反复运行 `discover`。
2. **确认阶段（`freeze-family` $\to$ `confirm`）受跨数据库确认账本保护（`R04`）**：
   - `confirm` 会访问确认折（如 `f2023`），并在本地 SQLite 与全局账本 `.predicatelab_global_ledger.sqlite` 中记录此次暴露。
   - 若同一确认折被二次调用 `confirm`，平台会自动将结果标记为 `confirmation_reused: true`，并将候选状态上限封顶为 `exploratory_positive`（若在 YAML 中开启 `run.strict_confirmation_ledger: true`，则直接抛出 `ConfirmationConsumed` 异常）。
3. **终审阶段（`finalize`）受全局一次性 Holdout 锁保护（`R03`）**：
   - `finalize` 会访问 `holdout_2024_2025`（或结构化数据的 `holdout_20pct`）。
   - 全局账本以 `(dataset_id, data_manifest_hash, test_start, test_end)` 为唯一键。**一旦对某个数据集执行了一次成功的 `finalize`，该数据集的 Holdout 窗口即被永久锁定**（即使新建 SQLite 数据库文件也无法绕过）。因此，请务必在完成全部探索与独立确认、敲定最终唯一候选谓词后，再执行 `finalize`。
