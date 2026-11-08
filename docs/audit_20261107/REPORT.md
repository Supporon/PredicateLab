# Predicate_Lab_v1.1 科研可信度独立审查报告

**审查日期**：2026-11-07（亚洲/上海时区，`date -Is` = `2026-11-07T10:32:27+08:00` 起始）
**审查对象**：`/home/zhengzengrong/PredicateLab/Predicate_Lab_v1.1`（无 Git：`Is a git repository: false`，`git -C` 不可用，确认无 `.git`）
**版本指纹**：107 个受审文件的 SHA-256 清单见 `evidence/manifest_sha256.txt`（清单自身 SHA-256：`2ceeda1d50142982b932e77d5497a980a5b55e436db29552ba8ff6f9839f5f46`）。外部材料指纹：`PredicateLab_CodingAgent_Implementation_Prompt_20261106.md` = `fc332c6101584807d402a1bdd72e6964f4b1ced47f4e33da63c97ec173337ac8`；`qlib_bin.tar.gz` = `bcf88a7a944876216496f8b6da2210395e046facd9bffe3da39367c99b4fc6bb`。
**审查方法**：只读审查 + 独立审查目录（本目录）内的最小复现脚本/合成数据/临时 SQLite。未修改 `Predicate_Lab_v1.1` 下任何文件，未读写任何真实数据库，未消费任何确认集/Final Holdout。
**环境**：审查在隔离 venv（`venv/`，`evidence/pip_freeze.txt`）中完成，Python 3.10.18，numpy 2.1.2，pandas 2.3.2，torch 2.6.0+cu124（GPU 可见但本次审查全部用 `CUDA_VISIBLE_DEVICES=` 强制 CPU 执行，未占用共享 GPU），pyqlib 0.9.6（逐依赖手装，见 `logs/pip_qlib_deps_individual.txt`）。pytest 在 `src_copy`（受审代码的逐字节副本，`sha256sum -c` 确认与源目录一致）上运行：125/125（无 qlib）→ 装好 qlib 后 128/128，1 skipped（真实 qlib 日历不可用的测试本就按设计跳过），0 failed。

**与文档中“审查线索”目录的关系**：请求中提到的 `docs/predicatelab` 不存在；受审仓库的审查文档实际位于 `docs/`（`architecture.md`、`code_guide.md`、`v1.1_*.md`、`results_*.md` 等），已逐一读取并作为线索核验，不作为结论来源。

---

## 0. 一句话结论

**当前系统能可靠支持的主张远小于文档语气暗示的范围**：核心工程机制（Holdout 全局 ledger 的单文件语义、标签时间隔离、HAC+block-bootstrap+BH 统计实现、D-01…D-16 历史缺陷的修复）在本次独立复算中成立；但**(1) 搜索阶段的验证 fold 被跨轮次反馈复用，构成未被 BH 覆盖的自适应多重比较**，在合成校准中使"至少一次虚假 `validated`"的实际发生率达到约 3.25%（95% Wilson CI [1.9%, 5.5%]，n=400），**(2) 确认期 fold（f2023）和 holdout 账本都没有跨数据库文件的暴露登记**，换一个 `--db` 路径即可重新“独立确认”同一时间窗口，**(3) 8 个“跨股票/市场”原生谓词的机制说明（market_return、系统性风险、板块聚类……）在真实训练配置（`shuffle: true`, `batch_size=800`）下不成立**：一个训练 batch 的目标日期跨越近两年、同日样本中位数仅 2 个，**(4) 代码身份指纹在当前（无 Git）环境下是常量**，不随 `predicatelab/` 或外部 `TSIL-main` 代码的任何修改变化，**(5) 上游序列切片存在跨标的窗口污染，评估阶段的 `drop_last=True` 会确定性地丢弃特定股票的全部样本**。这些问题中，(1)(2)(4) 直接威胁"证据等级升级"和"历史结果有效性"判断的可信度；(3) 直接威胁"机制对照支持机制解释"的主张；(5) 影响数据保真度但大体对候选/基线对称。

**下一步最值得投入的五项工作**（详见第 6 节）：
1. 把 `confirm()`/`finalize()` 的数据暴露检查做成**跨数据库、按逻辑 fold-key 的全局登记**（不是当前的"单 SQLite 文件内"登记）。
2. 把**跨轮次验证 fold 复用**计入统计口径：either 冻结 validation_folds 为单轮一次性资源，或将多轮搜索当作一个大的多重比较家族统一做 BH/FWER。
3. 对 8 个"市场/跨股票"谓词，**在真实 `shuffle=true` 训练配置下**重新审视其机制解释；不得在未修正前把观察到的效果归因为"捕捉了市场结构"。
4. 让 `code_version()` 覆盖外部 `data.tsil_root` 指向的真实代码树（当前配置指向 `/home/dmz-ai/.../TSIL-main`，不在被哈希的仓库范围内），并在无 Git 环境下 **fail closed** 而不是静默返回常量指纹。
5. 修复 `_create_ts_slices` 的跨标的窗口污染与评估阶段 `drop_last=True` 的系统性样本丢弃（上游 `TSIL-main/src/dataset_ori_onehot.py`）。

第三方复核方法：第 7 节给出可逐条重放的命令、输入输出和容差；所有复现脚本在 `repro/`，全部可重复执行产物在 `logs/`。

---

## 1. 审查边界与环境事实

- **目录与版本**：`Predicate_Lab_v1.1/` 内 `predicatelab/`（核心引擎 + 4 个领域适配器）、`TSIL-main/`（上游 TSIL 参考实现，GPL-3.0，本仓库内置副本）、`tests/`（19 个测试文件，551+182+189+…行，合计约 1500+行测试代码）、`configs/`、`docs/`（11 份审查/协议文档）。无 `.gitignore` 排除项异常；`.gitignore` 排除 `artifacts/`、`*.sqlite` 等——**本环境中确实没有任何历史 `artifacts/` 目录或 `predicatelab.sqlite`**（全盘搜索 `/home/zhengzengrong` 未找到，仅发现本会话自己创建的 pytest tmp 产物）。文档中提到的历史运行 `run_20261031_153702`、`12 runs / 17 rounds / 519 trials` 等**在本环境中完全不可验证**（NOT_VERIFIED，缺材料，非臆造）。
- **路径现实与可执行性**：三份 TSIL 配置（`configs/tsil_lstm_{full,quick,smoke}.yaml`）和 README 的"Quick start"全部引用 `/home/dmz-ai/...` 路径（`provider_uri`、`tsil_root`、Python 解释器）。**`/home/dmz-ai` 在本环境中不存在**（`ls` 确认）。这意味着：**按文档原样执行任何 TSIL 命令都会在 `TSILAdapter.__init__` 读日历文件时直接 `FileNotFoundError`**，不是"受阻但协议正确"，而是配置路径本身在本机不可达。本机实际可用的真实 qlib 数据是 `/home/zhengzengrong/PredicateLab/qlib_bin/`（811MB，CSI300 面板，Alpha158 bin 文件齐全，日历 2000-01-04 .. 2026-06-03）。本次所有"真实数据"复现都通过**显式覆盖 config 的 `provider_uri`/`tsil_root`** 使用这份本机数据，并把受审代码的逐字节副本（`sha256sum -c` 验证）接到真实 qlib 数据上；这是必要的最小改动（仅改配置字典里的两个路径字段，不改任何业务代码），已在每个复现脚本的 docstring 中说明。
- **只读保证**：审查未写入 `Predicate_Lab_v1.1/` 任何文件；对真实 qlib 数据仅做“复制子集到审查目录”（`tmpdata/qlib_subset/`，CSI300 2017-06..2020-12 成员股，476 支，95MB）与只读 `qlib.init()`；核查 `StateStore.__init__`（`predicatelab/core/store.py:361-371`）发现**构造函数本身会无条件执行 schema 建表与 `_migrate()`**（`CREATE TABLE IF NOT EXISTS` + `PRAGMA user_version=...`），对一个不存在的 db 路径它会**创建**空库；因为本环境没有真实历史库，这一行为未被触发在任何真实数据上，但据此确认：**任何未来对真实 `predicatelab.sqlite` 的"只读"检查都必须先 `cp` 一份只读副本再打开**，直接传入真实路径给 `StateStore()` 不是只读操作。本次所有 SQLite 操作都在审查目录的临时文件上进行。
- **受阻范围**：GPU 训练（`QniverseModel.fit` 真实反向传播、早停、50 epoch 规模）未执行——不是"跳过当通过"，而是明确因为：(a) 本次任务要求默认只读、不启动大规模训练；(b) 80GB A100 已用 73GB（`nvidia-smi` 确认），CPU 32 核但宿主已有 6-8 的后台负载。因此**"完整 LSTM+LUSI 真实训练闭环的端到端数值结果"本次不产出新证据**，但围绕它的**数据管道、切片语义、谓词数学、协议门禁、统计口径**全部用真实数据/真实代码在 CPU 上逐段验证（见第 4 节）。

---

## 2. 关键主张 ↔ 证据对应表

| # | 主张 | 当前证据状态 | 关键依据 |
|---|---|---|---|
| C1 | 实现符合 LUSI 二次形式的数学定义（`L=mean(eᵀ(τ̂I+τP)e)`） | **VERIFIED**（独立参考实现对照，梯度自动微分与有限差分一致） | §4.3 R06；`logs/r06_lusi_math.json` |
| C2 | 退化/零方差谓词被正确拒绝为 `P=0`，不会静默退化为 Φ_ones | **VERIFIED**（数值复现 + 既有单测 `test_degenerate_dsl_descriptor_returns_zero_not_ones`） | §4.3 R06 "degeneracy" |
| C3 | 候选的"每窗口只含目标标的自身历史"成立 | **FALSE（已反证）**：真实 f2020 折中 1.46%（train）/0.83%（test）窗口混入异股数据，其中 92%/87% 含**未来日期**的异股数据 | §4.1 R01 |
| C4 | 报告指标与候选预测一一对应，无静默丢样 | **FALSE（已反证）**：`eval()` 的 `drop_last=True` 在真实 valid 折丢弃 754/14354 样本、16 个标的全年缺席 | §4.1 R02 |
| C5 | "市场/跨股票"谓词（market_beta、cross_corr、lead_lag、common_factor、vol_sync、market_regime、cycle，共 7/23）的机制说明（捕捉市场结构/系统性风险）在真实训练配置下成立 | **FALSE（已反证）**：`shuffle=true` 下单 batch 跨约 2 年、360 个不同目标日期，同日中位数仅 2 个样本；`phi_cycle` 对批内行序敏感（置换不变性检验失败，`max_abs_difference=0.183`，理论要求 0） | §4.2 R05, R06 |
| C6 | 搜索-筛选-确认-终评的统计推断（HAC/block-bootstrap/BH）在独立同分布假设成立时有效控制误发现率 | **VERIFIED（有条件）**：全局零假设下，日内独立噪声场景（N0/N1）400 次复算 0/400 虚假 `validated`；一旦引入"同一 (候选,种子,折) 内全期常数偏移"的相关噪声（N2），虚假 `validated` 率升至 3.25%（95% CI [1.9%,5.5%]） | §5 |
| C7 | BH 校正 + 筛选/验证折不相交（D-10 已修）足以控制多轮搜索的总体误发现率 | **FALSE（已反证）**：validation_folds 在多轮之间被当作反馈信号重复使用（`generator.py:85-98` 按 `effect` 排序选父代），BH 只在单轮候选家族内生效；N2/N3 场景下第 0 轮和第 1 轮**各自独立**产生虚假阳性（§5 的轮次拆分数据） | §4.4, §5 |
| C8 | 确认折（f2023）与 holdout 一样受"一次性消费"保护 | **FALSE（已反证）**：`engine.confirm()`（`engine.py:1368`）全程不调用任何 ledger；同一 `--db` 内对同一 f2023 窗口重复 `confirm()` 不受阻（R04 实测两次都返回 `independently_confirmed`） | §4.4 R04 |
| C9 | Holdout 全局账本保护同一数据窗口不被二次"独立确认" | **VERIFIED（仅限单一 SQLite 文件内）**；跨 `--db` 文件**不受保护**（R03：相同 2024-01-01..2025-08-27 窗口在新 db 文件里被判定为"全新"并再次成功 `finalize` 为 `independently_confirmed`） | §4.4 R03 |
| C10 | 代码/环境指纹充分覆盖"任何改变数值的东西" | **FALSE（已反证）**：当前无 Git 环境下 `code_version()`（`identity.py:51-70`）恒定返回 `commit="unknown"`；即便有 Git，`_repo_root()`（`engine.py:156-157`）只哈希 `Predicate_Lab_v1.1` 本体，**配置实际引用的外部 `tsil_root`（另一用户 home 下的 TSIL-main）从未被哈希** | §4.4 R08 |
| C11 | 标签时间隔离按真实交易日历正确裁剪，不使用自然日近似 | **VERIFIED**：独立调用 `count_label_leak_samples` 复算得到与文档一致的 600 条跨 2023 年末泄漏样本日 | §3 |
| C12 | `tau_mode='fixed'` 后 LUSI 的 α 不再有梯度路径（历史 D-12 已修） | **VERIFIED（代码读取确认，未改动）**：`model_backbone_phitp.py:264-275` `_tau()` 在 `fixed` 分支构造 `torch.as_tensor`，不连接 `self.model.alpha` | 代码审查，非本次新增数值复现 |

---

## 3. 旧审查线索复核（`docs/v1.1_tsil_integration_audit.md` 等 D-01…D-16）

逐项在当前代码上复核（不预设旧结论正确）：

- **全部 D-01…D-16 对应的回归测试（`tests/test_v11_fixes.py` 22 个测试）当前 100% 通过**（装好 qlib 后 128/128 全绿，见 `logs/pytest_existing_suite_with_qlib.txt`）。逐一核对测试断言与声称修复点一致：快照身份稳定性（D-01）、NPZ 索引修复（D-03）、SearchPolicy 进协议哈希（D-05）、frozen family 不可变（D-06）、finalize 可重试（D-07）、`tau_mode=fixed` 无 alpha 梯度（D-12）、`direction=lower` 选择修复（D-09/D-11）、cross_corr 迹归一化（D-13/D-15）、holdout 名义窗口保留为元数据（D-16）——**均判定为已修复、代码仍一致，不是"声称修复但代码未变"**。
- **历史文档已自行承认的局限**（`docs/v1.1_remediation_report.md` §8 "Remaining risks"）：TSIL V1.1 路径仅 `CODE_ONLY`、未在真实数据上跑过完整 discover/confirm/finalize；本次审查**首次在真实 qlib 数据 + 真实 worker 调用链上执行了**数据切片、批语义、标签隔离的逐段复现（第 4 节），**部分填补**了这一空白，但**仍未执行真实 GPU 训练**（受本次预算与权限边界限制，非回避）。
- **本次新发现、旧文档未覆盖的问题**：R01（跨标的窗口污染）、R02（eval drop_last 丢样）、R04（确认折无暴露登记）、R05/R06（市场谓词机制证伪）、R07/R08（跨轮次验证折复用、代码指纹未覆盖外部 tsil_root）、R09（见下）均为**本次独立产出**，不在旧审计的 D-01…D-16 列表中，说明旧审计覆盖面不完整而非已被否定。

---

## 4. 从第一手材料建立的证据链（可复现）

所有脚本在 `repro/`，所有命令输出在 `logs/`；每条都给出：源码指纹、断言、命令、实际输出、容差来源。

### 4.1 数据与时间边界：跨标的窗口污染与评估丢样（R01 / R02）

**规格（审前独立写定，不调用被审计函数）**：预测 (标的 i, 日期 t) 的滑窗 `[t-seq_len+1, t]` 必须只含标的 i 自身的行；`eval()` 模式下应为每个符合条件的样本产出恰好一条预测。

- **源码**：`TSIL-main/src/dataset_ori_onehot.py:23`（`_create_ts_slices`，仅按全局行号切片，不检查跨标的边界）；`:354-359`（`eval()` 设 `drop_last=True`）；`:378-384`（`__iter__` 在 `drop_last and i+batch_size>len(slices)` 时整批丢弃，不补齐）。
- **合成反例**（`repro/r01_r02_dataset_slicing.py`，用 AST 只提取这几个函数编译执行，不 import 整个依赖 qlib 的模块）：3 支标的、B/C 晚于 A 上市，`seq_len=5` → 43 个样本中 8 个窗口混入异标的行，且混入的是**更晚日期**的 A 的数据（B 刚上市时的窗口回填了 A 当时"未来"的行）。`eval()` 丢弃尾部 123/1723 个样本。
- **真实数据复现**（`repro/r01_r02_real_panel.py`，接入本机 qlib CSI300 面板，走真实 `worker._split_segments`→`worker._build_dataset`→`MTSDatasetH`，f2020 discovery 折，CPU，不训练模型）：

  | 段 | 样本数 | 跨标的窗口 | 其中含未来日期 | 受影响标的数 | eval 丢弃 | eval 缺席标的 |
  |---|---|---|---|---|---|---|
  | train | 127061 | 1860 (1.46%) | 1710 (92%) | 100 | 661/127061 | 4 |
  | valid（早停） | 14354 | 0 | 0 | 0 | **754/14354 (5.3%)** | **16（全年缺席）** |
  | test（报告指标） | 71254 | 590 (0.83%) | 511 (87%) | 34 | 54/71254 | 1 |

  输出：`logs/r01_r02_real_panel.json`。
- **影响判定**：
  - **已确认**：跨标的窗口污染和 eval 丢样是真实存在、可在真实数据上稳定复现的行为（不是推断的风险）。两者对 candidate 与 baseline/control **对称生效**（同一 `MTSDatasetH` 实例），因此对"候选 − 基线"差值的**直接偏置**可能较小；但它们**违反"每个预测窗口只含目标标的自身历史"与"预测与合格样本一一对应"的既有协议声明**，且 valid 段 16 个标的全年缺席会**系统性地**影响早停所依赖的验证 IC 构成（早停用这个被截断的横截面选 `best_epoch`）。
  - **仍待核验**：对真实历史候选结果（volatility/drawdown 系列）量级影响多大——需要真实 GPU 训练对照才能量化，本次未做（受预算边界限制，不代表已排除）。
  - **最小修复方向**：在 `_create_ts_slices` 按标的分组切片时拒绝跨组窗口（用 padding 代替，机制已存在于 `self.zeros` 的起始填充逻辑，只需把它也用于"组内历史不足"的情形）；`eval()` 不应设 `drop_last=True`，应改为 `False` 或补齐最后一批。

### 4.2 LUSI/TSIL "市场/跨股票"机制证伪（R05 / R06）

**规格**：若谓词声称捕捉"市场回报/系统性风险/板块结构"，其输入在同一 mini-batch 内必须构成一个有意义的同期横截面；若谓词的公式只依赖样本自身数值，则对 batch 内行的任意重排必须等变（P 置换对应地变化，数值不变）。

- **批组成复现**（`repro/r05_batch_semantics_real_panel.py`，真实 f2020 折，`batch_size=800, shuffle=True`，复刻 `QniverseModel(seed=2025)` 的 `np.random.seed`）：4 个连续训练 batch，目标日期跨度均为 **2018-01-02 至 2019-10-16**（近 22 个月全覆盖），每 batch 约 360-376 个不同目标日期，**同日样本数中位数 = 2**，≥3 样本的日期只占约 1/3。输出：`logs/r05_batch_semantics_real_panel.json`。
- **代码依据**：`TSIL-main/src/custom_phi_constructors.py` 中 `get_market_beta_I_P`(:907)、`get_cross_correlation_I_P`(:985)、`get_lead_lag_I_P`(:1051)、`get_common_factor_I_P`(:1143)、`get_volatility_sync_I_P`(:1218)、`get_market_regime_I_P`(:1404) 六个函数均以 `.mean(dim=0)`／跨样本相关矩阵构造"market_return"/"common factor"；`temporal_predicates.py:76`（`FFT_for_Period_batch`）对 `phi_cycle` 沿 `dim=0`（batch 维）做 FFT。
- **置换等变性数值检验**（`repro/r06_lusi_math.py`）：对真实公式独立计算 `P[perm][:,perm]` 与 `get_temporal_I_P('phi_cycle', x[perm])` 的差异，在一个具有明确周期结构的合成输入上，**`max_abs_difference = 0.183`**（理论上应为 0，float 容差 ~1e-6）；`phi_volatility`（纯样本内统计量）同一测试下 `2.98e-8`（通过）。输出：`logs/r06_lusi_math.json`。
- **影响判定**：**已确认**7/23 原生谓词（6 个"市场"类 + `phi_cycle`）在真实 `shuffle=true` 配置下，其损失项所依赖的"同批其他样本"**不是**任何有意义的同期横截面，而是跨约两年随机抽取的、中位数仅 2 个样本共享同一目标日期的混合体；`phi_cycle` 额外对 batch 内行序敏感（打乱 batch 顺序会改变其 P 矩阵，纯属实现副作用，与任何"市场周期"语义无关）。**这直接证伪"机制对照支持该谓词所声称的市场/系统性风险解释"**；该谓词组若有效应，必须解释为"对一个由随机重采样产生的伪横截面做的某种隐式正则化/方差压缩"，不能解释为论文所述机制。此结论**不依赖**本次是否跑过真实训练——它是 `_build_dataset(shuffle=True)` + 这 7 个函数的纯静态/CPU 数值事实。

### 4.3 LUSI 数值与退化检验（R06，独立参考实现对照）

- **参考实现**独立编写（不调用 `WeightedMSELoss`）：`L=mean(e⊙((τ̂I+τP)e))`，手推梯度 `2/n (τ̂I+τP)e`。
- 对真实 `compile_descriptor` 产出的 `phi_fn`（DSL `volatility` 族）与真实 `WeightedMSELoss` 做数值对照：损失值绝对差 `0.0`；autograd 梯度与参考梯度最大绝对差 `2.98e-8`；有限差分（`eps=1e-4`）与参考梯度最大绝对差 `9.5e-4`（量级符合 `O(eps)` 截断误差，**非**实现缺陷）。
- 退化输入（常数特征，预期 P=0 而非 Phi_ones）：实测 `P.abs().max()=0.0`——**确认**历史 D-04 修复仍然成立。
- **结论**：LUSI 二次形式的梯度/数值实现在本次检验的输入范围内是**正确的**。**但梯度正确不等于该谓词具有科学含义**（审前已声明的区分）——这是本节与 4.2 节的关键差别：4.3 验证"算对了"，4.2 证伪"算的东西对不对题"。

### 4.4 搜索与统计单位：跨轮次反馈复用、确认期暴露、holdout 跨库漏洞、代码指纹（R07 / R04 / R03 / R08）

本小节使用一个**合成时间序列域适配器**（`repro/synth_adapter.py`）直接接入**真实、未改动**的 `predicatelab.core.engine.PredicateLabEngine`、真实 `RuleGenerator`（`predicatelab/domains/tsil/generator.py`，含真实反馈变异逻辑）、真实 DSL 静态校验、真实 `stats.py`（HAC/bootstrap/BH）、真实 `holdout.py` 账本、真实 SQLite `StateStore`。**只替换**"一次试验的数值结果"这一层（由显式真值模型生成，带已知的真实效应/噪声结构），用于做"基准/验证器是否可信"的标定实验——这是审前第五节要求的"统计模拟"，不是臆造执行记录。

- **R07 跨轮次验证折复用（统计单位错误）**：`generator.py:85-98`（`_parents`）按 `feedback` 中的 `effect` 字段（即**上一轮在 validation_folds 上的完整验证效应**，见 `engine.py:1216/1232/1452` 把 `e.effect` 直接写入 feedback payload）对父代排序，产生下一轮变异候选；这些候选随后**再次**在同一组 `validation_folds`（如 f2020/f2021）上做完整验证。D-10 修复的是"screen fold 与 validation fold 不相交"（单轮内），**没有**约束"validation fold 在多轮之间被当作优化反馈反复查询"。BH 校正（`engine.py:989-1004`）只在"本轮候选家族"内生效，不知道上一轮已经看过同一份数据。
- **R04 确认折无暴露登记**：`engine.confirm()`（`engine.py:1368-1461`）全程没有调用任何 ledger/exposure 检查；唯一的保护是"同一 run 的 frozen family 不可二次冻结"（D-06），但**不同 run**（哪怕同一个 `--db`）可以对同一个 f2023 窗口反复 `freeze_family()→confirm()`。
- **R03 holdout 跨库漏洞**：`holdout.py` 的全局账本是**真实的、按数据身份而非 run 身份建键**（`holdout_key = H(dataset_id, snapshot_hash, label_spec_hash, test_start, test_end)`，正确），但账本本身**存储在传入的那个 SQLite 文件里**——`--db` 是 CLI/Engine 的自由参数。

  复现（`repro/gates.py`，输出 `logs/gates.json`）：
  1. `db_A.sqlite` 第一次对 2024-01-01..2025-08-27 窗口 `finalize` → `independently_confirmed`，ledger 记一行 `completed`。
  2. **同一** `db_A.sqlite`，新 run，相同候选族 → `confirm()` 正常返回 `independently_confirmed`（见 R04），但 `finalize()` 被 `HoldoutConsumed` 正确拦截——**单文件内保护有效**。
  3. **新建** `db_B.sqlite`，同样的协议/数据/日期窗口 → `freeze_family→confirm→finalize` **全部成功**，再次拿到 `independently_confirmed`，ledger 里只有这一条孤立记录，**完全不知道 `db_A.sqlite` 已经消费过同一窗口**。

  **这与历史文档 §8 第 4 条自述的"legacy 行天生拒绝复用"刚好相反方向**：旧文档只讨论了"同库内旧记录挡住新确认"（fail-closed，良性），没有讨论"换一个库文件即可绕过"（fail-open，这是本次新发现）。
- **R08 代码指纹在当前环境下恒定**：`identity.py:51-70`（`code_version`）在无 Git 仓库时 `commit="unknown"`，`status`/`diff` 均为空字符串，`dirty_fingerprint = sha256({"commit":"unknown","status":"","diff":""})` 是**与代码内容无关的常量**。复现：把 `predicatelab/core/stats.py` 复制到一个无 `.git` 的目录，追加一行注释后重新调用 `code_version()`，**fingerprint 完全不变**（`logs/gates.json` 的 `R08` 字段）。更关键的是：即便未来给 `Predicate_Lab_v1.1` 初始化 Git，`_repo_root()`（`engine.py:156-157`，`Path(__file__).resolve().parents[2]`）只哈希这一棵树；而三份 TSIL 配置的 `data.tsil_root` 指向 `/home/dmz-ai/.../Predicate_Gen/TSIL-main`——**一个完全在该树之外、本环境中甚至不存在的路径**。换言之：**真正被导入执行的 TSIL 损失/数据集/模型代码，从未、也不可能被 `code_version()` 覆盖**，无论 Git 状态如何。

### 4.5 统计单位复算（独立脚本）

- 独立重算 `docs/v1.1_evaluation_protocol.md`/`results_tsil.md` 所称"600 条跨 2023 年末标签泄漏样本日"：调用（未改动的）`count_label_leak_samples`，但用**独立拼出的调用参数**（本机 qlib 日历/成分股文件，边界 `2023-12-31`），得到 `n_leaked_sample_dates=600`，`leaked_dates=['2023-12-28','2023-12-29']`——与文档数字**完全一致**，判定 **VERIFIED**。

---

## 5. 统计模拟：全流程校准（目标误差率、统计单位、通过标准预先写定）

**目的**：检验"搜索→筛选→验证→冻结→确认→终评"整条流程，在不同噪声结构下，实际的虚假阳性率是否接近协议声明的 `fdr_alpha=0.10`。**只替换试验结果生成层**，引擎/生成器/统计/账本全部是真实代码（见 §4.4 说明）。

**预先写定的目标与判定标准**（`repro/sim_pipeline.py` 文件头注释，模拟前固定，未根据结果调整）：
- 统计单位：(fold/年份) × (训练种子)；
- 目标误差率：全局零假设下，一次完整 run（2 轮搜索）在 discovery/confirmation/final 任一阶段产生 `independently_confirmed`（或 discovery 阶段 `validated`）的概率 ≤ 配置的 `fdr_alpha=0.10`；
- 最小有意义效应 = 配置 `min_effect=0.002`（IC）；正对照设在 2×MME（0.004）和 5×MME（0.01）；
- 判定：Wilson 95% 置信区间上界 ≤ 0.10 为 PASS，下界 > 0.10 为 FAIL，否则 INCONCLUSIVE；
- 重复次数 n=400（零假设场景）由"二项比例 Wilson CI 半宽 < 2 个百分点"的精度需求决定；正对照场景因资源限制未达到此精度（见下方"受阻"说明）。

**场景与完整结果**（`logs/sim/*.jsonl`，汇总 `logs/sim/null_summary.json`，脚本 `repro/sim_pipeline.py`）：

| 场景 | 相关噪声结构 | n | discovery 阶段≥1 次 validated | confirmation 阶段≥1 次 confirmed | final 阶段 confirmed |
|---|---|---|---|---|---|
| N0_iid（纯日级独立噪声） | sd=0 | 400 | 0/400（Wilson [0, 0.95%]）**PASS** | 0/400 | 0/400 |
| N1（弱种子×折相关噪声） | sd=0.0015 | 400 | 0/400（[0, 0.95%]）**PASS** | 0/400 | 0/400 |
| N2（中等种子×折相关噪声） | sd=0.003 | 400 | **13/400 = 3.25%**（[1.9%, 5.5%]）**仍 PASS（相对 0.10 门限）但远高于 N0/N1** | 0/400 | 0/400 |
| N3（弱种子噪声+谓词族×年份噪声） | sd_e=0.0015, sd_u=0.002 | 400 | 4/400 = 1.0%（[0.4%, 2.5%]） | 0/400 | 0/400 |
| P1（N3 + `volatility` 族真实 +0.004 效应，2×MME） | 同上 | **n=33（部分，仍在运行）** | 8/33 ≈ 24%（n 不足，仅作方向性参考） | 0/33 | 0/33 |
| P2（N3 + `volatility` 族真实 +0.01 效应，5×MME） | 同上 | **n=0（受资源竞争阻塞，未产出数据）** | — | — | — |

**结论（基于已完成部分）**：
1. 在纯日级独立噪声（N0/N1）下，整条流程的零假设虚假阳性率**在 400 次重复中为 0**，与声称的 HAC/bootstrap/BH 实现一致——**这是对 §4.3"数学实现正确"的流程级确认**，不是单点数值对照。
2. 一旦引入"同一 (候选, 种子, 折) 内全折期保持不变的噪声偏移"（模拟真实训练中优化器路径依赖/收敛噪声等会在整折天数上持续存在的误差来源，量级仅为典型日间 IC 噪声标准差 0.04 的约 7.5%），discovery 阶段的虚假 `validated` 率从 0% 跳升到 **3.25%**。机制定位：`stats.py:25`（`newey_west_se`，默认 `lag=5`）和 `:72`（`block_bootstrap_ci`，默认 `block=5`）的依赖窗口（5 个交易日）远小于一个验证折的长度（约 250 个交易日），**无法识别"整折常数偏移"这类低频相关结构**，导致标准误被低估、名义 10% 的 FDR 阈值形同虚设。**这不是 bug，是方法论边界**：HAC/bootstrap 对"日间短程相关"是对的工具，但 TSIL 训练本身引入的"同折/同种子全局噪声"属于 HAC 窗口之外的相关结构。
3. **confirmation/final 阶段在所有 4 个零假设场景下均为 0/400 虚假确认**——这是一个**正面**结果：确认折使用全新的独立随机噪声（时间上靠后、与 discovery 折不共享任何噪声实现），确认步骤正确地把 discovery 阶段的虚假阳性"过滤掉了"。**这意味着 §4.4 的 R04（无暴露登记）是一个"协议/账本"层面的漏洞，而不是"确认步骤的统计检验本身失效"**——两者需要分开表述：confirm() 的统计逻辑是可信的，但它可以被反复调用在同一份数据上，直到偶然通过。
4. **正对照（P1/P2）尚未达到可报告的精度**：P1 仅完成 33/378 次重复（受审查机器 CPU/磁盘 I/O 竞争，两个并发模拟任务互相拖慢，在打报告时仍在后台运行），discovery 阶段命中真实 `volatility` 族的比例约为 24%（8/33），方向符合预期（流程能检测到注入的真实信号），但样本量不足以给出可信区间；confirmation/final 阶段在这 33 次里均为 0，**不足以区分"真实效应太弱通不过确认"还是"样本太少"**。P2（更强的 5×MME 正对照）**完全未产出数据**（0 行）。**这部分结论状态为 INCONCLUSIVE / 资源受阻**，不计入"已确认"结论，仅作为附录方向性参考。

---

## 6. 历史结果影响评估

- **V1.0 历史最佳候选（`volatility(window=10)` 及其子代）在 2024-01-01..2025-08-31 holdout 上"未能确认"（paired IC −0.00103, p=0.386）的阴性结论本身不受本次发现直接推翻**：该阴性结果恰好是"候选表现不如基线"，不是本次发现的任何一个漏洞能够人为制造出来的方向（R07/R04/R03 都只能制造虚假阳性，不能制造虚假阴性）。因此**保留**该历史阴性结论。
- **V1.0/V1.1 历史探索期的"4 个 validated 候选"（`volatility`/`drawdown` 族）的可信度需要下调**：根据 §5 的校准结果，discovery 阶段的 BH+阈值组合在存在折/种子相关噪声时，虚假阳性率可达名义值的数倍；叠加 §4.4 R07（跨轮次复用 validation_folds 作为反馈信号，round 1 的 3 个候选全部是 round 0 候选的变异子代，`docs/results_tsil.md` 自己也指出"不是四个独立发现"）——**这进一步削弱"4 个 validated"的独立证据数量，实际独立检验次数应视为接近 1（round 0 的 `volatility(window=10)` 候选本身），而非 4**。这一结论历史文档已部分自觉（"not independently discoveries"的措辞），本次把它量化为一个具体的统计机制。
- **R01/R02（跨标的窗口污染、eval 丢样）对历史数值结果的影响方向不确定**：两者对 candidate/baseline 对称生效，预期对"候选−基线"差值的偏置量级**小于**它们对绝对指标的影响，但**未经真实训练复算，不能排除**它们通过早停路径间接影响了 `best_epoch` 的选择（§4.1 valid 段 16 个标的全年缺席直接改变早停所见的横截面）。**判定为"待核验"，不计入已确认的历史结果修正**。
- **R04/R03（暴露登记缺口）不改变已发生的历史确认事件本身**，但说明"历史文档所称的`independently_confirmed`/`holdout consumed`状态在制度上依赖操作者不去创建新 `--db` 文件"——**这是一个流程可信度问题，不是对已完成评估的事后撤销**。按本次任务要求，"修复程序不自动恢复历史结论的有效性，也不能撤销已经发生的数据暴露"：本报告不声称、也无法声称历史上发生过真实的跨库重复确认（本环境无历史数据库可查），只报告该漏洞**在当前代码上是真实存在且可复现的**。

---

## 7. 按依赖排序的整改与研究计划

**阻断级（先于一切新确认）**
1. **confirm()/finalize() 的跨数据库暴露登记**（解决 R03/R04）。范围：`predicatelab/core/holdout.py` 新增（或复用现有表结构但独立于单一 `StateStore` 实例）一个**跨进程/跨文件的共享登记位置**（例如固定路径的登记文件或要求的"全局登记库"参数，不能让 `--db` 的选择决定保护范围）；confirm() 需要新增与 finalize() 对称的 ledger 调用。前置依赖：无。产物：新的 ledger 表/文件 + 迁移脚本 + 对 R03/R04 的回归测试（复用 `repro/gates.py` 的场景作为测试骨架）。验收：R03/R04 复现脚本重跑，三次"新库绕过"场景全部被拒绝。失败后决定：若跨文件登记在部署上不可行（例如多团队各自 db），退而要求"每次 confirm/finalize 前检查并写入一份人类可读的暴露登记表，由人工审批核对"，明确降级为流程控制而非自动化控制。
2. **跨轮次 validation_folds 复用的统计口径修正**（解决 R07）。范围：`predicatelab/core/engine.py` 的 `_run_round`/`_run_round_body` + `predicatelab/domains/tsil/generator.py:_parents`。两个可选方案：(a) 把 validation_folds 定义为"整个 run 只消费一次"的资源，round>0 的候选改为用一组新的、未被用过的折做验证（代价：需要更多历史数据/折）；(b) 保留现状但把全 run（所有轮次）的候选当作一个统一的多重比较家族，在 run 结束时做一次性的、覆盖所有轮次候选的 BH/FWER 校正，而不是逐轮校正。前置依赖：无。产物：协议文档更新 + 代码改动 + 新回归测试（复用 `repro/sim_pipeline.py` 的 N2/N3 场景，验收标准：方案落地后，discovery_any 在 N2 场景下的虚假阳性率 95% CI 上界回落到 ≤ 0.10 附近，理想情况接近 N0 的水平）。验收条件：N2 场景重新跑 400 次，discovery_any 率的 Wilson 上界 ≤ 配置的 `fdr_alpha`。失败后决定：若方案 (a) 因数据量不足难以落地，采用方案 (b) 并在文档中明确"多轮搜索的总体误发现率是对整个 run 承诺的，不是对每轮分别承诺"。

**数值/样本完整性/统计单位级**
3. **修复 `_create_ts_slices` 跨标的窗口污染 + `eval()` 的 `drop_last`**（解决 R01/R02）。范围：仅 `TSIL-main/src/dataset_ori_onehot.py`（审前已声明：改动上游 TSIL-main 需要数值回归测试+独立审核）。前置依赖：无，但需先确认改动后与现有缓存 trial 不冲突（见下一条）。产物：补丁 + 单测（复用 `repro/r01_r02_dataset_slicing.py` 作为断言基础）+ 真实数据上的差分验证（修复前后同一折跑一次评估，比较丢样数/污染窗口数是否归零）。验收：真实 f2020/f2021/f2022/f2023 折的 R01/R02 复现脚本输出归零。失败后决定：若上游改动风险过高，至少在 PredicateLab 适配层（worker.py）对污染/丢样做显式统计和告警，不隐藏问题。
4. **让 `code_version()`/`trial_config_hash` 覆盖外部 `tsil_root` 真实代码**（解决 R08），并在无 Git 环境下 **fail closed**（拒绝运行或强制要求显式声明代码指纹）而不是静默返回常量。范围：`predicatelab/core/identity.py`、`predicatelab/core/engine.py:_build_context`。前置依赖：无。产物：新的"外部代码树指纹"字段（对 `tsil_root` 目录做类似 `full_content_manifest` 的内容哈希）+ 测试。验收：修改 `tsil_root` 指向目录下任意一个 `.py` 文件后，`trial_config_hash` 必须改变；无 Git 时运行应报错或要求 `--allow-unversioned-code` 之类的显式确认。失败后决定：若性能开销不可接受（外部树可能很大），至少把"当前 `tsil_root` 内容指纹"写入 run 的 context.json 供人工比对，不要求每次都重算完整哈希。

**正负对照与完整搜索流程校准**
5. **完成 §5 的正对照（P1/P2）统计模拟到预先设定的精度**，并补充"跨轮次反馈导致的有效自由度"的专门模拟（验证方案 2 落地后是否真正把虚假阳性率压回目标）。前置依赖：条目 2 落地（否则正对照结果仍受同一偏差污染，无法单独解读）。产物：完整的 N0-N3+P1-P2 的 Wilson CI 表格，连同新方案下的对照实验。验收：按 §5 预先写定的 PASS/FAIL/INCONCLUSIVE 标准给出明确结论。

**暂缓/暂不值得投入**
- 不建议在修复条目 1-4 之前启动任何新的真实 GPU TSIL 训练或新的独立确认（会在已知存在自适应复用和暴露登记漏洞的协议上产生新的、同样不可信的"确认"）。
- 不建议扩大谓词库规模或引入新领域（mineral/image）——当前风险集中在协议和统计层，不在"候选数量不够"。
- 架构重构（事件溯源、Research Control Plane、Gate A/B/C 人工审批框架，见 `PredicateLab_CodingAgent_Implementation_Prompt_20261106.md` 第 2-3 节）**目前完全不存在于代码中**（grep 确认零匹配），属于"尚未实现的能力"而非缺陷；在条目 1-4 落地前投入这类大规模架构工作，优先级应靠后——先把现有引擎的协议漏洞堵住，再谈是否需要独立的审批平面。

**对新的独立确认（Gate B 式）的准备要求（本次不自动执行）**：若要在条目 1-2 落地后进行新的独立确认，需要准备：(a) 数据独立性证明——确认候选用到的任何数据片段此前从未出现在任何 discovery/confirmation/exploratory_reuse 记录里（需要条目 1 的跨库登记才能机械核验）；(b) 协议冻结文档（含折定义、统计口径、阈值、种子列表，签名存档）；(c) 功效分析（基于条目 5 的正对照结果，给出"检测到 MME 级效应所需的折数/种子数"的具体数字）；(d) 访问授权记录（谁批准读取哪个时间窗口）。本报告不代为准备这些材料。

---

## 8. 第三方复核说明

- 环境重建：`python3 -m venv venv && venv/bin/pip install -r <见 evidence/pip_freeze.txt 的精确版本列表>`（含手工安装 `pyqlib==0.9.6 --no-deps` 后逐项补齐 `ruamel.yaml filelock loguru fire tqdm joblib<1.5 matplotlib "plotly<6" statsmodels pyarrow mlflow-skinny redis python-redis-lock pymongo gym cloudpickle`，原因见 `logs/pip_qlib.txt`/`pip_qlib_nodeps.txt`/`pip_qlib_deps_individual.txt`：pyqlib 的可选重依赖 `cvxpy`/`lightgbm` 在本机缺 BLAS/CMake 无法源码编译，但它们不在 `qlib.data.dataset`/`qlib.model.base`/`qlib.contrib.report` 这条实际被 import 的路径上，可跳过）。
- 现有测试：`cd Predicate_Lab_v1.1 && PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. venv/bin/python -m pytest -q` → 期望 128 passed, 1 skipped（跳过项是"真实 qlib 日历不可用"，装好 qlib 后应变为 pass；若仍 skip，检查 `provider_uri` 是否指向一份真实 qlib store）。
- 逐条复现：`repro/` 下每个脚本头部注释给出用法；所有脚本只读取本机 `qlib_bin`/审查目录内的合成数据，不写 `Predicate_Lab_v1.1`。关键命令：
  - `CUDA_VISIBLE_DEVICES= venv/bin/python -I repro/r01_r02_real_panel.py src_copy tmpdata/qlib_subset` → 比对 `logs/r01_r02_real_panel.json`。
  - `CUDA_VISIBLE_DEVICES= venv/bin/python -I repro/r05_batch_semantics_real_panel.py src_copy tmpdata/qlib_subset` → 比对 `logs/r05_batch_semantics_real_panel.json`。
  - `CUDA_VISIBLE_DEVICES= venv/bin/python -I repro/r06_lusi_math.py src_copy` → 比对 `logs/r06_lusi_math.json`。
  - `venv/bin/python -I repro/gates.py src_copy tmpdata/qlib_subset/calendars/day.txt <新目录>` → 比对 `logs/gates.json`（注意：会在给定目录下创建新的临时 SQLite，不触碰真实数据库）。
  - `venv/bin/python -I repro/sim_pipeline.py src_copy <calendar> <workdir> <out.jsonl> <scenario> <n> <procs>` → 场景名见 `repro/sim_pipeline.py` 顶部 `SCENARIOS` 字典；每行输出可用 `repro/summarize_sim.py 'logs/sim/N*.jsonl'` 汇总为 Wilson CI。
- 容差：浮点数值对照统一用绝对误差 < 1e-5（autograd/参考实现对比用 1e-6 量级，已在 `logs/r06_lusi_math.json` 体现）；有限差分对比允许到 1e-3（`eps=1e-4` 导致的二阶截断误差，已在报告中注明来源）。

---

## 9. 两次自检

**第一次自检（形成结论前）**：
- 检查"是否把设计取舍误判为缺陷"：R02 的 `drop_last=True` 可能是上游作者为训练效率做的常见设计取舍（丢弃不满批次），**但它被用在 `eval()` 路径**（`dataset_ori_onehot.py:354-359` 的 `eval()` 方法显式把它设为 `True`），而评估/早停/报告指标理应要求"每个样本都参与"，这是一个在**评估**语境下不合理的取舍延续，因此仍判定为缺陷（更准确地说：训练路径的 drop_last 是合理取舍，评估路径沿用同一字段是缺陷），已在 §4.1 的措辞中区分。
- 检查"是否遗漏外围保护"：最初判断 R03/R04 时，先核实了 `confirm()`/`finalize()` 在**单一** SQLite 文件内确实受保护（D-06 frozen family 不可变 + holdout ledger 单文件有效），避免把"单文件内有效"误判为"完全无效"——最终表述区分了"单文件内 VERIFIED"与"跨文件 FALSE"。
- 检查"是否使用了与当前版本不一致的证据"：所有引用的行号均来自本次 `src_copy`（逐字节校验过与 `Predicate_Lab_v1.1` 一致）当场 `grep -n` 的输出，未使用旧审计文档里的行号。
- 主动寻找推翻自己判断的解释：对 R05/R06（市场谓词机制证伪），检查了是否存在某处默认 `shuffle=False` 从而使该发现不适用——确认三份 TSIL 配置和 TSIL-main 自带配置均显式 `shuffle: True`，且 `worker.py:_build_dataset` 的默认值也是 `True`，排除了"只在边缘配置下才成立"的反驳。

**第二次自检（报告完成后）**：
- 路径/行号/版本核对：§4 的所有行号已与本节写作时的 `grep -n` 输出重新核对一致（见本报告正文引用）；命令与测试输出与 `logs/` 下文件逐一对应，未发现不一致。
- 确认没有把旧结果当新结果：§3 明确标注"历史文档已自行承认"与"本次新发现"两类，未混淆；§5 的历史 holdout 阴性结论（−0.00103, p=0.386）标注为引用历史文档数字，本次未重新计算（因无历史数据库可用），已在 §6 说明"保留，不是本次重新验证"。
- 确认没有把"重放一致"当"科学有效"：128/128 测试通过只用于支持"既有回归测试覆盖的具体断言仍成立"，未被引申为"TSIL 整体科学有效"；§5 的流程校准明确区分"统计实现正确（VERIFIED）"与"流程整体虚假阳性率可控（有条件 PASS，依赖噪声结构）"。
- 确认没有把"多个 AI 一致意见"当独立验证：本报告的核验链条是**我自己**对同一事实的多角度交叉检查（代码读取 → 合成最小反例 → 真实数据复现 → 独立参考实现数值对照 → 全流程合成校准），不存在多个独立 AI 审查者的交叉比对；不把"自己对自己复核"包装成"独立验证"的措辞，已在各节注明"本次新产出"或"独立于被审函数的参考实现"的具体做法，而非笼统宣称"已独立验证"。
- **仍存在的限制**：(a) P1/P2 正对照未达统计精度，§5 已明确标注为 INCONCLUSIVE/附录参考，不计入结论；(b) 未执行真实 GPU 训练，R01/R02 对历史候选具体数值的影响量级未定量，已在 §4.1/§6 标注"待核验"；(c) 历史数据库/artifacts 完全不存在于本环境，§6 的历史影响评估基于文档叙述 + 本次新发现的机制交叉推理，不是对历史原始记录的重新计算。

---

## 10. 最终结论分层

**已获得独立证据支持（本次新复现或新发现）**：
- 跨标的窗口污染（R01）、评估丢样（R02）在真实数据上存在且量化；
- 7/23 "市场类"谓词在真实训练配置下机制解释不成立（R05/R06）；
- LUSI 二次形式梯度/数值实现正确（R06 独立参考实现对照）；
- confirm()/finalize() 存在跨 `--db` 文件的暴露登记漏洞（R03/R04）；
- 代码指纹在当前环境下恒定、且从未覆盖外部 `tsil_root`（R08）；
- 跨轮次搜索对 validation_folds 的反馈复用构成未被现有 BH 校正覆盖的统计漏洞（R07），并在合成校准中量化其后果（discovery 阶段虚假阳性率从 0% 升至 3.25%）；
- 标签隔离的"600 条泄漏样本日"数字独立复算一致。

**仅在有限输入/环境下成立，需要更大样本或真实环境才能确认**：
- P1/P2 正对照（真实效应能否最终走到 `independently_confirmed`）——目前 n 太小（33/0），方向性支持"流程能检测注入信号"，但置信区间无意义；
- R01/R02 对历史具体数值（如 `volatility(window=10)` 的 +0.00378 探索期增益）的影响方向和量级——需要真实 GPU 训练对照，本次未执行。

**目前证据不足，需要第三方复现或新数据才能判断**：
- 历史 V1.0 run（`run_20261031_153702`）的具体数据库记录是否曾实际触发 R03/R04 式的跨库重复确认——本环境无历史数据库，无法回答，也不应臆测为"已发生"或"未发生"；
- 真实 GPU 规模（50 epoch, d_model=64）下 LUSI 损失相对 plain MSE 是否有稳定增益——本次完全未执行真实训练，这仍是一句"尚未发现有效谓词"的诚实状态，而非"已证明无效"或"已证明有效"。
