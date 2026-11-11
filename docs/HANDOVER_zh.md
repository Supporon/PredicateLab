# PredicateLab V1.1 交付说明

> 面向接收本包的**研究者 / 工程师**。这份文档说明包内有什么、怎么跑起来、
> 结果怎么读，以及**哪些话现在还不能说**。更细的代码走读见
> [code_guide.md](code_guide.md)，规范性的协议见
> [v1.1_evaluation_protocol.md](v1.1_evaluation_protocol.md)。

## 0. 版本信息

| 项目 | 内容 |
|---|---|
| 仓库 | <https://github.com/Supporon/PredicateLab>（`main` 分支） |
| 代码基线 | PredicateLab V1.1，交付日期 2026-11-11 |
| 测试状态 | 本包在 Python 3.11.9 上 `190 passed`（无 qlib） |
| 开发环境 | README 记录为 Python 3.12 / pydantic 2.11 / numpy·pandas / scikit-learn 1.5 / torch / qlib 0.9.6 |
| 第三方代码 | `TSIL-main/` 为上游 TSIL 参考实现，**GPL-3.0**（见其 `LICENSE`） |
| 根目录许可 | 未附 `LICENSE`；再分发或商用前请与作者确认 |

## 1. 这是什么

PredicateLab 是一个**谓词发现与证据验证平台**：核心引擎与领域无关，具体数据
通过 domain adapter 接入。它把"搜索候选谓词 → 冻结候选族 → 独立确认 →
一次性 holdout 终评"做成有状态、可恢复、可审计的流水线，统计上使用
Newey-West HAC、moving-block bootstrap 和 Benjamini-Hochberg FDR。

包内包含四个领域适配器：

| 领域 | 状态 | 说明 |
|---|---|---|
| `tsil` | 可用（需 qlib + 数据） | 真实 CSI300、上游 TSIL LSTM/LUSI loss、23 个原生谓词 + 6 个时序谓词 |
| `tabular` | 可用 | 独立的 scikit-learn 适配器，不依赖 TSIL |
| `small_sample` | 可用 | 合成数据，有显式潜在真值，用于测量 I 类错误与功效 |
| `mineral` / `image` | 仅契约 | 描述符与适配器接口示例，执行路径抛 `DomainNotImplemented` |

## 2. 包里有什么、没有什么

**有：**

- `predicatelab/` — 核心引擎、统计、holdout、身份指纹、SQLite 状态库、各领域适配器
- `configs/` — 21 份配置（TSIL、tabular、small-sample、LLM 实验、tau sweep）
- `tests/` — 21 个测试文件（本包 190 项通过）
- `docs/` — 架构、代码走读（中文）、评估协议、兼容性、审计与修复报告、结果文档
- `scripts/` — 快照打包、绘图、证据校验、小样本正式研究分析脚本
- `published/` — 轻量结果快照：`results/`、`logs/`、`figures/`、`code/`，
  含 `MANIFEST.json` 与 `checksums.sha256`（约 18 MB）
- `TSIL-main/` — 上游实现副本（GPL-3.0）

**没有（有意排除）：**

- 真实行情数据、qlib 数据目录、任何 `*.sqlite` 历史库
- 模型权重（`*.pt`/`*.pth`）、逐 trial 的 `result.json`/`job.json` 明细
- `.env` 与任何 API key（LLM 功能需要你自己设置 `DEEPSEEK_API_KEY`）
- 原始 `artifacts/` 运行目录（`published/` 是它的轻量投影）

因此：**TSIL 的端到端训练不能开箱即跑**。你需要自行准备 qlib 0.9.6 与
CSI300 数据，并修改配置里的 `provider_uri` / `tsil_root`。tabular 与
small-sample 两个领域不需要 qlib。

## 3. 环境与安装

```bash
cd PredicateLab_v1.1_20261111
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # 核心依赖 + pytest
export PYTHONPATH=.
```

TSIL 场景另行安装（qlib 会锁定自己的 numpy/pandas 版本范围）：

```bash
pip install torch pyqlib==0.9.6          # 按你的 CUDA 环境选择 torch
```

`requirements.txt` 只列核心依赖；`pytest.ini` 关闭了 qlib 的可选
`fugue_test` 插件，避免它在配置阶段导入 `fsspec`。

## 4. 五分钟自检

```bash
export PYTHONPATH=.
python -m predicatelab.cli domains
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q      # 期望 190 passed
```

`domains` 应列出 `tabular`、`tsil`、`small_sample`、`mineral`、`image`。
small-sample 的快速冒烟（不需要 qlib）：

```bash
python -m predicatelab.cli validate-config --config configs/small_sample_smoke.yaml
python -m predicatelab.cli discover --config configs/small_sample_smoke.yaml --rounds 1
```

## 5. 目录导览

```text
predicatelab/
├── cli.py                 # 命令行入口
├── core/
│   ├── engine.py          # Probe → Screen → Full validation → BH FDR → Confirm → Finalize
│   ├── stats.py           # HAC t、block bootstrap、BH FDR
│   ├── holdout.py         # 一次性 holdout、确认门控、标签前瞻隔离
│   ├── store.py           # SQLite 状态库 + 全局暴露账本
│   ├── identity.py        # 源码树 / 数据的 SHA-256 指纹
│   └── artifacts.py       # 产物分类与快照
└── domains/
    ├── tabular/           # 结构化数据
    ├── tsil/              # 量化时序（qlib + GPU worker）
    ├── small_sample/      # 合成小样本
    └── mineral/ image/    # 仅契约
```

## 6. 一次完整流程

```bash
export PYTHONPATH=.
CFG=configs/tsil_lstm_smoke.yaml          # 换成你改好路径的配置

python -m predicatelab.cli validate-config --config $CFG
python -m predicatelab.cli discover      --config $CFG --rounds 1
python -m predicatelab.cli freeze-family --config $CFG --run-id run_...
python -m predicatelab.cli confirm       --config $CFG --run-id run_...
python -m predicatelab.cli finalize      --config $CFG --run-id run_... --candidate-id cand_...
python -m predicatelab.cli report        --config $CFG --run-id run_...
python -m predicatelab.cli evidence      --config $CFG --run-id run_...
python -m predicatelab.cli export        --config $CFG --run-id run_... --out artifacts/run_.../bundle
```

产物写入 `artifacts/run_<timestamp>/`：`context.json`、`report.json`、
`report.md`、`evidence.jsonl`、逐 trial 日志；`freeze-family` 追加
`frozen_family.json`，`confirm` 追加 `confirmation.json`，`finalize` 追加
`frozen_manifest.json` 与 `final_evaluation.json`，`export` 追加带
`checksums.sha256` 的不可变 bundle。

## 7. 结果怎么读

**看 `result.state`，不要只看退出码。** `finalize` 退出 0 只表示执行完成；
科学状态在载荷里，取值是 `independently_confirmed`、`resampling_confirmed`、
`insufficient`、`rejected`。

**未定义的统计量是 `null`，并记录在 `null_paths` 中，不是 NaN。** 不要对
NaN/None 做隐式填充或"0 即无效"的解释。

`published/<experiment>/README.md` 说明该实验发布/排除了什么；
`MANIFEST.json` 记录被压缩或排除的逐 trial 明细；`checksums.sha256` 可校验
文件完整性（在 `published/` 目录下 `sha256sum -c checksums.sha256`）。

## 8. 证据状态与使用边界（重要）

### 8.1 先读审计报告

在引用任何 TSIL/LUSI 结果之前，先读
[audit_20261107/REPORT.md](audit_20261107/REPORT.md) 与
[v1.1_remediation_report.md](v1.1_remediation_report.md)。审计的原始结论是：

- 工程机制（holdout 单文件语义、标签时间隔离、HAC/bootstrap/BH 实现、
  D-01…D-16 修复）在独立复算中成立；
- 但存在未修复风险：跨轮次验证 fold 复用构成未被 BH 覆盖的自适应多重比较；
  确认 fold 与 holdout 账本没有跨数据库文件的暴露登记；部分"跨股票/市场"
  谓词在 `shuffle=true` 真实训练配置下的机制解释不成立；无 Git 环境下代码
  指纹会退化为常量；上游切片存在跨标的窗口污染与 `drop_last` 丢样。

各条目的修复状态以 `v1.1_remediation_report.md` 为准，不要把"已修复"
推广到报告未覆盖的条目。

### 8.2 历史 TSIL 运行不是 V1.1 阳性证据

V1.0 的 `run_20261031_153702` 在 2024–2025 holdout 上**未确认**
（paired IC −0.00103，p = 0.386）。该 holdout 在 V1.1 下已标记为 consumed，
不能再次读取。Discovery 阶段的 `exploratory_positive` 不等于独立确认。

### 8.3 小样本研究是合成证据

`docs/results_small_sample_20261108.md` 是 1,200 次重复的**合成**正式研究
（预注册：`docs/preregistration_small_sample_20261108.md`）。它验证的是平台
门控行为，不是真实世界结论。关键数字：null 确认假阳性 3/298 = 1.0%
（95% CI 0.3%–2.9%）；效应量 s=3.0 时 n=256 的确认恢复率 0.700，n=24 为
0.103；执行完整性 1,192 ok / 8 blocked / 0 error。小 n（≤64）下的确认只能读作
"通过了独立门控"，不能读作"效应已确立"。

### 8.4 这个包能支持什么

- **可以**：复现平台机制、在自有数据上跑完整生命周期、检验门控的
  I 类错误/功效、扩展新领域或新谓词。
- **不可以**：把 `published/` 中的历史 exploratory 结果当作已确认发现；
  在未核对审计与修复报告的情况下声称某条机制解释成立；把合成小样本结果
  外推到真实市场。

## 9. 测试与校验

```bash
export PYTHONPATH=.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
# 本包基线：190 passed（无 qlib；TSIL 相关测试会按设计跳过）
```

`published/checksums.sha256` 校验已发布快照；交付 zip 的 SHA-256 见随包的
`VERSION.txt`（若有）。

## 10. 许可与引用

- `TSIL-main/` 为 **GPL-3.0** 上游代码副本，分发时保留其 `LICENSE` 与版权声明。
- 仓库根目录**没有** LICENSE 文件。若要在论文、产品或公开仓库中再分发，
  请先与作者确认许可与引用方式。
- 引用本平台时，请同时引用其依赖的上游 TSIL 工作（见 `docs/tsil_experiment.md`）。

## 11. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `FileNotFoundError` 读日历文件 | 配置里的 `provider_uri` / `tsil_root` 仍指向原作者机器，改成你的路径 |
| `holdout already consumed` | 该时间窗口已被消费；V1.1 不允许二次读取，换数据或新窗口 |
| `code_fingerprint changed since run creation` | 运行期间源码变动；恢复干净工作树后重跑（这是设计行为） |
| LLM 相关配置报 "no API key" | 设置 `DEEPSEEK_API_KEY`（或 `DEEPSEEK_API_KEY_FILE`），不要把 key 写进配置 |
| pytest 因 `fugue_test`/`fsspec` 报错 | 使用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`（`pytest.ini` 已设置） |

---

有疑问时，优先读 `docs/code_guide.md`（中文端到端走读）和
`docs/v1.1_evaluation_protocol.md`（规范性规则），再读具体结果文档。
