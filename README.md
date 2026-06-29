# MethyAgent

基于 LangGraph 的双 Agent 甲基化数据自动采集系统。

## 系统架构

```
用户输入关键词
      │
      ▼
┌─────────────────────────────────────────────────────┐
│                  Orchestrator (LangGraph)             │
│  parse_query → DatabaseAgent → LiteratureAgent → Report │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │  共享注册表       │
              │  (SQLite)        │
              │  去重 + 状态追踪  │
              └─────────────────┘
```

**Agent 1 (DatabaseAgent)**：直接从 TCGA GDC 和 GEO 数据库检索并下载甲基化数据。

**Agent 2 (LiteratureAgent)**：从 PubMed / PMC / bioRxiv 文献中挖掘数据集引用，补充下载 Agent 1 未覆盖的数据。

两个 Agent 通过共享 SQLite 注册表协调，Agent 2 下载前自动检查注册表，跳过已由 Agent 1 下载的数据集。

## GEO Search 流程（v5）

GEO 检索采用 **三步漏斗式过滤**，从粗到细逐步筛选：

```
用户查询
  │
  ▼
parse_query → intent dict
  │
  ▼
build_geo_search_string(intent)  ← cancer_synonyms.yaml（16 癌种同义词 + 8 技术词 + 8 液体活检词）
  │
  ▼
NCBI esearch → GSE UID list
  │
  ▼
batch esummary → 元数据 + 平台/年份过滤
  │
  ▼
Step 1: GEO metadata screen（GSE 级，LLM 粗筛，无 NCBI 调用）
  LLM 读 title/summary/overall_design vs intent
  明显不符（cell line / 错误癌种 / 错误样本类型）→ 丢弃
  │
  ▼ screen_keep=True
Step 2: GSM sample metadata judge（GSM 级，ground truth，两次 LLM 调用）
  ├── 全量 efetch 所有 GSM MINiML XML（无上限，结果缓存到 CSV）
  ├── LLM Call 1（per-GSM，并发）
  │     输入：intent + 单个 GSM characteristics（~100-300 tokens/call）
  │     输出：{"include": bool, "reason": str|null}
  │     → 写入 sample_metadata.csv（全量，每行一个 GSM）
  ├── 计算 include_fraction = include数 / 总GSM数
  └── LLM Call 2（dataset 级，单次）
        输入：include_fraction + exclude_fraction + GEO summary（~200-400 tokens）
        输出：{"dataset_keep": "true"|"false"|"unsure", "reason": str}
  │
  ├── "true"  → 直接进 awaiting_approval（跳过 Step 3）
  ├── "false" → 丢弃
  └── "unsure" → Step 3
  │
  ▼ dataset_keep="unsure"
Step 3: PubMed verify（仅对 unsure 数据集）
  有 PMID + 有 abstract → LLM 对比 GEO summary vs 摘要 → keep=True/False
  无 PMID 或无 abstract → pubmed_keep=False（丢弃）
  │
  ▼
awaiting_approval → 人工审批 → pending → daemon 下载
  │
  ▼
Registry 去重 → 注册 → 下载
  │
  ▼
sample_metadata.csv + report_<ts>.json
```

### 第 1 层：检索式构建

`build_geo_search_string(intent)` 将用户意图拼成 NCBI E-utilities 查询字符串，各子句用 `AND` 连接。查询字符串上限 400 字符，超长时自动裁剪癌种同义词（`MAX_QUERY_LENGTH` 机制）。

| 子句 | 逻辑 | 示例 |
|------|------|------|
| **癌种** | 从 `cancer_synonyms.yaml` 查 TCGA code 对应同义词，拼成 OR 组 | COAD → `(CRC OR "colorectal cancer" OR "colon cancer" OR ...)` |
| **样本类型** | 液体活检（cfdna/plasma/serum）用 YAML 中 8 个高区分度词做 OR；否则走 `SAMPLE_TYPE_GEO_TERMS` 映射 | cfdna → `(cfDNA OR "cell-free DNA" OR ctDNA OR plasma OR serum OR "liquid biopsy" OR "circulating DNA")` |
| **甲基化技术** | 指定平台时用 GPL 号 + 俗名；未指定时用 YAML 中 8 个技术词做 OR | 无平台 → `("DNA methylation" OR 450K OR EPIC OR RRBS OR WGBS OR "bisulfite sequencing" OR "methylation array" OR methylome)` |
| **物种** | 固定加 `(human OR "Homo sapiens")` | — |
| **年份** | 可选的 PDAT 范围过滤 | `("2020/01/01"[PDAT] : "2024/12/31"[PDAT])` |
| **Entry Type** | 固定 `GSE[Entry Type]` | — |

### 第 2 层：元数据获取 + 平台/年份过滤

`GEOClient.search_gse()` → `filter_methylation_datasets()`：

1. **esearch** 拿到 GSE UID 列表（max_results=2000）
2. **batch esummary** 批量获取元数据（title, summary, overall_design, platforms, sample_count, year, pubmed_ids）
3. **过滤**：data_type 检测（array vs sequencing）、platform_canonical 匹配、year 范围；platform_unknown 的保留（不过度过滤）

### Step 1：GEO metadata screen（GSE 级粗筛）

`_geo_screen_datasets_concurrent(datasets, max_concurrent=5)`

LLM 仅读取 GEO 元数据（title、summary、overall_design），无需 NCBI API 调用，快速丢弃明显不符的数据集：

- **RULE 1**：cell line / organoid / in-vitro → 直接拒绝
- **RULE 2**：数据类型明显不是 DNA 甲基化 → 拒绝
- **RULE 3**：癌种明显不符 → 拒绝
- **RULE 4**：样本类型明显不符 → 拒绝
- **RULE 5（默认）**：保留（宽松，允许假阳性，Step 2 做精细判断）

### Step 2：GSM sample metadata judge（GSM 级，ground truth）

`_sample_metadata_judge_concurrent(datasets, max_concurrent=3)`

这是主过滤器，基于 GSM 级别的真实样本信息做判断，分两次 LLM 调用：

#### LLM Call 1：per-GSM include/exclude（并发，Semaphore=5）

对每个 GSM 独立判断，输入极短（单个样本的 characteristics）：

```
输入：
  Intent: {cancer_type}, {sample_type}, {query_detail}
  GSM: {gsm_id}
    source_name: {source_name}
    molecule: {molecule}
    characteristics: {key: value, ...}

输出：
  {"include": true, "reason": null}          ← include 时 reason 为 null
  {"include": false, "reason": "tumor tissue, not cfDNA"}
```

结果写入 `sample_metadata.csv`（全量，不截断）：
- include=true → 该 query 列写 `"include"`
- include=false → 该 query 列写 `"exclude: {reason}"`

#### LLM Call 2：dataset_keep 三态判断（单次）

基于汇总统计 + GEO summary，输入极短（~200-400 tokens）：

```
输入：
  Intent: {cancer_type}, {sample_type}
  Dataset: {accession}
    title / summary / overall_design
    include_count: {k}  (include_fraction: {k/n:.1%})
    exclude_count: {n-k}  (exclude_fraction: {(n-k)/n:.1%})

输出：
  {"dataset_keep": "true" | "false" | "unsure", "reason": "..."}
```

**三态路由规则**：

| dataset_keep | 条件 | 行为 |
|-------------|------|------|
| `"true"` | include_fraction ≥ 20% 且 GEO summary 一致 | → awaiting_approval（跳过 Step 3） |
| `"false"` | include_fraction < 5% 且 GEO summary 明确不符 | → 丢弃 |
| `"unsure"` | 边界情况，或 characteristics 字段缺失/模糊 | → Step 3 PubMed 核验 |

**CSV 缓存**：`sample_metadata.csv` 写入 `{output_dir}/{accession}/sample_metadata.csv`，同一数据集再次查询时复用 efetch 结果，仅重跑 LLM 判断，追加新 query 列。

#### GSM 分组策略（用于 CSV group 列）

根据 GSM title 关键词分组（仅用于 CSV 标注，不影响 LLM 判断）：

| 分组 | 匹配关键词 |
|------|-----------|
| `plasma_cfdna` | plasma, cfdna, cell-free, cell free, serum, liquid biopsy, circulating, ctdna |
| `tissue` | tumor, tumour, tissue, biopsy, ffpe, frozen, gdna, genomic dna, primary, cancer tissue, solid tumor |
| `wbc_blood` | wbc, pbmc, buffy coat, leukocyte, whole blood, peripheral blood, mononuclear |
| `normal` | normal, healthy, adjacent, control, benign |
| `cell_line` | cell line, organoid, in vitro, culture |
| `other` | 不匹配以上任何关键词的样本 |

### Step 3：PubMed verify（仅对 unsure 数据集）

`_pubmed_verify_datasets_concurrent(unsure_list, max_concurrent=5)`

**仅当 Step 2 返回 `dataset_keep="unsure"` 时触发**，不再对所有数据集执行。

**核验流程（每个 unsure GSE）**：

1. 取 `pubmed_ids[0]`（esummary 已返回）
2. `GEOClient.fetch_pubmed_abstract(pmid)` → NCBI efetch 获取摘要
3. 将 GEO 元数据 + 摘要发给 LLM（`LLM_VERIFY_SYSTEM_PROMPT`，固定不变触发 Z.AI system cache）
4. LLM 返回 `keep=True/False` + 更新字段

**严格策略（unsure 状态下不保守保留）**：

| 情况 | 行为 |
|------|------|
| 无 PMID | `pubmed_keep=False`，丢弃（unsure + 无文章 = 无法确认 = 不下载） |
| 摘要获取失败 | `pubmed_keep=False`，丢弃 |
| LLM / 解析出错 | `pubmed_keep=False`，丢弃（保守拒绝） |

> **设计原则**：Step 2 已经是 ground truth 判断，进入 Step 3 的数据集本身就是"存疑"的。无法通过文献确认的存疑数据集，宁可漏掉也不要噪音。

**5 并发**：asyncio + Semaphore(5)。

### Token 控制策略

| LLM 调用 | 输入内容 | Token 量 |
|---------|---------|---------|
| Step 1 screen（per-GSE） | title + summary + overall_design | ~300-600 tokens |
| Step 2 Call 1（per-GSM） | intent + 单个 GSM characteristics | ~100-300 tokens/call |
| Step 2 Call 2（per-dataset） | include/exclude 统计 + GEO summary | ~200-400 tokens |
| Step 3 PubMed verify（per-unsure） | GEO metadata + PubMed abstract | ~800-1500 tokens |

全量 GSM 数据只写 CSV，**不发给任何 LLM**。

### 第 4 层：去重 + 注册 + 人工审批 + 下载

1. **Registry 去重**：已存在的 accession 跳过
2. **注册**：`upsert_dataset()` 写入 SQLite（含 `sample_metadata_path` 字段）
3. **人工审批**：Web UI "审批下载" Tab → 勾选确认 → `POST /datasets/approve`
4. **daemon 下载**：后台轮询 `pending` 状态数据集，执行下载
5. **报告**：`report_<ts>.json` + `sample_metadata.csv`（每个数据集一份）


## 安装

```bash
cd methyagent
pip install -r requirements.txt
```

## 配置

编辑 `config/settings.yaml`：

```yaml
llm:
  backend: openai          # openai | anthropic | ollama | zhipu
  model: gpt-4o
  api_key_env: OPENAI_API_KEY

download:
  output_dir: ./data/methylation
  max_concurrent: 5
```

设置 API Key 环境变量：

```bash
export OPENAI_API_KEY=sk-...
export NCBI_API_KEY=...   # 可选，提高 NCBI 速率限制
```

## 使用方法

### 语义搜索（自然语言）

```bash
python main.py --query "EPIC平台在2024年的乳腺癌相关数据"
python main.py --query "breast cancer WGBS methylation 2022-2023"
python main.py --query "2020-2023年肺癌450K甲基化数据"
python main.py --query "结直肠癌cfDNA甲基化血浆数据"
```

### 精确 Accession 下载

```bash
python main.py --query "下载GEO编号GSE124600的所有数据"
python main.py --query "GSE124600 GSE200234"
python main.py --query "TCGA-BRCA methylation data"
```

### 运行模式

```bash
# 仅运行数据库搜索（跳过文献挖掘）
python main.py --query "..." --agent db-only

# 仅运行文献挖掘（跳过数据库搜索）
python main.py --query "..." --agent lit-only

# 两个 Agent 都运行（默认）
python main.py --query "..." --agent both
```

### 其他命令

```bash
# 查看注册表状态
python main.py --status

# 解析查询但不下载（调试用）
python main.py --query "..." --dry-run

# 详细日志
python main.py --query "..." --verbose

# 自定义输出目录
python main.py --query "..." --output-dir /data/my_methylation
```

## 项目结构

```
methyagent/
├── config/
│   ├── settings.yaml          # 配置文件
│   └── cancer_synonyms.yaml   # 癌种同义词 + 技术词 + 液体活检词
├── agents/
│   ├── database_agent.py      # Agent 1：GEO + TCGA 搜索下载（三步过滤：GEO screen → GSM judge → PubMed verify）
│   ├── literature_agent.py    # Agent 2：文献挖掘 + 补充下载
│   └── orchestrator.py        # LangGraph 图定义与编排
├── tools/
│   ├── geo_tools.py           # GEO NCBI E-utilities API（含 get_all_gsm_metadata + get_representative_gsm_details + fetch_pubmed_abstract）
│   ├── tcga_tools.py          # GDC REST API
│   ├── pubmed_tools.py        # PubMed / PMC / bioRxiv
│   ├── download_tools.py      # 异步断点续传下载器
│   └── parser_tools.py        # 关键词解析 + accession 提取 + 同义词扩展
├── registry/
│   └── registry.py            # SQLite 注册表（去重核心，含核心列 + PubMed 核验列）
├── state/
│   └── graph_state.py         # LangGraph TypedDict 状态
├── utils/
│   ├── logger.py              # 日志配置
│   └── llm_factory.py         # LLM 后端工厂
├── main.py                    # CLI 入口
└── requirements.txt
```

## 输出

运行完成后在 `output_dir` 生成：

```
data/methylation/
├── GSE124600/
│   └── GSE124600_series_matrix.txt.gz
├── TCGA-BRCA/
│   └── *.methylation_array.sesame.level3betas.txt
├── GSE220160/
│   └── sample_metadata.csv             # GSM 级样本元数据（v5 新增）
├── report_20240522_143021.json         # 完整报告（JSON）
├── report_20240522_143021.md           # 可读报告（Markdown）
└── geo_candidates_20240522_143021.json  # GEO 候选列表（含 PubMed 核验字段）
```

`geo_candidates_<ts>.json` 结构示例：

```json
{
  "query": "结直肠癌cfDNA甲基化血浆数据",
  "timestamp": "2024-05-22T14:30:21+00:00",
  "total": 12,
  "candidates": [
    {
      "accession": "GSE220160",
      "title": "Plasma cfDNA methylation in CRC patients",
      "cancer_type": "colorectal cancer",
      "platform": "450K",
      "sample_count": 130,
      "year": 2022,
      "data_type": "array",
      "sample_type": "plasma",
      "pubmed_ids": ["35123456"],
      "pubmed_verified": true,
      "pubmed_keep": true,
      "paper_pmid": "35123456",
      "consistency": "consistent",
      "stage_treatment": "stage II-III, treatment-naive",
      "usable": 1,
      "recommended_action": "download",
      "reason": "Abstract confirms plasma cfDNA from CRC patients, n=130",
      "notes": "sample_count GEO=120 paper=130"
    }
  ]
}
```

注册表保存在 `registry/methyagent.db`（SQLite）。

## 去重机制

```
Agent 2 下载前检查流程：

提取到 accession X
        │
        ▼
查询 registry.db WHERE accession = X
        │
   ┌────┴────┐
   │ 存在    │ 不存在
   ▼         ▼
跳过，记录  写入注册表 → 下载
"已由Agent1  status=pending
 覆盖"
```

## Registry 数据库 Schema

`datasets` 表核心列：

| 列名 | 类型 | 说明 |
|------|------|------|
| accession | TEXT PK | GSE / TCGA 编号 |
| source | TEXT | GEO / TCGA |
| cancer_type | TEXT | 癌种（PubMed 核验后更新） |
| platform | TEXT | 450K / EPIC / WGBS / RRBS |
| sample_type | TEXT | tumor / cfdna / plasma / wbc ...（PubMed 核验后更新） |
| sample_count | INTEGER | 样本数（PubMed 核验后如差异 >20% 则修正） |
| download_status | TEXT | pending / downloading / done / failed / skipped |
| disease_groups | TEXT | 癌种分组（v2 新增） |
| stage_treatment | TEXT | 分期/治疗信息（PubMed 核验后更新） |
| available_file_type | TEXT | 检测到的文件类型（v2 新增） |
| sample_level_annotation | TEXT | GSM 级注释 JSON（v2 新增） |
| usable | INTEGER | 0=排除, 1=可用（与 pubmed_keep 同步） |
| recommended_action | TEXT | download / review / skip（PubMed 核验输出） |
| reason | TEXT | 核验结论（PubMed 核验输出） |
| consistency | TEXT | consistent / minor_discrepancy / major_discrepancy（v4 新增） |
| notes | TEXT | 自由备注，追加不覆盖 |
| pubmed_verified | INTEGER | 1=已完成 PubMed 核验，0=跳过（v4 新增） |
| pubmed_keep | INTEGER | 1=核验通过，0=核验拒绝（v4 新增） |
| sample_metadata_path | TEXT | GSM 级样本元数据 CSV 路径（v5 新增） |
| paper_pmid | TEXT | 核验所用 PMID（v4 新增） |

旧数据库自动通过 `_migrate_schema()` 迁移，无需手动操作。

## 支持的数据类型

| 类型 | 平台 | 文件格式 |
|------|------|---------|
| Illumina 450K | HumanMethylation450 | beta 值矩阵 .txt.gz |
| Illumina EPIC | HumanMethylationEPIC | beta 值矩阵 .txt.gz |
| WGBS | 全基因组亚硫酸盐测序 | .bismark.cov.gz, .bed.gz |
| RRBS | 简化亚硫酸盐测序 | .cov.gz, .bed.gz |

## 注意事项

- TCGA 公开数据（Level 3 beta 值）无需 token
- TCGA 受控数据（Level 1/2 原始数据）需要 dbGaP 授权，在 `settings.yaml` 中配置 `GDC_TOKEN`
- NCBI API Key 可选，但建议设置（提高速率限制从 3 req/s 到 10 req/s）
- 云服务器 IP 可能被 NCBI 标记为 abuse，可通过 `settings.yaml` 的 `geo.proxy` 或环境变量 `NCBI_PROXY` 配置 SOCKS5/HTTP 代理
- 补充材料解析仅支持 PMC 开放获取文章
- Step 2 使用 `LLM_GSM_JUDGE_SYSTEM_PROMPT`（Call 1）和 `LLM_DATASET_KEEP_SYSTEM_PROMPT`（Call 2），Step 3 使用 `LLM_VERIFY_SYSTEM_PROMPT`；三个 prompt 均固定不变以触发 Z.AI 隐式缓存（cached_tokens 降费加速）
- PubMed 核验（Step 3）**仅对 Step 2 返回 `dataset_keep="unsure"` 的数据集执行**，大幅减少 NCBI efetch 和 LLM 调用次数
- Step 3 中无 PMID / 摘要获取失败 / LLM 出错时，数据集**默认丢弃**（unsure + 无法确认 = 不下载）；Step 1 screen 出错时仍保守保留
- `keep=False` 的数据集直接丢弃，不写入注册表；`recommended_action=review` 的数据集写入注册表供人工复查
- `cancer_synonyms.yaml` 可独立更新，无需改代码即可添加新癌种同义词
- 查询字符串上限 400 字符（`MAX_QUERY_LENGTH`），超长时自动裁剪癌种同义词，避免触发 NCBI abuse 检测
- v5 新增 GSM 级两次 LLM 判断（`_sample_metadata_judge_concurrent`），是主过滤器；旧的 `_llm_judge_datasets_concurrent` 保留在代码中但不在主流程中调用
