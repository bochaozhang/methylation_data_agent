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

## GEO Search 流程（v4）

GEO 检索采用 4 层漏斗式过滤，从宽到窄逐步筛选：

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
_pubmed_verify_datasets_concurrent(5 并发)  ← 主过滤器
  每个 GSE：efetch PubMed abstract → LLM 对比 GEO summary/design vs 摘要
  keep=True → 保留并更新字段；keep=False → 丢弃并记录原因
  │
  ▼
Registry 去重 → 注册 → 下载
  │
  ▼
geo_candidates_<ts>.json + report_<ts>.json
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

最终查询形如：
```
(CRC OR "colorectal cancer" OR ...) AND (cfDNA OR "cell-free DNA" OR ...) AND ("DNA methylation" OR 450K OR EPIC OR ...) AND (human OR "Homo sapiens") AND GSE[Entry Type]
```

### 第 2 层：元数据获取 + 平台/年份过滤

`GEOClient.search_gse()` → `filter_methylation_datasets()`：

1. **esearch** 拿到 GSE UID 列表（max_results=2000）
2. **batch esummary** 批量获取元数据（title, summary, overall_design, platforms, sample_count, year, pubmed_ids）
3. **过滤**：data_type 检测（array vs sequencing）、platform_canonical 匹配、year 范围；platform_unknown 的保留（不过度过滤）

### GSM 分组选取策略

`get_representative_gsm_details(accession, wanted_sample_type)` 用于从 GSE 的全部 GSM 中选取代表性样本做 efetch MiniML，避免只取前 N 个导致混合设计数据集的偏差。

**分组规则**：根据 GSM title 中的关键词，将每个样本归入以下 5 组 + 1 个兜底组（按优先级从上到下匹配，首次命中即归入）：

| 分组 | 匹配关键词 |
|------|-----------|
| `plasma_cfdna` | plasma, cfdna, cell-free, cell free, serum, liquid biopsy, circulating, ctdna |
| `tissue` | tumor, tumour, tissue, biopsy, ffpe, frozen, gdna, genomic dna, primary, cancer tissue, solid tumor |
| `wbc_blood` | wbc, pbmc, buffy coat, leukocyte, whole blood, peripheral blood, mononuclear |
| `normal` | normal, healthy, adjacent, control, benign |
| `cell_line` | cell line, organoid, in vitro, culture |
| `other` | 不匹配以上任何关键词的样本 |

**选取规则**：

| 数据集样本数 | 选取策略 |
|-------------|---------|
| n ≤ 30 | 全取（小数据集，完整覆盖） |
| n > 30 | 每组最多 6 个代表，总上限 30 个 |

大样本数据集的选取细节：
- 用户查询的 `wanted_sample_type` 对应的分组优先选满（如查 cfDNA → `plasma_cfdna` 组先选）
- 其余分组按顺序填充，直到总上限 30
- 只对选中的 GSM 做 efetch MiniML（节省 API 调用）

### 第 3 层：PubMed 文献核验（主过滤器）

`_pubmed_verify_datasets_concurrent(datasets, max_concurrent=5)`

**每一个** esummary 返回的 GSE 都经过此步骤，无论样本类型或其他元数据如何。LLM 直接对比 GEO 的 summary/overall_design 与文章摘要，判断两者是否一致，并输出 `keep` 决策。

**核验流程（每个 GSE）**：

1. 取 `pubmed_ids[0]`（esummary 已返回，零额外 API 调用）
2. `GEOClient.fetch_pubmed_abstract(pmid)` → NCBI efetch 获取摘要全文
3. 将 GEO 元数据（title、summary、**overall_design**、platform、sample_count、sample_type、cancer_type）+ 摘要发给 LLM
4. LLM 使用固定 `LLM_VERIFY_SYSTEM_PROMPT`（触发 Z.AI system cache）返回 JSON：

```json
{
  "keep": true,
  "confirmed_sample_type": "plasma",
  "confirmed_cancer_type": "colorectal cancer",
  "sample_count_in_paper": 130,
  "stage_treatment": "stage II-III, treatment-naive",
  "accession_mentioned": true,
  "consistency": "consistent",
  "recommended_action": "download",
  "reason": "Abstract confirms plasma cfDNA from CRC patients, n=130",
  "notes": ""
}
```

5. 根据 `keep` 字段决定去留：

| `keep` | 行为 |
|--------|------|
| `true` | 保留数据集，更新字段（sample_type、cancer_type、sample_count、stage_treatment 等） |
| `false` | 丢弃数据集，记录 reason 到日志 |

**更新字段规则**：

| 字段 | 更新条件 |
|------|---------|
| `sample_type` | 摘要中明确描述生物材料类型（非 unknown） |
| `cancer_type` | 摘要中明确癌种名称 |
| `sample_count` | 摘要中 n= 与 GEO 差异 >20% 时修正 |
| `stage_treatment` | 摘要中有分期/治疗信息 |
| `consistency` | consistent / minor_discrepancy / major_discrepancy |
| `usable` | 与 keep 同步（keep=false → usable=0） |
| `recommended_action` | download / review / skip |
| `reason` | 核验结论一句话摘要 |
| `notes` | 追加差异说明（不覆盖已有内容） |

**保守策略（宁可多留，不误杀）**：

| 情况 | 行为 |
|------|------|
| 无 PMID | `keep=True`，notes 追加 `no_pubmed_link` |
| 摘要获取失败 | `keep=True`，notes 追加 `abstract_unavailable` |
| LLM / 解析出错 | `keep=True`，notes 追加 `verify_error: ...` |

**5 并发**：asyncio + Semaphore(5)，efetch + LLM 调用并行执行。

日志格式示例：
```
pubmed_verify: 45 total → 38 verified, 5 no-PMID, 2 errors | 41 kept, 4 rejected
  Rejected GSE999001 (PMID=99999999): reason=Abstract describes tumor tissue, not plasma cfDNA
```

### 第 4 层：去重 + 注册 + 下载

1. **Registry 去重**：已存在的 accession 跳过
2. **注册**：`upsert_dataset()` 写入 SQLite（含核心列 + PubMed 核验列）
3. **下载**：构建下载任务，执行
4. **报告**：`generate_report` 节点输出 `report_<ts>.json` + `geo_candidates_<ts>.json`

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
│   ├── database_agent.py      # Agent 1：GEO + TCGA 搜索下载（PubMed 核验主过滤器 + 5 并发）
│   ├── literature_agent.py    # Agent 2：文献挖掘 + 补充下载
│   └── orchestrator.py        # LangGraph 图定义与编排
├── tools/
│   ├── geo_tools.py           # GEO NCBI E-utilities API（含 get_representative_gsm_details + fetch_pubmed_abstract）
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
- PubMed 核验使用 Z.AI GLM，`LLM_VERIFY_SYSTEM_PROMPT` 固定不变以触发隐式缓存（cached_tokens 降费加速）
- PubMed 核验对**每一个** esummary 返回的 GSE 执行，每个数据集消耗 1 次 NCBI efetch + 1 次 LLM 调用
- 无 PMID / 摘要获取失败 / LLM 出错时，数据集默认保留（保守策略）
- `keep=False` 的数据集直接丢弃，不写入注册表；`recommended_action=review` 的数据集写入注册表供人工复查
- `cancer_synonyms.yaml` 可独立更新，无需改代码即可添加新癌种同义词
- 查询字符串上限 400 字符（`MAX_QUERY_LENGTH`），超长时自动裁剪癌种同义词，避免触发 NCBI abuse 检测
- v4 移除了 GSM 级 LLM judge（`_llm_judge_datasets_concurrent`），该方法保留在代码中但不在主流程中调用
