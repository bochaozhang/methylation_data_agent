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

## GEO Search 流程（v3）

GEO 检索采用 4 层漏斗式过滤，从宽到窄逐步筛选：

```
用户查询
  │
  ▼
parse_query → intent dict
  │
  ▼
build_geo_search_string(intent)  ← cancer_synonyms.yaml（16 癌种同义词 + 18 技术词 + 17 液体活检词）
  │
  ▼
NCBI esearch → GSE UID list
  │
  ▼
batch esummary → 元数据 + 平台/年份过滤
  │
  ▼
get_gsm_details() → GSM Characteristics  ← efetch MiniML XML（Source-Name / Extracted-Molecule / Characteristics）
  │
  ▼
_llm_judge_datasets_concurrent(5 并发)  ← GLM system cache（Z.AI 隐式缓存，cached_tokens 降费加速）
  │
  ▼
Registry 去重 → 注册 → 下载
  │
  ▼
geo_candidates_<ts>.json + report_<ts>.json
```

### 第 1 层：检索式构建

`build_geo_search_string(intent)` 将用户意图拼成 NCBI E-utilities 查询字符串，各子句用 `AND` 连接：

| 子句 | 逻辑 | 示例 |
|------|------|------|
| **癌种** | 从 `cancer_synonyms.yaml` 查 TCGA code 对应同义词，拼成 OR 组 | COAD → `(CRC OR "colorectal cancer" OR "colon cancer" OR adenoma OR ...)` |
| **样本类型** | 液体活检（cfdna/plasma/serum）用 YAML 中 17 个词做 OR；否则走 `SAMPLE_TYPE_GEO_TERMS` 映射 | cfdna → `(cfDNA OR "cell-free DNA" OR ctDNA OR plasma OR "liquid biopsy" OR ...)` |
| **甲基化技术** | 指定平台时用 GPL 号 + 俗名；未指定时用 YAML 中 18 个技术词做 OR | 无平台 → `("DNA methylation" OR methylome OR 450K OR EPIC OR MCTA OR WGBS OR ...)` |
| **物种** | 固定加 `(human OR "Homo sapiens")` | — |
| **年份** | 可选的 PDAT 范围过滤 | `("2020/01/01"[PDAT] : "2024/12/31"[PDAT])` |
| **Entry Type** | 固定 `GSE[Entry Type]` | — |

最终查询形如：
```
(CRC OR "colorectal cancer" OR ...) AND (cfDNA OR "cell-free DNA" OR ...) AND ("DNA methylation" OR 450K OR EPIC OR ...) AND (human OR "Homo sapiens") AND GSE[Entry Type]
```

### 第 2 层：元数据获取 + 平台/年份过滤

`GEOClient.search_gse()` → `filter_methylation_datasets()`：

1. **esearch** 拿到 GSE UID 列表（max_results=100）
2. **batch esummary** 批量获取元数据（title, summary, platforms, sample_count, year, sample_titles）
3. **过滤**：data_type 检测（array vs sequencing）、platform_canonical 匹配、year 范围；platform_unknown 的保留（不过度过滤）

### 第 3 层：LLM 判断（替代关键词过滤器）

`_llm_judge_datasets_concurrent(datasets, wanted_sample_type, max_concurrent=5)`：

1. **GSM 详情获取**（`_enrich_dataset_for_llm`）：对每个 GSE，调用 `get_gsm_details()` efetch MiniML XML，提取前 5 个 GSM 的 Source-Name、Extracted-Molecule、Characteristics
2. **LLM 判断**（`_llm_judge_dataset`）：把 dataset 元数据 + GSM 详情发给 GLM，system prompt 固定（触发 Z.AI system cache），user message 包含动态内容。LLM 返回 JSON：
   ```json
   {"keep": true/false, "confidence": "high|medium|low", "reason": "...", "detected_sample_type": "plasma|tumor|..."}
   ```
3. **5 并发**：asyncio + Semaphore(5)，最多 5 个 LLM 调用并行
4. **保守策略**：LLM 出错或证据不足时 keep=True（宁可多留，不误杀）
5. **硬规则**：cell line / organoid / in-vitro → keep=false

### 第 4 层：去重 + 注册 + 下载

1. **Registry 去重**：已存在的 accession 跳过
2. **注册**：`upsert_dataset()` 写入 SQLite（含 8 个新列：disease_groups, stage_treatment, available_file_type, sample_level_annotation, usable, recommended_action, reason, notes）
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
│   └── cancer_synonyms.yaml   # 癌种同义词 + 技术词 + 液体活检词（v3 新增）
├── agents/
│   ├── database_agent.py      # Agent 1：GEO + TCGA 搜索下载（含 LLM judge + 5 并发）
│   ├── literature_agent.py    # Agent 2：文献挖掘 + 补充下载
│   └── orchestrator.py        # LangGraph 图定义与编排
├── tools/
│   ├── geo_tools.py           # GEO NCBI E-utilities API（含 get_gsm_details）
│   ├── tcga_tools.py          # GDC REST API
│   ├── pubmed_tools.py        # PubMed / PMC / bioRxiv
│   ├── download_tools.py      # 异步断点续传下载器
│   └── parser_tools.py        # 关键词解析 + accession 提取 + 同义词扩展
├── registry/
│   └── registry.py            # SQLite 注册表（去重核心，含 8 个新列）
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
└── geo_candidates_20240522_143021.json  # GEO 候选列表（含 LLM judge 字段）
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
      "sample_count": 120,
      "year": 2022,
      "data_type": "array",
      "sample_type": "cfdna",
      "detected_sample_types": ["cfdna", "plasma"],
      "pubmed_ids": ["35123456"],
      "llm_keep": true,
      "llm_confidence": "high",
      "llm_reason": "plasma cfDNA confirmed by GSM source_name",
      "usable": 1,
      "recommended_action": "download",
      "reason": null,
      "notes": null
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
| cancer_type | TEXT | 癌种 |
| platform | TEXT | 450K / EPIC / WGBS / RRBS |
| sample_type | TEXT | tumor / cfdna / plasma / wbc ... |
| download_status | TEXT | pending / downloading / done / failed / skipped |
| disease_groups | TEXT | 癌种分组（v2 新增） |
| stage_treatment | TEXT | 分期/治疗信息（v2 新增） |
| available_file_type | TEXT | 检测到的文件类型（v2 新增） |
| sample_level_annotation | TEXT | GSM 级注释 JSON（v2 新增） |
| usable | INTEGER | 0=LLM 排除, 1=可用（v2 新增） |
| recommended_action | TEXT | download / review / skip（v2 新增） |
| reason | TEXT | LLM 判断理由（v2 新增） |
| notes | TEXT | 自由备注（v2 新增） |

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
- 补充材料解析仅支持 PMC 开放获取文章
- LLM judge 使用 Z.AI GLM，system prompt 固定不变以触发隐式缓存（cached_tokens 降费加速）
- `cancer_synonyms.yaml` 可独立更新，无需改代码即可添加新癌种同义词
