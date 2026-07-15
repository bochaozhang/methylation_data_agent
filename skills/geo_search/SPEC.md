# geo-search（GEO 候选召回）Skill

适用对象：癌症早筛 AI Scientist 中负责 **GEO 候选数据集召回**的 skill（`geo-search`）。
本 skill 只做一件事：根据 parsed intent，构建扩展检索式，在 GEO 中宽泛召回候选 GSE 列表，供下游 `geo-filter` 判定。

> **本文件是文档，不是 LLM 指令。** 检索式由确定性代码构造（`build_geo_search_string` + 本 skill 自有的 `synonyms.yaml`），不靠 LLM 拼。LLM 的价值留给下游 filter 的语义判定。GEO-only；TCGA 是另一个独立模块，不在本 skill。

## Scope（做什么 / 不做什么）

- Does: 根据意图构建 GEO 检索式 → esearch + esummary → 输出候选 GSE 完整元数据列表。
- Does NOT:
  - 判断数据是否可用 → `geo-filter` 负责。
  - 搜索 TCGA / 其他数据库 → 各自独立模块。
  - 下载 → `geo-download` 负责。

## 输入 State 字段

从 `parsed_intent`（canonical `SearchIntent`）读取：

- `mode`（accession / semantic）、`accessions.geo`（accession 模式直取）
- `cancer_type.{display, tcga_code}`（驱动同义词扩展）
- `sample_type` / `sample_types`（驱动样本类型子句；液体活检词集合）
- `platform`、`data_type`、`year_start` / `year_end`
- `extra_keywords`（补充词）

## 输出 State 字段

- `candidate_gse_list`：每条 = `GEOClient.filter_methylation_datasets()` 返回的完整 dict（accession / title / summary / overall_design / platform_canonical / platforms / sample_count / year / pubmed_ids / data_type / sample_titles / 注入的 cancer_type）。**不要砍成薄结构**——下游 filter 要用这些字段。
- `search_queries`：实际用到的 NCBI 检索式（写进 per-query CSV 日志）。
- `search_log`：简述（模式、检索式数、召回数）。

## 检索式构造（确定性，代码实现）

1. **癌种同义词**：按 `tcga_code` 查 `synonyms.yaml` 的 `cancer_synonyms`（全称/缩写/解剖拆分/TCGA 名/癌前病变），OR 连接。
2. **甲基化技术词**：`synonyms.yaml` 的 `methylation_tech_terms`（DNA methylation / 450K / EPIC / RRBS / WGBS / bisulfite sequencing / MCTA / MeDIP / methylome …）。指定 platform 时用平台专有词（如 450K→GPL13534 等）。
3. **液体活检词**：当 `sample_type ∈ {cfdna, plasma, serum}` 或 sample_types 含这些时，加 `synonyms.yaml` 的 `liquid_biopsy_terms`（cfDNA / cell-free DNA / ctDNA / plasma / serum / liquid biopsy / non-invasive …）。
4. **物种**：固定 `(human OR "Homo sapiens")`。
5. **年份**：可选 PDAT 范围。
6. **Entry Type**：固定 `GSE[Entry Type]`。
7. **长度安全**：检索式上限 400 字符，超长自动裁剪癌种同义词，避免触发 NCBI abuse 检测。

## 执行范围（现实边界）

- 只搜 **GEO Series（GSE）**（`db=gds` + `GSE[Entry Type]`）。GSE 是数据所在；GSM/GPL 通过 series 触达。
- **不分别搜 GDS / GSM / GPL**（代码未实现多层级并行检索；如需，未来扩展）。

## 核心原则

1. **宽搜**：第一步检索可以宽，不要一开始加 NOT 条件，避免漏掉有用数据。
2. **关键词只用于召回**：关键词命中 ≠ 数据可用；纳入/排除由 `geo-filter` 判定。
3. **必须扩展同义词**：不能只搜癌种全称 / 只搜 `DNA methylation` / 只搜 `cfDNA`。
4. **同义词表是数据**：更新检索词 = 编辑 `skills/geo_search/synonyms.yaml`，不改代码。
