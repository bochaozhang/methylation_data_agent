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

## 安装

```bash
cd methyagent
pip install -r requirements.txt
```

## 配置

编辑 `config/settings.yaml`：

```yaml
llm:
  backend: openai          # openai | anthropic | ollama
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
│   └── settings.yaml          # 配置文件
├── agents/
│   ├── database_agent.py      # Agent 1：GEO + TCGA 搜索下载
│   ├── literature_agent.py    # Agent 2：文献挖掘 + 补充下载
│   └── orchestrator.py        # LangGraph 图定义与编排
├── tools/
│   ├── geo_tools.py           # GEO NCBI E-utilities API
│   ├── tcga_tools.py          # GDC REST API
│   ├── pubmed_tools.py        # PubMed / PMC / bioRxiv
│   ├── download_tools.py      # 异步断点续传下载器
│   └── parser_tools.py        # 关键词解析 + accession 提取
├── registry/
│   └── registry.py            # SQLite 注册表（去重核心）
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
├── report_20240522_143021.json   # 完整报告（JSON）
└── report_20240522_143021.md     # 可读报告（Markdown）
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
