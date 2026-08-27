# MethyAgent

双 Agent 甲基化数据自动采集系统：自然语言查询 → GEO/TCGA 检索 → 三态过滤（download / manual_review / exclude，GSM 样本级判定）→ 自动下载 → 下载后核验（列数 / GSM→列映射 / 疾病分组）→ SQLite 注册表全程追踪。

## 系统架构

```
用户 query（CLI 或 HTTP API）
      │
      ▼
┌──────────────────────────────────────────────────────┐
│  task_queue（SQLite）                                  │
│    agent1-query  : 搜索 → 过滤 → 注册（download→pending）│
│    agent1-download: 轮询 pending → 下载 → 核验 → done   │
│    agent2        : PubMed/PMC 文献挖掘补充              │
│    webui         : FastAPI（端口 80）提交/审批/看板      │
└──────────────────────────────────────────────────────┘
      │                                    │
      ▼                                    ▼
registry/methyagent.db              data/{GSE}/…
（datasets / samples /              （矩阵文件、per-GSM 文件、
 query_sample_map / …）              query_subset、quarantine/）
```

- **agent1（DatabaseAgent）**：GEO（E-utilities）+ TCGA（GDC）检索下载。skills 管线：`geo_search`（确定性召回）→ `geo_filter`（三态相关性判定，LLM + GSM 级证据）→ `geo_download`（三级 tier 下载 + LLM 按样本类型选文件 + Phase-2 核验）。
- **agent2（LiteratureAgent）**：从文献挖掘数据集引用，下载前查注册表去重。
- 两个 Agent 通过共享 SQLite 注册表协调（含 GSM 级多对多表，同一 GSE 被多个 query 命中互不覆盖）。

---

## Quick Start（Docker，推荐）

```bash
git clone <repo-url> methylation_data_agent
cd methylation_data_agent

# 1. 配置密钥
cp .env.example .env
# 编辑 .env，至少填：ZHIPU_API_KEY（LLM）、NCBI_API_KEY（GEO 检索）
#   可选：GEO_EMAIL、GDC_TOKEN（TCGA 受控数据）

# 2. 建目录（compose bind mount 需要）
mkdir -p registry data

# 3. 起服务（首次会自动 build 镜像，python:3.11-slim）
docker compose up -d
docker compose ps          # webui healthy 即就绪

# 4. 健康检查
curl -s http://localhost/health
# → {"status":"ok","agent1_alive":true,...}

# 5. 提交查询（生产路径：download 桶自动下载，无需人工审批）
curl -s -X POST http://localhost/query -H "Content-Type: application/json" \
  -d '{"query":"结直肠癌和非癌对照的血浆cfDNA甲基化数据","agent_type":"database"}'
# → {"task_id":"<uuid>","status":"pending",...}   记下 task_id

# 6. 看进度（一个 query 通常 40-60 分钟）
docker logs -f methyagent-agent1-query      # 搜索+过滤阶段
docker logs -f methyagent-agent1-download   # 下载+核验阶段

# 7. 取结果（见下文"查询结果与数据定位"）
```

浏览器打开 `http://localhost/` 可用 Web UI（提交、Review Queue 审批 manual_review 数据集、看板）。

---

## 本地安装（不用 Docker）

```bash
cd methylation_data_agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # Python 3.11+

cp .env.example .env && vim .env       # 同上填 key
set -a; source .env; set +a            # CLI 不自动加载 .env，需手动 source

python main.py --status                # 自检：应打印注册表统计
```

本地 CLI 与 Docker 常驻服务**不要同时跑同一注册表**（都会写 `registry/methyagent.db`）。

---

## 配置

`config/settings.yaml`（默认已可用，关键项）：

```yaml
llm:
  backend: zhipu            # zhipu | openai | anthropic | ollama | deepseek
  api_key_env: ZHIPU_API_KEY
  # model 从 .env 的 ZHIPU_MODEL 读（glm-4-flash 免费 / glm-4-air / glm-4-long…）

download:
  output_dir: ./data/methylation   # 本地 CLI 的下载目录
  max_concurrent: 5

agent1:
  pipeline: skills          # skills（默认）| legacy（回滚用旧固定管线）

registry:
  db_path: ./registry/methyagent.db
```

Docker 环境下 `DATA_DIR=/app/data` 会覆盖 output_dir（= 宿主机 `./data`）。
LLM 后端切换只需改 `backend` + 对应 `.env` key（openai 兼容端点可用
`OPENAI_BASE_URL`，工厂自动识别 bigmodel.cn）。

---

## 使用方法

### 提交查询

query 是自然语言（中英均可），说清 癌症类型 + 样本类型（平台/年份可选）。
支持直接给 accession。

```bash
# 方式 A：HTTP API（推荐，Docker 部署的生产路径）
curl -s -X POST http://localhost/query -H "Content-Type: application/json" \
  -d '{"query":"结直肠癌和非癌对照的血浆cfDNA甲基化数据","agent_type":"database"}'
# download 桶注册即 pending，下载 worker 自动下载

# 方式 B：CLI（本地或容器内）
python main.py --query "breast cancer WGBS methylation 2022-2023" --agent db-only
python main.py --query "下载GEO编号GSE124600的所有数据"
docker exec -w /app methyagent-agent1-query python main.py --query "..." --agent db-only
```

**两种方式的差异**：CLI（legacy 编排）跑完过滤后数据集停在 `awaiting_approval`，
需再触发下载：

```bash
sqlite3 registry/methyagent.db "UPDATE datasets SET download_status='pending'
  WHERE download_status='awaiting_approval' AND needs_review=0"
```

（manual_review 的保持待审是设计行为；也可在 Web UI Review Queue 逐个 approve。）

### CLI 其他命令

```bash
python main.py --query "..." --dry-run    # 只解析意图，不下载（调试）
python main.py --status                   # 注册表统计
python main.py --query "..." --verbose    # DEBUG 日志
python main.py --query "..." -o /data/x   # 自定义输出目录
```

### 监控进度

```bash
# task 状态（pending → running → done）
sqlite3 registry/methyagent.db "SELECT task_id,status FROM task_queue
  WHERE query!='__heartbeat__' ORDER BY created_at DESC LIMIT 3"

# 数据集流水
sqlite3 registry/methyagent.db "SELECT accession,event,substr(message,1,80),timestamp
  FROM download_log ORDER BY id DESC LIMIT 10"

docker logs -f methyagent-agent1-query --since 5m
```

完成标志：`[agent1 pipeline] Done — GEO N ok / M fail, ..., K excluded.`
核验结论行：`[verify] cols X vs Y gsms; map ...; groups ...`。

### 查询结果与数据定位

task 完成后结果在 `task_queue.result_json`（三桶：`datasets_downloaded` /
`datasets_review` / `datasets_excluded` 计数）。

```bash
# ① task 结果
sqlite3 registry/methyagent.db \
  "SELECT result_json FROM task_queue WHERE task_id LIKE '<task前8位>%'" | python3 -m json.tool

# ② task → GSE（⚠️ 必须走 query_dataset_map；datasets.task_id 是单值列，只存
#    第一个命中该 GSE 的 task，直接查会漏）
sqlite3 registry/methyagent.db \
  "SELECT accession FROM query_dataset_map WHERE task_id LIKE '<task前8位>%'"

# ③ GSE → 文件（local_path 是入口；容器 /app/data = 宿主机 ./data）
sqlite3 registry/methyagent.db "
  SELECT d.accession, d.download_status, d.local_path FROM query_dataset_map q
  JOIN datasets d USING(accession)
  WHERE q.task_id LIKE '<task前8位>%' AND d.download_status='done'"

# ④ 样本级：这个 task 要某 GSE 的哪些 GSM、病例/对照构成
sqlite3 registry/methyagent.db "
  SELECT s.sample_group, s.cancer, COUNT(*) FROM query_sample_map m
  JOIN samples s ON s.accession=m.accession AND s.gsm=m.gsm
  WHERE m.task_id LIKE '<task前8位>%' AND m.accession='<GSE>'
    AND m.verdict='download' GROUP BY 1,2"
```

数据目录布局：

```
data/
├── GSE124600/                  # 每个 GSE 一个目录
│   ├── *_umepm.txt.gz          #   保留的矩阵（LLM 按样本类型筛选后）
│   ├── GSE124600_query_subset.txt.gz   #   query-cancer 列子集（多癌数据集）
│   ├── GSE124600_series_matrix.txt(.gz)
│   └── sample_metadata.csv     #   GSM 级样本标注（人类可读）
├── GSE186007/
│   └── GSM7426116/*.cov.gz    # Tier-3 per-sample：每 GSM 一个子目录
├── quarantine/{GSE}/           # 核验失败隔离区（移动不删除）
└── query_logs/query_*_<task前8位>.csv   # 每次查询的逐 GSE 判定流水
```

---

## 数据处理管线（当前行为）

```
query → parse（规则 + LLM）→ intent
  → geo_search：NCBI esearch/esummary 确定性召回（同义词扩展、平台/年份过滤）
  → geo_filter（三态，只判相关性）：
      LLM 读 GSE 元数据 + 代表 GSM characteristics → 每个样本 include/exclude
      → GSE 级 verdict：download / manual_review / exclude
      → GSM 级判定写 query_sample_map（多 task 各自成行）
      → sample_metadata.csv + query_logs/*.csv
  → download worker（对 pending 数据集）：
      三级 tier：① series_matrix 有数据 → 下它；② 非 RAW 补充文件全下；
                ③ 都没有 → 逐 download-GSM 抓页面下 per-sample 文件
      → tar/zip 成员级提取（在垃圾预删之前）
      → 垃圾预删（空/README/p值表）+ LLM 按样本类型 keep/drop（保守兜底：
        无 LLM/解析失败/全拒 → 保留全部非垃圾 + manual_review）
      → Phase-2 核验：
          ① 样本列数 vs 该 task 的 download GSM 数（容差 max(5,10%)，
            跨文件聚合并集；唯一会触发 quarantine 的检查）
          ② GSM→列映射：列名含 GSM 号 → series_matrix !Sample_title 索引
            （提交者自定义列名如 Pcrc90 由此确定性解出）→ 低覆盖率挂 review
          ③ 疾病分组完整性（只报告进 notes）
      → ① 失败 → 文件移 quarantine/{GSE}/，outcome 回退 manual_review
      → cancer 子集：经映射精确选 query-cancer 列写 {GSE}_query_subset.txt.gz
```

设计规格：`skills/geo_filter/SPEC.md`（过滤规则，用户维护）、
`skills/geo_download/SPEC.md`（下载与核验）。

---

## Registry Schema（registry/methyagent.db）

| 表 | 粒度 | 说明 |
|----|------|------|
| `task_queue` | task | 提交的查询（status、result_json） |
| `datasets` | GSE | 主表：元数据、`recommended_action`（download/manual_review/exclude）、`download_status`（pending/downloading/done/failed/no_file/awaiting_approval/skipped）、local_path、reason 等 |
| `samples` | (GSE,GSM) | 每 GSM 的 source_name/group/cancer/characteristics_json |
| `query_sample_map` | (task,GSE,GSM) | **每次查询对每个样本的判定**（verdict + reason）；task↔GSE↔GSM 权威多对多表 |
| `query_dataset_map` | (task,GSE) | task↔GSE 映射（按 task 查 GSE 用它） |
| `download_log` | 事件 | 每 GSE 下载/核验事件流水 |

旧库打开自动迁移（`_migrate_schema`），无需手动操作。

## 项目结构

```
methylation_data_agent/
├── main.py                     # CLI 入口
├── docker-compose.yml          # 4 服务（override 文件为开发热载配置）
├── Dockerfile                  # python:3.11-slim 单镜像
├── config/
│   ├── settings.yaml           # 主配置
│   └── cancer_synonyms.yaml    # 癌种同义词 + 技术/液体活检词表
├── agents/
│   ├── agent1_pipeline.py      # skills 管线编排（daemon 主路径）
│   ├── database_agent.py       # legacy 编排（CLI 路径 / 回滚）
│   ├── literature_agent.py     # Agent 2
│   └── orchestrator.py         # LangGraph 图（CLI 用）
├── skills/
│   ├── geo_search/             # 确定性 GEO 召回
│   ├── geo_filter/             # 三态过滤（SPEC.md 为行为源）
│   ├── geo_download/           # 下载 + 核验（verify.py / archive_extract.py）
│   └── adaptive_evidence/      # manual_review 自取证（bind_tools agent）
├── tools/                      # geo_tools / tcga_tools / pubmed_tools /
│                               # download_tools（断点续传）/ parser_tools
├── registry/registry.py        # SQLite 注册表
├── api/                        # FastAPI（main.py + templates/）
├── scripts/agent_daemon.py     # 常驻 worker 入口（query/download/literature 模式）
├── state/ · utils/ · tests/
```

## 支持的数据类型

| 类型 | 平台 | 文件形态 |
|------|------|---------|
| Illumina 450K / EPIC | 芯片 | series_matrix / 合并 β 矩阵 .txt.gz（IDAT 等原始不保留） |
| WGBS / RRBS / MCTA-Seq 等 | 测序 | 合并矩阵 或 per-sample `.cov`/`.bed`/bsmap 文件（标 `needs_processing=merge_per_sample`） |
| TCGA | — | Level 3 β 值（公开无需 token；受控数据配 GDC_TOKEN） |

## 注意事项

- **改代码后必须重启容器**：`docker compose restart agent1-query agent1-download agent2`（bind mount 热载文件，但 daemon 常驻进程不重读代码）
- **NCBI 限流**：无 API key 3 req/s、有 key 10 req/s；连续抓 100+ GSM 页面可能触发 abuse-redirect（表现为静默 miss），大批量分批跑
- **磁盘**：per-sample 测序数据集单 GSE 可达 17GB，批量查询前 `df -h`
- **成本**：一次 query（~136 候选）消耗约 130 万 LLM token
- 下载引擎带断点续传（`.part` + Range），中断重跑自动续
- TCGA 公开数据（Level 3）无需 token；受控数据需 dbGaP 授权（`GDC_TOKEN`）
- `cancer_synonyms.yaml` 可独立扩充，加癌种同义词不用改代码
