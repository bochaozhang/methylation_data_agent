# MethyAgent MCP 接口使用说明（上游 Agent 接入指南）

日期：2026-10-19
适用版本：MCP 分支 Phase 1+2（9 个 tool）
配套文档：docs/mcp_interface_plan.md（设计与实施报告）

---

## 1. 这是什么

MethyAgent 是一个 DNA 甲基化数据检索 agent：给它一个自然语言或结构化的检索意图，它会
在 GEO（和 TCGA/GDC）中召回候选数据集，用 LLM 按 SPEC 规则逐个判定可用性（download /
manual_review / exclude），把可用数据下载到本地，并在注册表中维护数据集与样本级归属。

你（上游 agent）通过 **MCP 协议**使用它：预览召回、提交任务、轮询状态、取回结果
（判定分组 + 数据文件在磁盘上的路径）。

**核心心智模型：**

- 一个 query task = 一次完整 pipeline（检索 → 逐数据集 LLM 判定 → 注册 → 异步下载），**约 1 小时**，立即返回 task_id
- **task 边界 = 子任务边界**：主数据（训练）和辅数据（验证）请拆成两个带 label 的 task 提交（见 §5.3）
- 返回的文件路径是**宿主机路径**，直接读文件即可；MCP 不传输文件内容（GB 级）

## 2. 接入方式

### Claude Code（本机）

```bash
claude mcp add methyagent -- \
  /home/ubuntu/bochaozhang/methylation_data_agent/.venv/bin/python \
  /home/ubuntu/bochaozhang/methylation_data_agent/mcp_server/__main__.py
```

### 其他 MCP 客户端（支持 stdio）

启动命令同上（任意工作目录可运行）。环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `MCP_API_BASE` | `http://localhost:80` | REST API 地址（远程部署时改这里） |
| `MCP_DATA_HOST_PATH` | `<repo>/data` | 数据目录的宿主机路径（路径翻译用） |
| `MCP_API_TIMEOUT` | `300` | 每请求超时秒数（dry-run 检索约 50s，勿低于 120） |

### 不用 MCP，直接 REST

MCP 是 REST 的薄适配层，所有能力都有等价 HTTP 端点（见 §8），curl/httpx 可直接调用。

## 3. Tool 一览

| Tool | 类型 | 用途 | 耗时 |
|---|---|---|---|
| `search_datasets` | 只读 | **dry-run**：parse + GEO 检索，返回候选 GSE 列表，不判定不入库 | ~50s |
| `submit_query` | 提交 | 提交一个完整任务（自由文本或结构化 intent） | 立即返回 |
| `submit_compound_query` | 提交 | 一次提交 main + auxiliary 两个带 label 的任务 | 立即返回 |
| `get_task_status` | 只读 | 任务状态 + 完成时的 result 摘要 | 快 |
| `get_task_result` | 只读 | 判定分桶（download/manual_review/exclude）+ 文件路径 | 快 |
| `cancel_task` | 写 | 取消任务：pending 立即取消；**running 协作式取消**（worker 在候选边界检查标志，秒级~2 分钟内停下） | 快 |
| `list_datasets` | 只读 | 浏览注册表（可按 download_status 过滤） | 快 |
| `get_dataset` | 只读 | 单个数据集全量记录（含路径 + LLM 判定证据） | 快 |
| `health` | 只读 | 后端心跳 + 注册表统计 | 快 |

## 4. 推荐工作流

```
① search_datasets 预览召回（便宜、无副作用）
        │  召回太少 → 调整 intent（放宽平台/年份/样本类型）
        ▼
② submit_query 或 submit_compound_query 提交
        ▼
③ get_task_status 轮询（done / failed / cancelled）
        ▼
④ get_task_result 取判定分组 + 文件路径
        ▼
⑤ 直接读磁盘上的数据文件与 sample_metadata.csv
```

## 5. 关键用法详解

### 5.1 dry-run：先看召回再决定提交

```
search_datasets(query="结直肠癌血浆cfDNA甲基化", max_candidates=20)
search_datasets(intent={...结构化...})          # 不走 LLM parse，完全确定性
```

返回：normalize 后的 intent、实际发给 NCBI 的 esearch 检索串、候选 GSE（accession /
title / platform / sample_count / year / PMID）+ 总召回数。**强烈建议先 dry-run**——完整
任务约 1 小时且每个候选都会消耗一次 LLM 判定。

### 5.2 单任务提交：自由文本 vs 结构化

**自由文本**（服务端 LLM parse，适合探索）：

```
submit_query(query="colorectal cancer plasma cfDNA methylation, CRC vs healthy")
```

**结构化**（推荐给 agent 调用方；确定性、不丢信息、不消耗 parse LLM）：

```
submit_query(intent={
  "mode": "semantic",
  "cancer_type": {"display": "colorectal cancer", "tcga_code": "COAD"},
  "sample_types": ["cfdna", "plasma", "serum"],
  "platform": null,              // null=不限；"450K"/"EPIC"/"WGBS"/"RRBS"
  "year_start": null, "year_end": null,
  "notes": "early screening, case vs healthy controls"
}, label="main")
```

intent 字段完整定义见 §7。

### 5.3 复合查询：训练数据 + 验证数据

上游主 agent 常见需求：主检索（如血浆 cfDNA 病例 vs 健康对照，做训练）+ 辅检索（如肿瘤
vs 癌旁组织 / TCGA COAD/READ，定义标志物空间，做验证）。**不要合成一条 query**（实测会丢
掉辅检索子句：合并检索串召回不了组织系列，且 TCGA 分支会因 sample_type=cfDNA 被跳过）。
正确姿势：

```
submit_compound_query(
  main={
    "intent": {"mode": "semantic",
               "cancer_type": {"display": "colorectal cancer", "tcga_code": "COAD"},
               "sample_types": ["cfdna", "plasma", "serum"]}
  },
  auxiliary={
    "intent": {"mode": "semantic",
               "cancer_type": {"display": "colorectal cancer", "tcga_code": "COAD"},
               "sample_types": ["tumor", "adjacent"]}
  }
)
→ {"main": {"task_id": "...", "label": "main", ...},
   "auxiliary": {"task_id": "...", "label": "auxiliary", ...}}
```

- 两个 task 各有独立检索串与判定标准；label 默认 `main` / `auxiliary`，可自定义
- 同一 GSE 被两边都要时：只下载一次，样本级归属见每个 GSE 目录下的 `sample_metadata.csv`
  （每个历史 task 一列，标注该 task 对每个样本的 download/not-download 判定）
- **注意**：辅任务样本类型含 tumor/adjacent 时会触发 TCGA/GDC 直查，TCGA 甲基化数据量可能
  达 GB 级，确认磁盘预算后再提交

### 5.4 轮询与取结果

```
get_task_status(task_id)      # pending → running → done | failed | cancelled
get_task_result(task_id)      # done 后：
```

`get_task_result` 返回：

```json
{
  "task_id": "...", "label": "main", "status": "done",
  "summary": {"datasets_downloaded": [...], "datasets_review": [...], "datasets_excluded": 126},
  "counts": {"download": 6, "manual_review": 4, "exclude": 126},
  "download": [        // 全量记录
    {"accession": "GSE124600", "title": "...", "platform": "450K",
     "sample_count": 340, "download_status": "done",
     "local_path": "/home/.../data/GSE124600/GSE124600_sumP_....txt.gz"}
  ],
  "manual_review": [...],       // 需人工确认的数据集（含判定证据见 get_dataset）
  "exclude": []                 // 默认只计数（136 个会撑爆上下文）；exclude_limit=N 可列出前 N 个
}
```

`manual_review` 的数据集需要人工确认后才进入下载（审批走 REST `POST /review/{acc}/approve`，
MCP 写操作是 Phase 3）。

### 5.5 找数据文件

```
get_dataset("GSE124600")   →  local_path + sample_metadata_path（宿主机路径）
list_datasets(status="done", limit=50)
```

每个 GSE 目录（`data/GSExxxxxx/`）通常包含：数据文件（series_matrix / 补充文件）、
`sample_metadata.csv`（样本级元数据 + 各 task 的逐样本判定）、`gsm_metadata_cache.json`。
下载是异步的：任务 done 后 `download_status` 可能仍在 `downloading`，用 `get_dataset` 追踪。

## 6. 语义与坑

| 事项 | 行为 |
|---|---|
| **去重** | 提交时若存在等价的 pending/running 任务（规范化 intent 相等，或自由文本 query 全等），返回既有 task_id 且 `deduplicated=true`（不新建）。想强制重跑传 `force=true`。已 done 的任务不会被去重命中 |
| 任务可协作式取消 | `cancel_task`：pending 立即取消；running → `cancel_requested`，worker 在每个候选判定边界检查标志并停止（在飞的单个候选判定需先完成，通常秒级~2 分钟）。已产生的部分 GSM 判定记录保留（任务作用域），但不会触发 register/下载。worker 崩溃时自愈逻辑会直接把 `cancel_requested` 落成 `cancelled`（不会复活已取消任务） |
| worker 故障自愈 | worker 重启会把遗留 `running` 的孤儿任务自动拉回 `pending` 重跑 |
| verdict 随机性 | filter 判定是 LLM 调用（无 verdict cache），同一查询两次跑的边缘判定可能不同（download↔manual_review 浮动属正常） |
| 已下载数据不重下 | 重复查询同一 GSE 零重复下载（dest-exists 跳过 + 状态不降级） |
| TCGA | 样本类型为液体（cfdna/plasma/serum/wbc/whole_blood）时自动跳过 TCGA；组织类型（tumor/adjacent/normal）才走 GDC |
| 检索失败静默 | 已知限制：代理/网络故障时检索返回 0 候选，任务会"成功"完成但结果为空——结果为 0 时请检查 `health` 与 dry-run（修复在计划中） |

## 7. 结构化 intent 字段参考

| 字段 | 类型 | 取值 / 说明 |
|---|---|---|
| `mode` | str | `"semantic"`（默认）或 `"accession"`（配合 accessions.geo） |
| `accessions.geo` | list | `["GSE124600", ...]`，accession 模式直接按编号取 |
| `cancer_type` | obj | `{"display": "colorectal cancer", "tcga_code": "COAD"}`；tcga_code 驱动同义词扩展与 TCGA 直查 |
| `platform` | str/null | `"450K"` / `"EPIC"` / `"WGBS"` / `"RRBS"` / null（不限） |
| `data_type` | str/null | `"array"` / `"sequencing"` / `"both"` / null |
| `sample_types` | list | `cfdna` `plasma` `serum` `whole_blood` `wbc` `tumor` `adjacent` `normal` 等；同时含液体+组织 = both focus |
| `sample_type_detail` | str | 自由文本补充 |
| `year_start` / `year_end` | int/null | 年份过滤 |
| `extra_keywords` | list | 附加检索词（会进检索串） |
| `notes` | str | 特殊说明（给 filter LLM 的上下文，如 "case vs healthy only"） |

规范：intent 提交后服务端会先 normalize 再落库；normalize 是幂等的，重放安全。

## 8. REST 等价接口（不用 MCP 时）

| MCP tool | HTTP |
|---|---|
| search_datasets | `POST /search/dry-run` `{query|intent, max_candidates}` |
| submit_query | `POST /query` `{query|intent, agent_type, label, force}` |
| get_task_status | `GET /tasks/{id}` |
| get_task_result | `GET /tasks/{id}` + `GET /query-datasets/{id}`（+`/samples`） |
| cancel_task | `POST /tasks/{id}/cancel` |
| list_datasets | `GET /datasets?status=&limit=` |
| get_dataset | `GET /datasets/{accession}` |
| health | `GET /health` |

注意：REST 返回的路径是容器路径（`/app/data/...`），对应宿主机 `<repo>/data/...`（compose
挂载）；MCP 层已自动翻译。manual_review 审批：`POST /review/{accession}/approve|reject`。

---

有问题先看 `docs/mcp_interface_plan.md`（设计取舍与已知问题清单）。
