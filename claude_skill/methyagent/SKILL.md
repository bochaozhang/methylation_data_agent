---
name: methyagent
description: 通过 MCP 使用本机/远程 MethyAgent 数据采集系统（GEO/TCGA 甲基化数据检索→LLLM 过滤→下载→注册表）。当用户要找甲基化数据集（如"找结直肠癌 cfDNA 甲基化数据"）、查任务进度/结果、审批待审数据集、或定位已下载数据文件时使用。前提：methyagent MCP server 已连接。
---

# MethyAgent 数据采集 Skill

MethyAgent = DNA 甲基化数据检索 agent。一个 query 任务 ≈ 1 小时（检索 → 逐数据集
LLM 判定 → 注册 → 异步下载），提交后立即返回 task_id。**先 dry-run 预览，再提交**。

## 核心流程

```
① search_datasets      预览召回（~10-15s，免费，无副作用）
② submit_query         或 submit_compound_query（训练/验证拆两个带 label 的子任务）
③ get_task_status      轮询 pending→running→done/failed
④ get_task_result      判定分桶（download / manual_review / exclude）+ 文件路径
⑤ manual_review 项 → list_awaiting_approval → approve_dataset / reject_dataset
⑥ get_dataset / list_datasets   拿宿主机文件路径，直接读盘上的数据文件
```

## 关键规则

- **复合需求拆开提**：主数据（训练）与辅数据（验证）用 `submit_compound_query`
  （label=main/auxiliary），**绝不合并成一条 query**（合并会丢辅检索子句）。
- **优先结构化 intent** 而非自由文本（确定性、不丢信息）：`cancer_type{display,
  tcga_code}` + `sample_types[]`（cfdna/plasma/serum/tumor/adjacent/normal）+
  platform（450K/EPIC/WGBS/RRBS 或 null）。
- **重复提交自动去重**（`deduplicated=true` 即已有等价任务在跑）；队列满载返回 429。
- **取消**：pending 立即；running 协作式（worker 在候选边界停，秒级~2 分钟）。
- **结果语义**：`disease_groups`/`sample_level_annotation` 是下游资格判定输入；
  verdict 每次重跑都会重新判定（SPEC 更新即时生效，属设计）。
- **样本 cancer=unclear** 且标题带提交者编码（如 `COAD.ct-N`）时：
  `decode_sample_titles(accession)`（先 dry-run 看提案，确认后 apply=True）。
- 数据文件**不走 MCP 传输**——路径是宿主机路径，直接读文件；每 GSE 目录下的
  `sample_metadata.csv` 是样本级标注表。
- TCGA 数据走设计信任（direct download 判定），组织类查询自动触发 GDC 直查。

## 工具速查（13 个）

| 场景 | 工具 |
|---|---|
| 预览/探索 | `search_datasets`（dry-run）|
| 提交 | `submit_query` / `submit_compound_query` |
| 跟踪 | `get_task_status` / `get_task_result` / `cancel_task` |
| 审批 | `list_awaiting_approval` / `approve_dataset` / `reject_dataset` |
| 数据 | `list_datasets` / `get_dataset` / `decode_sample_titles` |
| 其他 | `health` |

详细字段参考、语义与坑、REST 等价端点：仓库 `docs/mcp_usage_guide.md`。
