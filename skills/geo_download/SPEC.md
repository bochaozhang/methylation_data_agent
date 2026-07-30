# geo-download（GEO 下载与核验）Skill

适用对象：接收 `geo-filter` 的 `download_list`，下载甲基化文件、下载后核验、记录溯源，写回 State。
Tier 2（supplementary files）下载**所有非 RAW 的 supp 文件**，再由 **LLM 按样本类型**（匹配 query，如血浆 cfDNA vs 组织）选择保留哪些、删除其余（先记 md5+原因）；仅用降级的 `inspect_matrix_head` 预删明显垃圾（README/p 值表）。

> **Phase 1（当前）**：下载 + md5；Tier 2 下载后做**样本类型驱动的文件选择**（LLM 判定文件列对应样本是否匹配请求；保守兜底保留全部非垃圾 + manual_review）。`qc_passed` 暂按"下载成功 + md5"置位。
> **Phase 2（后续）**：四项核验中其余三项（样本列数 / GSM→列映射 / 疾病分组）+ 不通过 quarantine + outcome 回退。输出 schema 已预留字段（`files_failed_qc`、`files_discarded`、`outcome_final` 的 revert 值），Phase 2 只填实现不改契约。

## Scope

- Does: 下载 `download_list` 中记录的实际 GEO supplementary 文件（复用 `DownloadEngine` + `build_geo_download_tasks`），记 md5 + 溯源，输出 `download_results`。Tier 2 下载后用 LLM 按样本类型选择保留文件、删除其余；`inspect_matrix_head` 仅预删垃圾。
- Does NOT: 判断哪些 accession 该下（`geo-filter` 已定）；不重判；Phase 2 其余三项核验（列数/GSM 映射/疾病分组）尚未实现。

## 输入 State 字段

- `download_list`：`geo-filter` 输出的记录（outcome=download，含 `accession` / `supplementary_files` / `files[]` / `flags`）。
- `output_dir`：保存目录。

## 输出 State 字段

```json
{
  "download_results": [
    {
      "accession": "GSExxxxxx",
      "files_downloaded": [
        {"name":"...","local_path":"...","size_bytes":0,"qc_passed":true,
         "data_form":null,"provenance":{"source_url":"","checksum_md5":""}}
      ],
      "files_failed_qc": [],
      "files_discarded": [
        {"name":"...","value_type":"non_methylation","reason":"...","md5":"...","source_url":"..."}
      ],
      "outcome_final": "download_success | failed | no_files",
      "flags": "继承自 geo-filter",
      "notes": ""
    }
  ],
  "download_log": "本次下载整体说明"
}
```

## Phase 1 执行流程

1. 对 `download_list` 每条记录按三级回退构建下载任务：
   - Tier 1：`series_matrix_has_data(acc)` 为真 → 下 series_matrix（GEO 编译的 β 矩阵）。
   - Tier 2：有 `supplementary_files` → `build_geo_download_tasks(rec, output_dir, download_all_non_raw=True)`，**下载所有非 RAW 的 supp 文件**（不再用 `_is_methylation_file` 关键词预过滤；RAW.tar 等始终排除）。
   - Tier 3：都没有 → 按 `download=true` 的 GSM 抓单样本 supp。
2. `DownloadEngine.download_many_sync(tasks)` 下载（含 md5、断点续传、并发）。
3. **Tier 2 文件选择**（两步）：
   - **垃圾预过滤**（降级的 `inspect_matrix_head`）：只删明显非数据文件——空/README/二进制、p-value/logFC 差异表。**不再用值域当 A 级硬门**（整数 read-counts、0–100 score、带坐标列的 MCTA-Seq 矩阵都放行）。`series_matrix` 直接信任保留。
   - **LLM 按样本类型选文件**：一次 `llm.invoke`，喂入 query（`raw_query` + 请求 `sample_type` + `cancer_type`）、数据集设计、**目标样本摘要**（`sample_metadata.csv` 里标 `download` 的样本数 + `source_name`/`group` 分布）、以及每个候选文件的解压表头。LLM 按「文件列对应的样本类型是否匹配请求」（如血浆 cfDNA vs 组织）决定每个文件 keep/drop，输出 `{reasoning, files:[{name,keep,sample_type,reason}]}`。
   - **保守兜底**：无 LLM / 解析失败 / LLM 一个都没留 → 保留全部非垃圾文件并标 `manual_review`（绝不因 LLM 抖动而误删真数据）。被丢弃文件先记 md5+原因再删除。
4. 疾病亚集（Phase 2c）：在保留后的文件上做 query-cancer 列子集。
5. 按 accession 聚合：保留到 ≥1 个文件 → `download_success`；supp 全下载但全被丢弃 → `no_files`（notes 记丢弃原因）；下载失败 → `failed`。
6. 每文件记溯源（source_url、md5）。

## Phase 2 待办（本次不实现）

- 下载后**其余三项**核验：样本列数 vs GSM 数、GSM→列映射、疾病分组可分。（文件「保留哪个」已由 Tier 2 LLM 按样本类型决定；值类型不再作为硬门。）
- 核验失败 → 移至 `{output_dir}/quarantine/{accession}/`，outcome 回退（`qc_failed_reverted_lead` / `qc_failed_reverted_manual_review`）。
- tar 包优先按成员取；逐样本文件标 `needs_processing=merge_per_sample`。
- 见 `~/skills/skills_geo-download_SKILL.md` 完整规格。

## 核心原则（Phase 1）

1. Tier 2 下载所有非 RAW 的 supp 文件；再用 **LLM 按样本类型**选择保留哪些（匹配请求样本，如血浆 cfDNA vs 组织），删除其余；`inspect_matrix_head` 只预删垃圾，不作 A 级硬门。不整包拉 RAW.tar。
2. 下载后记 md5 + 溯源（source_url）；被丢弃文件也记 md5+原因，可追溯。
3. LLM 不可用 / 解析失败 / 全被拒 → 保守保留全部非垃圾文件并标 `manual_review`，绝不误删真数据。
4. 失败 accession 记 `outcome_final=failed`（或 `no_files`）+ notes，不静默吞错。
