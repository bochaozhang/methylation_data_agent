# geo-download（GEO 下载与核验）Skill

适用对象：接收 `geo-filter` 的 `download_list`，下载 A 级甲基化文件、下载后核验、记录溯源，写回 State。
本 skill 只下载 `geo-filter` 标记为 `download=true` 的文件（Phase 1：按 GEO supplementary_files 实际文件下载 + md5）。

> **Phase 1（当前）**：下载 + md5，**不做下载后内容核验**。`qc_passed` 暂按"下载成功 + md5"置位。
> **Phase 2（后续）**：四项核验（值类型 / 样本列数 / GSM→列映射 / 疾病分组）+ 不通过删文件/quarantine + outcome 回退。输出 schema 已为 Phase 2 预留字段（`files_failed_qc`、`outcome_final` 的 revert 值），Phase 2 只填实现不改契约。

## Scope

- Does: 下载 `download_list` 中记录的实际 GEO supplementary 文件（复用 `DownloadEngine` + `build_geo_download_tasks`），记 md5 + 溯源，输出 `download_results`。
- Does NOT: 判断哪些该下（`geo-filter` 已定）；不重判；Phase 1 不核验文件内容。

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
      "outcome_final": "download_success | failed",
      "flags": "继承自 geo-filter",
      "notes": ""
    }
  ],
  "download_log": "本次下载整体说明"
}
```

## Phase 1 执行流程

1. 对 `download_list` 每条记录，用 `build_geo_download_tasks(rec, output_dir)` 由实际 `supplementary_files` 构建下载任务（`_is_methylation_file` 过滤；无则回退 series_matrix）。
2. `DownloadEngine.download_many_sync(tasks)` 下载（含 md5、断点续传、并发）。
3. 按 accession 聚合：全部成功 → `outcome_final=download_success`；否则 `failed`，notes 记错误。
4. 每文件记溯源（source_url、md5）。

## Phase 2 待办（本次不实现）

- 下载后四项核验：值类型（β/M/比例/配对计数）、样本列数 vs GSM 数、GSM→列映射、疾病分组可分。
- 核验失败 → 删文件（或移至 `{output_dir}/quarantine/{accession}/`），outcome 回退（`qc_failed_reverted_lead` / `qc_failed_reverted_manual_review`）。
- tar 包优先按成员取；逐样本文件标 `needs_processing=merge_per_sample`。
- 见 `~/skills/skills_geo-download_SKILL.md` 完整规格。

## 核心原则（Phase 1）

1. 只下载 `download=true` 的文件（Phase 1 按实际 supplementary_files 的甲基化文件）；不整包拉 RAW.tar。
2. 下载后记 md5 + 溯源（source_url）。
3. 失败 accession 记 `outcome_final=failed` + notes，不静默吞错。
