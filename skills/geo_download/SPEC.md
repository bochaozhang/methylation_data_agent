# geo-download（GEO 下载与核验）Skill V2

适用对象：接收 `geo-filter` 的 `download_list`，下载甲基化文件、下载后核验、记录溯源，写回 State。
Tier 2（supplementary files）与 Tier 3（每 GSM supp 文件）下载**所有非 RAW 的文件**，再由 **LLM 按样本类型**（匹配 query，如血浆 cfDNA vs 组织）选择保留哪些、删除其余（先记 md5+原因）；仅用降级的 `inspect_matrix_head` 预删明显垃圾（README/p 值表）。这是三态 filter 之后**格式可用性判定的所在**（filter 不再判文件格式）。

> **Phase 1（当前）**：下载 + md5；Tier 2 下载后做**样本类型驱动的文件选择**（LLM 判定文件列对应样本是否匹配请求；保守兜底保留全部非垃圾 + manual_review）。`qc_passed` 暂按"下载成功 + md5"置位。
> **Phase 2（后续）**：四项核验中其余三项（样本列数 / GSM→列映射 / 疾病分组）+ 不通过 quarantine + outcome 回退。输出 schema 已预留字段（`files_failed_qc`、`files_discarded`、`outcome_final` 的 revert 值），Phase 2 只填实现不改契约。

## Scope

- Does: 下载 `download_list` 中记录的实际 GEO supplementary 文件（复用 `DownloadEngine` + `build_geo_download_tasks`），记 md5 + 溯源，输出 `download_results`。Tier 2 下载后用 LLM 按样本类型选择保留文件、删除其余；`inspect_matrix_head` 仅预删垃圾。
- Does NOT: 判断哪些 accession 该下（`geo-filter` 已定）；不重判；Phase 2 其余三项核验（列数/GSM 映射/疾病分组）尚未实现。

## 输入 State 字段

- `download_list`：`geo-filter` 输出的记录（三态 filter 的 download 桶，含 `accession` / `supplementary_files` / `download_tier` / `flags`；filter 不再判文件格式）。
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
3. **Tier 2/3 文件选择**（两步，Tier 2 与 Tier 3 下载的文件同等适用）：
   - **垃圾预过滤**（降级的 `inspect_matrix_head`）：只删明显非数据文件——空/README/二进制、p-value/logFC 差异表。**不再用值域当 A 级硬门**（整数 read-counts、0–100 score、带坐标列的 MCTA-Seq 矩阵都放行）。`series_matrix` 直接信任保留。
   - **LLM 按样本类型选文件**：一次 `llm.invoke`，喂入 query（`raw_query` + 请求 `sample_type` + `cancer_type`）、数据集设计、**目标样本摘要**（`sample_metadata.csv` 里标 `download` 的样本数 + `source_name`/`group` 分布）、以及每个候选文件的解压表头。LLM 按「文件列对应的样本类型是否匹配请求」（如血浆 cfDNA vs 组织）决定每个文件 keep/drop，输出 `{reasoning, files:[{name,keep,sample_type,reason}]}`。
   - **保守兜底**：无 LLM / 解析失败 / LLM 一个都没留 → 保留全部非垃圾文件并标 `manual_review`（绝不因 LLM 抖动而误删真数据）。被丢弃文件先记 md5+原因再删除。
4. 疾病亚集（Phase 2c）：在保留后的文件上做 query-cancer 列子集。
5. 按 accession 聚合：保留到 ≥1 个文件 → `download_success`；supp 全下载但全被丢弃 → `no_files`（notes 记丢弃原因）；下载失败 → `failed`。
6. 每文件记溯源（source_url、md5）。

## Phase 2 文件类型判断（核心：只保留可建模的「特征×样本甲基化矩阵」）

### 1. 什么算可用矩阵（A 级 = 文件级必要条件，非充分；样本级另见「样本级判断」）

满足以下全部才算 A 级（A 级是**可保留的必要条件**，不是充分条件——样本级不通过仍不保留）：

- 行 = 甲基化特征：探针（cg 号）/ 单 CpG / 区域（promoter、CpG island、DMR 等）/ 基因相关区域（如 promoter/gene body/CGI，需有坐标或明确区域定义，不接受无区域定义的基因汇总统计）；
- 列 = 样本：最终形态须能整理成「特征 × 样本」矩阵（合并矩阵每个 GSM 一列；或一文件一样本、能建立 GSM→文件映射并机械合并），且能把每列/每文件对回疾病分组；
- 值为下列任一种可建模的甲基化定量信号：
  - **标准甲基化水平**：β 值（0–1）、M 值、甲基化比例（0–1 或 0–100%）、或甲基化/非甲基化 read 计数对（可还原比例），标 `merge_scope=standard_level`（仍须做常规批次/平台评估）；
  - **assay-native 甲基化信号**：由甲基化选择性实验直接产生，经过文库量/测序深度或其他明确方法归一化，能在样本间定量比较，并可直接用于队列内特征筛选或模型构建。此类信号须标 `value_semantics=normalized_assay_signal`、`merge_scope=within_assay_only`，不得未经转换与 β/M/比例或其他技术平台直接混并。
- 形态（两种等价，区别只在下游处理，见输出 `data_form` / `needs_processing`）：
  - 已是多样本合并矩阵；或
  - 一组「每样本一份、机械合并即得矩阵」的甲基化值文件（如 Bismark `.cov`、`.bed` methylation calls、region × 样本甲基化矩阵；区别于第 2 节的 DMR/DMP 结果表）。合并虽是机械操作，但产物可能很大，通常需按 coverage/缺失过滤或聚合到区域。
- **小型预设 target panel 默认不进入下载清单**：默认以 500 个独立目标区域作为参考界线；少于 500 的 panel 通常 → lead（`targeted_panel`），≥500 也仍须满足上述矩阵、归一化和样本映射条件。500 是默认参考线而非单独裁决；若公开矩阵保留完整的 genome-scale/高维检测空间，而不是只保留论文最终筛出的 classifier markers，则按完整检测空间判断。

> M 值说明：M = log2(β/(1−β))，是 β 的 logit，可取任意实数（负=低甲基化，正=高甲基化）。β 与 M 代表同一甲基化水平，任一形态的矩阵都算可用；建模/差异分析常用 M，生物学解释常用 β。

### 2. 不算可用矩阵 / 不保留

- **芯片原始/中间态**：IDAT（芯片标准 raw，需独立 normalize 流程，本 agent 不保留，非不可用）、signal intensity 矩阵、detection p-value。
- **测序原始/需重分析**：fastq、BAM/CRAM、per-position signal/coverage bigWig（非甲基化值矩阵；**若疑似甲基化比例 bigWig 且为唯一甲基化形态可保留**）。
- **不可比较的单一计数/丰度**：未归一化、不能在样本间定量比较的 read/enrichment/barcode/marker count，不算 A 级；若给出可还原比例的配对计数，或满足 §1 的 normalized assay-native 信号，则不在此列。
- **小型预设 panel**：即使有逐样本 count/信号，默认只作靶向 marker/文章证据，→ lead（`targeted_panel`）；尺度和例外按 §1 的参考界线判断。
- **统计结果而非 per-sample 矩阵**：判定看**有无 per-sample 甲基化值列**——只给 group 均值/logFC/p 值、显著 CpG/marker list、火山图/ROC 数据、无 per-sample 列 → 不保留；若 DMR/DMP/marker 文件**附带每样本可建模甲基化信号列**，则那些列按矩阵收（差异统计列忽略）。
- **非目标修饰**：5hmC 不属于本任务目标，→ 不保留（`non_target_modification_5hmC`）。
- **非甲基化数据**：QC 报告、FastQC、比对统计、coverage summary、测序质量表。

### 3. 技术类型识别（文件判断的前提；GEO 字段都不可单信）

- 没有任何单一字段权威：`GPL` 只到平台（芯片能定 450K/EPIC/27k；测序只到测序仪，分不出 RRBS/WGBS/MCTA/MeDIP）；`gdsType` 是 curator 归的粗类、常错；`summary`/`extract protocol` 是作者自由文本，可能含糊或错。
- 做法：多源交叉（GPL + gdsType + extract protocol/summary），冲突时**以打开文件看到的形态和数值语义为最终裁决**。

### 4. 测序类必须逐文件打开核对

芯片 β/M 矩阵从文件说明通常能较可靠识别；**测序甲基化格式高度不统一，不能只看扩展名（.tsv/.txt/.bed/.gz）或技术名就判定**。下载前先用文件名/大小、README、extract protocol、GSE summary、压缩包文件清单判断；必要时临时取压缩包目录或前若干行做预检（预检只是 hint，真值以「下载执行」的下载后核验为准）。核对：

1. 行列方向：列是样本（或一文件一样本需合并）？还是坐标/统计表？
2. 值类型：是标准甲基化水平、normalized assay-native 信号，还是未归一化 count / 坐标 / p 值 / logFC？
3. 若为 assay-native 信号：归一化方法是否明确、样本间是否可比、是否须限制为 `within_assay_only`？
4. 聚合粒度：per-CpG / per-region / per-gene？候选空间是否足以支持数据驱动筛选？
5. 样本注释：能否把每列/每文件对回 GSM 与疾病分组？

### 5. 平台专档（速查）

- 450K/850K/EPIC：多样本 β/M 矩阵、或每样本 β（需机械合并）→ 下载；IDAT / signal intensity / detection p-value → 不下载。
- 测序类（RRBS/WGBS/MCTA/MeDIP 等）：region/CpG 的 β/M/比例/配对计数，或满足 §1 的高维 normalized assay-native 矩阵 → 下载；后者标 `within_assay_only`。BAM/CRAM/fastq/bigWig/不可比较的单一计数 → 不下载。
- panel/qMSP 类：单独记录靶向的基因/区域/探针/坐标/序列、样本类型、病例/对照数、文章性能指标；小型预设 panel 默认不下载、不作 discovery 矩阵，仅作为靶向标志物/文章证据进入 lead。达到 §1 参考规模且其逐样本矩阵满足 A 级其他条件时，才进入 download 判断。
- 下载后**其余三项**核验：样本列数 vs GSM 数、GSM→列映射、疾病分组可分。（文件「保留哪个」已由 Tier 2 LLM 按样本类型决定；值类型不再作为硬门。）
- 核验失败 → 移至 `{output_dir}/quarantine/{accession}/`，outcome 回退（`qc_failed_reverted_manual_review`）。
- tar 包优先按成员取；逐样本文件标 `needs_processing=merge_per_sample`。

## 核心原则（Phase 1）

1. Tier 2 下载所有非 RAW 的 supp 文件；再用 **LLM 按样本类型**选择保留哪些（匹配请求样本，如血浆 cfDNA vs 组织），删除其余；`inspect_matrix_head` 只预删垃圾，不作 A 级硬门。不整包拉 RAW.tar。
2. 下载后记 md5 + 溯源（source_url）；被丢弃文件也记 md5+原因，可追溯。
3. LLM 不可用 / 解析失败 / 全被拒 → 保守保留全部非垃圾文件并标 `manual_review`，绝不误删真数据。
4. 失败 accession 记 `outcome_final=failed`（或 `no_files`）+ notes，不静默吞错。
