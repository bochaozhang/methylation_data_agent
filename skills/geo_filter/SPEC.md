# GEO 数据检索注意事项 v4

适用对象：癌症早筛 AI Scientist 中负责 GEO 公共数据检索的 agent。

本文档只针对 GEO 检索。GEO 是目前主要数据来源，但不是唯一来源；CFEA、EWAS Data Hub、TCGA/Xena、ArrayExpress/ENA、NGDC/GSA/OMIX、SRA、文章补充表等也可能有重要数据，需要另行检索或补充。

> 要点：只下载 A 级（特征×样本 甲基化矩阵，且样本级须通过）；其余进 lead / exclude / manual_review，判定看值、不看文件名/扩展名。

## GEO 检索总原则

- 检索目标是找到“可用于癌症早筛液体活检方法开发的数据”，不是找到所有癌症甲基化相关数据。
- GEO 中标题命中不代表数据可用，必须看 Summary、Overall design、Sample characteristics 和关联文章。
- GEO 判断不能只看 GSE 页面。必须先读数据集页面的 Summary 和 Overall design；如果信息不够，必须点开 GSM sample 页面，检查 Sample type、Source name、Characteristics、Extracted molecule、Molecule、Treatment protocol、Extract protocol 等字段。很多关键判断只在 GSM 页面出现。
- GEO 页面信息经常不完整，必须追踪 PMID/DOI，到原文 Methods、Data availability、Supplementary tables 中确认样本类型、病例对照、处理状态和文件说明。
- 文章或 GEO Summary 中提到 cfDNA/血液验证，不代表 GEO 公开了 cfDNA 数据；有些研究只公开组织发现数据，血液验证只在文章中以 qPCR、panel 或 marker 结果呈现。必须看 GSM 样本类型，确认 GEO 实际提供的是什么数据。
- 不要只搜 GEO DataSets，也要搜 GEO Series/GSE、Sample/GSM、Platform/GPL。很多数据只有 Series，没有被整理成 DataSet。
- 关键词只能用于召回候选数据，不能用关键词命中直接决定纳入或排除。最终判断必须理解上下文语义：样本实际来自哪里、测了什么、公开了哪些文件、哪些样本可用。
- 不要只搜癌种全称，要同时搜缩写、病理类型、TCGA 项目名、癌前病变和常见英文写法。
- 不要只搜 `DNA methylation`，还要搜 450K、850K、EPIC、RRBS、WGBS、bisulfite sequencing、MCTA、MeDIP、methylome、methylation profiling 等技术词。
- 不要只搜 `cfDNA`，还要搜 cell-free DNA、ctDNA、circulating tumor DNA、plasma、serum、blood、liquid biopsy、non-invasive、early detection、screening、diagnosis 等词。
- 中文语境下的“血浆”“血清”“游离 DNA”“循环肿瘤 DNA”在英文文章中可能对应 plasma DNA、serum DNA、cfDNA、ctDNA，不要漏掉。
- 第一步检索可以宽，后续筛选必须严格；不要一开始加太多 NOT 条件，否则可能漏掉有用数据。
- 页面判断通过 ≠ 可下载。还必须先过「样本级判断」、再过「文件类型判断」，按下述四态之一决定每条候选数据：
  - **exclude**：样本级明确无用、无参考价值（细胞系/类器官/动物模型/不可拆的非目标癌种/纯图或无甲基化值的 PDF 等）→ 不下载、不进线索清单，仅记 accession+原因作审计；
  - **download**：样本级判断通过（非 exclude）且 该 GSE 有 ≥1 个 A 级文件（A 级定义见「文件类型判断 §1」）→ 只下载那几个 A 级文件（不是整个 GSE 的全部 supplementary），进入**下载清单**；
  - **lead**：相关**且有参考价值**但不可下载（无 A 级文件、只在文章里有、只有 raw/marker list）→ 进入**线索清单**，不下载。（case-only/pooled/组织 only 等样本级受限的情况，若有 A 级文件见「需要逐条标注」、多为 download+flag，无 A 级文件才进 lead。）
  - **manual_review**：信息不足或冲突（sample type 查过 GEO 与文章仍笼统、GEO 与文章标签冲突、文件形态无法判定）→ 暂不下载，标人工复核。
- 每条候选数据都要落在上述四态之一并记录原因，不能只输出 accession 列表。

## 检索词设计

以结直肠癌为例，不能只搜：

```text
colorectal cancer DNA methylation
```

应扩展为：

```text
(CRC OR "colorectal cancer" OR "colorectal carcinoma" OR "colon cancer" OR "rectal cancer" OR COAD OR READ OR adenoma OR "advanced adenoma")
AND
("DNA methylation" OR methylome OR "methylation profiling" OR 450K OR 850K OR EPIC OR RRBS OR WGBS OR "bisulfite sequencing")
AND
(human OR "Homo sapiens")
```

如果重点找液体活检数据，再加血液相关词：

```text
(CRC OR "colorectal cancer" OR "colon cancer" OR "rectal cancer" OR adenoma OR "advanced adenoma")
AND
(cfDNA OR "cell-free DNA" OR ctDNA OR "circulating tumor DNA" OR plasma OR serum OR blood OR "liquid biopsy" OR "non-invasive")
AND
(methylation OR methylome OR 450K OR 850K OR EPIC OR RRBS OR WGBS OR "bisulfite sequencing")
```

不同癌种都要建立同义词表。例如：

- 肺癌：lung cancer、lung carcinoma、NSCLC、SCLC、LUAD、LUSC、pulmonary nodule、early-stage lung cancer。
- 肝癌：liver cancer、hepatocellular carcinoma、HCC、LIHC。
- 卵巢癌：ovarian cancer、ovarian carcinoma、OV、HGSC、serous ovarian cancer、endometrioid、clear cell、mucinous。
- 胃癌：gastric cancer、stomach cancer、gastric carcinoma、GC。
- 食管癌：esophageal cancer、oesophageal cancer、ESCC、EAC、esophageal squamous cell carcinoma。
- 结直肠癌：CRC、colorectal cancer、colon cancer、rectal cancer、COAD、READ、adenoma、advanced adenoma。

## 候选数据必须检查的内容

- GEO 层级信息：GSE 是研究/系列页面，GSM 是单个样本页面，GPL 是平台页面；判断数据可用性时，GSE 和 GSM 都要看，GPL 用于确认平台。
- 物种：只保留 human/Homo sapiens；小鼠、动物模型默认排除。
- 样本类型：cfDNA、ctDNA、plasma、serum、whole blood、WBC、PBMC、tumor tissue、adjacent tissue、normal tissue、ascites、pleural effusion、BALF 等要分清。
- GSM 页面字段：重点看 Sample type、Source name、Characteristics、Extracted molecule、Treatment protocol、Extract protocol、Data processing、Supplementary file。
- 疾病类型：目标癌种、癌前病变、良性疾病、健康人、其他癌种是否能区分。
- 样本量：病例数、健康对照数、良性/癌前病变数、癌旁/正常组织数。
- 分期和处理状态：是否早期、是否术前、是否治疗前；治疗后样本不能和治疗前混用。
- 是否为原发癌：转移灶、复发样本、耐药样本要单独标记，默认不用于早筛数据整理。
- 技术类型：450k、850k/EPIC、27k、RRBS、WGBS、MCTA、MeDIP、panel、qMSP 等。
- 文件类型：normalized beta matrix、IDAT、average beta、raw intensity、detection p-value、bed、bigWig、fastq、marker list 等（仅用于识别与记录，是否下载以「文件类型判断」为准）。
- 是否有样本级注释：必须能把每个 GSM 对应到疾病状态、样本类型、处理状态；只有总样本数不够。**来源不限**——GSM `characteristics` 取不到时，优先用 GSM 标题前缀 + 文章 Methods/补充样本表/样本编号表把每个样本逐个对回分组（abstract 多只到队列级，不足以逐样本映射）；能对回即算有注释、可判 download/lead，仍不能逐样本映射时记 manual_review。
- 数据是否可直接使用：是否 A 级（特征×样本甲基化矩阵）——判定见「文件类型判断」。
- GEO 注释和文章是否一致：如果不一致，以原文 Methods、补充表、样本编号说明为准，并记录冲突。

## 样本级判断

样本层面决定一条候选数据是优先、排除还是需逐条标注；下载门槛统一为 A 级（见「文件类型判断」），下面四节只调样本严格度与下载动作。**冲突时的优先级：逐条标注（显式裁定）> 默认不下载 > 调档 > 优先下载**——逐条标注点名的情况（27k、case-only、pooled、sequencing 跨平台等）覆盖调档的一般门槛；默认不下载（细胞系/动物等）覆盖优先下载的偏好。

### 优先下载的数据

- 血浆/血清 cfDNA 或 ctDNA，且有癌症病例和健康人/良性病变/癌前病变对照。
- 早期、术前、治疗前样本。
- 癌组织 vs 癌旁/正常组织，适合寻找肿瘤来源甲基化标志物。
- 健康人 cfDNA、全血、WBC、PBMC、正常组织，可用于判断血细胞背景和组织特异性。
- 样本量较大、样本注释清楚、有关联文章的数据。
- 同一个研究中能明确拆分目标癌种、正常对照、良性病变、癌前病变的数据。

### 默认不下载的数据

- 细胞系，例如 HCT-116、A549、NCI-H3122、CL1-5 等。
- 类器官、PDX、xenograft、动物模型、小鼠样本。
- 体外培养体系、药物处理、DNMTi 处理、辐射处理、基因编辑处理后的样本。
- 治疗后样本、复发样本、耐药模型、药物疗效预测队列；如果能单独提取治疗前 baseline 样本，可以只下载治疗前部分。
- 转移灶样本，除非任务明确需要；早筛数据整理中一般只保留原发癌、癌前病变和对照。
- 腹水、胸水、肺泡灌洗液等非血浆/血清样本，除非任务明确需要；不能和血浆 cfDNA 混用。
- 非目标癌种且无法拆分的数据。
- 目标癌种样本数过少且无特殊参考价值的数据（组织参考线 <10 例；cfDNA 本就稀缺、不按此卡，见「按样本类型调档」）。
- 以上默认 outcome=exclude；落在「需要逐条标注」的情况按该节裁定。

### 需要逐条标注下载动作的数据

下列数据不能套用默认规则，必须**逐条注明 outcome（download / lead / exclude / manual_review）及原因**：

- pooled cfDNA：A 级 → **download**，作为外部参考池，flag「pooled：无样本独立性，不能当独立样本训练」；非 A 级 → lead。
- 只有病例、没有健康对照的 cfDNA：A 级 → **download**，作为外部病例参考池，flag「case_only：无对照，不能独立做 case-control」。
- 只有健康对照、没有病例的 cfDNA：A 级 → **download** 作 cfDNA 背景参考，flag「control_only：无病例，不能独立训练诊断模型」。
- 只有组织、没有血液：A 级组织 → **download**（用于标志物发现），flag「组织 only，不能证明血液可检测」。
- 只有全血/WBC：A 级 → **download** 作血细胞背景参考，flag「非健康 cfDNA 对照」。
- 27k 数据：A 级但分辨率低；组织侧 → **lead**（有 450K/EPIC/测序替代时不优先）；cfDNA 侧罕见，若 A 级且无替代 → download。
- RRBS/WGBS/MCTA/MeDIP（A 级矩阵）：→ **download**，flag「跨平台，勿与其他平台直接混并，需批次校正」。
- 只有 fastq / BAM：非 A 级，本 agent 不下载 → **lead**（raw_only）；即使高价值也由独立的 raw-data 重处理流程承接，不计入本规则的 download。
- 泛癌/混合癌种：**只取目标癌种样本**。能按样本编号拆出目标癌种 → **download**，并在 sample manifest 标目标子集（合并矩阵整文件下载、下游只用目标样本列；per-sample 文件则只下目标样本的文件）+ flag；目标癌种例数过少（组织参考 <10 例；cfDNA 可低于 10 但需 flag）且无特殊参考价值 → **lead**；拆不出目标癌种 → **lead**。
- 数据锁定、需申请、当前无法获得但相关：→ **lead**（locked），记 access 方式供后续申请。
- GEO 中 sample type 写得笼统、或 GEO 与文章标签冲突：→ **manual_review**，查清前不下载。

### 按样本类型调档（门槛不变 = A 级，按稀缺度调严格度）

- **组织（数据多、可挑）**：A 级甲基化矩阵、样本量/注释/对照/治疗前达标 → **download**。**芯片组织（450K/850K/EPIC β/M）数据多、严格挑**——样本量过小、注释差、无特殊价值 → **lead**；**测序组织（region/CpG 矩阵，须值为甲基化水平或可还原比例）较稀缺、样本量可适度从宽**，但须 flag 跨平台、不与其他平台直接混并（见「需要逐条标注」），且能覆盖芯片探针之外的 CpG 位点。
- **cfDNA / 液体活检（数据少）**：只要 A 级甲基化矩阵 + 有病例与对照 → **download**，不限技术（RRBS/WGBS/MCTA 等测序矩阵都收），不限样本量、容忍非常规 region 粒度（case-only 见「需要逐条标注」，A 级 → download + flag）。无 A 级文件 → **lead**。
- 注：不可还原为甲基化比例的单一计数（含 panel/富集计数）一律**不进入下载清单**，不因 cfDNA 稀缺而放宽为 A 级；无参考价值 → exclude，仅有文章/靶向证据价值（文章报告的 marker，非计数文件本身）→ lead（article_only）；不作为 discovery 矩阵。

## 文件类型判断（核心：只下载已是「特征×样本 甲基化矩阵」的数据）

### 1. 什么算可用矩阵（A 级 = 文件级必要条件，非充分；样本级另见「样本级判断」）

满足以下全部才算 A 级（A 级是**可下载的必要条件**，不是充分条件——样本级不通过仍不 download）：

- 行 = 甲基化特征：探针（cg 号）/ 单 CpG / 区域（promoter、CpG island、DMR 等）/ 基因相关区域（如 promoter/gene body/CGI，需有坐标或明确区域定义，不接受无区域定义的基因汇总统计）；
- 列 = 样本：最终形态须能整理成「特征 × 样本」矩阵（合并矩阵每个 GSM 一列；或一文件一样本、能建立 GSM→文件映射并机械合并），且能把每列/每文件对回疾病分组；
- 值 = **甲基化水平**：β 值（0–1）、M 值、甲基化比例（0–1 或 0–100%）、或 甲基化/非甲基化 read 计数对（可还原比例）；
- 形态（两种等价，区别只在下游处理，见输出 `data_form` / `needs_processing`）：
  - 已是多样本合并矩阵；或
  - 一组「每样本一份、机械合并即得矩阵」的甲基化值文件（如 Bismark `.cov`、`.bed` methylation calls、region × 样本 甲基化矩阵；区别于第 2 节的 DMR/DMP 结果表）。合并虽是机械操作，但产物可能很大，通常需按 coverage/缺失过滤或聚合到区域。

> M 值说明：M = log2(β/(1−β))，是 β 的 logit，可取任意实数（负=低甲基化，正=高甲基化）。β 与 M 代表同一甲基化水平，任一形态的矩阵都算可用；建模/差异分析常用 M，生物学解释常用 β。

### 2. 不算可用矩阵 / 不下载（不进入下载清单）

- **芯片原始/中间态**：IDAT（芯片标准 raw，需独立 normalize 流程，本 agent 不下载，非不可用）、signal intensity 矩阵、detection p-value。
- **测序原始/需重分析**：fastq、BAM/CRAM、per-position signal/coverage bigWig（非甲基化值矩阵；**若疑似甲基化比例 bigWig 且为唯一甲基化形态 → manual_review 打开确认，不自动 lead**）。
- **不可还原为甲基化比例的单一计数/丰度**：文件每个特征只给一个 read/enrichment count、没有配对的 methylated/unmethylated 计数（如纯 barcode-count、marker-count、富集计数，含 panel/富集类方法）——无法换算成甲基化水平，不能当矩阵用。若文件给出配对的 methylated/unmethylated 计数（可还原比例），属第 1 节 A 级，不在此列。
- **统计结果而非 per-sample 矩阵**：判定看**有无 per-sample 甲基化值列**——只给 group 均值/logFC/p 值、显著 CpG/marker list、火山图/ROC 数据、无 per-sample 列 → 不下载；若 DMR/DMP/marker 文件**附带每样本 β/M 值列**，则那些列按矩阵收（差异统计列忽略）。
- **非甲基化数据**：QC 报告、FastQC、比对统计、coverage summary、测序质量表。
- **非数据**：纯图片、临床/表型表（无甲基化值）。含 marker/位点表的补充 PDF 不在此列——走 lead=article_only（见「文章反向追踪」）。

### 3. 技术类型识别（文件判断的前提；GEO 字段都不可单信）

- 没有任何单一字段权威：`GPL` 只到平台（芯片能定 450K/EPIC/27k；测序只到测序仪，分不出 RRBS/WGBS/MCTA/MeDIP）；`gdsType` 是 curator 归的粗类、常错；`summary`/`extract protocol` 是作者自由文本，可能含糊或错。
- 做法：多源交叉（GPL + gdsType + extract protocol/summary），冲突时**以打开文件看到的形态为最终裁决**。

### 4. 测序类必须逐文件打开核对

芯片 β/M 矩阵从文件说明通常能较可靠识别；**测序甲基化格式高度不统一，不能只看扩展名（.tsv/.txt/.bed/.gz）或技术名就判定**。下载前先用文件名/大小、README、extract protocol、GSE summary、压缩包文件清单判断；必要时临时取压缩包目录或前若干行做预检（预检只是 hint，真值以「下载执行」的下载后核验为准）。核对：

1. 行列方向：列是样本（或一文件一样本需合并）？还是坐标/统计表？
2. 值类型：是甲基化水平，还是 read count / 坐标 / p 值 / logFC？
3. 聚合粒度：per-CpG / per-region / per-gene？
4. 样本注释：能否把每列/每文件对回 GSM 与疾病分组？

### 5. 平台专档（速查）

- 450K/850K/EPIC：多样本 β/M 矩阵、或每样本 β（需机械合并）→ 下载；IDAT / signal intensity / detection p-value → 不下载。
- 测序类（RRBS/WGBS/MCTA/MeDIP 等）：region/CpG 矩阵、`.cov`、`.bed` calls、region × 样本矩阵 → 下载；BAM/CRAM/fastq/bigWig/单一不可还原计数 → 不下载。
- panel/qMSP 类：单独记录靶向的基因/区域/探针/坐标/序列、样本类型、病例/对照数、文章性能指标；若文件只给单一 count/信号、无法还原甲基化比例，标 `count_or_signal_not_methylation`、`targeted_not_discovery`，不下载、**不作 genome-wide discovery matrix（可作为靶向标志物/文章证据参与后续 panel 验证）**。

## 文章反向追踪

GEO 信息经常不够，必须找文章。至少做以下检查：

- GEO 页面有 PMID/DOI 时，打开原文和补充材料。
- 没有 PMID/DOI 时，用 GSE 编号、标题、作者在 PubMed、Google Scholar 或期刊网站搜索。
- 在文章中查 Methods、Cohort description、Data availability、Supplementary table、Sample information。
- 确认样本是否治疗前、是否早期、是否原发癌、是否血浆/血清 cfDNA。
- 确认文章中的 cfDNA/血液验证是否真的对应 GEO 公开数据；如果 GEO 只公开组织数据，必须标记为组织数据或文章标志物线索。
- 确认 GEO 中的样本数量是否和文章一致。
- 如果文章列出了具体样本编号，如哪些是腺瘤、哪些是转移灶、哪些是健康人，要记录这些编号。
- 如果 GEO 标签和文章标签冲突，要标记“需人工复核”，不能直接进入下载清单。
- 如果文章只有标志物结果，没有全量矩阵，也要记录为文章标志物线索，不要误认为有可用 GEO 数据。

## 输出格式要求

每条候选数据（**以 GSE 为单位**）输出一条记录，按 `outcome` 分入**下载清单**、**线索清单**或**排除审计**，`manual_review` 单列。字段如下：

```json
{
  "source": "GEO",
  "accession": "GSE（一条候选 = 一个 GSE；GSM 进 sample manifest，GPL 进 platform）",
  "title": "",
  "pmid": "",
  "doi": "",
  "cancer_type": "",
  "sample_type": "tissue / cfDNA / whole_blood / ...",
  "disease_groups": "",
  "sample_size": "",
  "stage_or_treatment_status": "",
  "technology": "",
  "platform": "GPLxxxx",
  "sample_level_annotation": "yes/no/unclear",
  "annotation_source": "GSM_characteristics / sample_title / paper_table / mixed / unclear",
  "outcome": "download / lead / exclude / manual_review",
  "files": [
    {
      "name": "supp 文件名或 URL",
      "is_A_level": true,
      "download": true,
      "data_form": "merged_beta_matrix / per_sample_calls / region_matrix / paired_counts",
      "needs_processing": "none / merge_per_sample / coverage_filter_or_aggregate",
      "gsm_to_column_mapping": "yes/no/partial",
      "reason": ""
    }
  ],
  "lead_type": "no_A_file / article_only / sample_limited / raw_only / locked / 其他",
  "exclude_reason": "cell_line / animal_model / non_target_unsplittable / no_reference_value / 其他",
  "flags": "case_only / pooled / 跨平台 / 组织only / 无对照 / 血细胞背景 / 泛癌需拆分 等",
  "reason": "",
  "notes": ""
}
```

规则：

- `outcome=download` 当且仅当**样本级通过 且** `files[]` 中至少一个 `download=true`；此时**只下载**这些文件，不下载该 GSE 的其余 supplementary。
- `outcome=lead` / `exclude` 时 `files[]` 须列出该 GSE 的关键文件、标 `download=false` + `reason`（供审计，不留空）；`lead` 填 `lead_type`（**仅收有参考价值的**），`exclude` 填 `exclude_reason`（细胞系/动物模型/不可拆非目标癌种/无参考价值）、不进线索清单。
- `outcome=manual_review` 时说明缺什么信息、需人工核实什么。
- 不要只输出“找到了 GSEXXXX”。必须说明样本是什么、每个文件下不下、为什么。
- 下载清单以**核验后成品**为准：核验前的 download 若核验不过会回退 lead / manual_review（见「下载执行」）。

## 下载执行

agent 对下载清单中 `outcome=download` 的记录执行下载。规则：

- **只下 `files[]` 中 `download=true` 的文件**，不要整包拉 RAW.tar；若 A 级文件在某个 tar 内，优先按成员取；**无法按成员取时，下载到临时区解包、只保留 A 级成员、丢弃其余**（或转 manual_review），不要把整包当下载成品。
- **下载后立即核验**（页面描述常与实文件不符，必须复验；这是文件形态的最终裁决）：
  1. 重跑「文件类型判断 §4」的打开核对——确认值确实是甲基化水平（β/M/比例/配对计数），不是 count 或统计表；
  2. 样本列数（或逐样本文件数）对得上 GSM 数；
  3. 能建立 GSM → 列（或 GSM → 文件）的映射；
  4. 核验不过 → **删除已下载文件（或移到 quarantine 目录）**，outcome 回退为 lead / manual_review + 记 QC 失败原因，不进下载清单成品，避免下游误用。
- **记溯源**：GSE、各文件 → `data_form`、来源 URL、下载数据/版本、GSM→列映射。
- **逐样本 call 文件**（`.cov`/`.bed`/每样本 β）：标 `needs_processing=merge_per_sample`，提示下游需合并（通常还需 coverage/缺失过滤或聚合到区域）。
- **不重复下载**：已存在且校验通过的文件跳过；大文件先核对大小/条目确认为目标 A 级文件再下。

## 给 agent 的短规则

检索 GEO 时先宽搜，再严格筛；关键词只用于召回，不替代语义判断。每个 GSE/GSM 必须回答：样本是什么、是不是人、是不是目标癌种、是不是 cfDNA/血液或可用于标志物发现的组织、病例和对照有多少、是否治疗前、文件是什么、哪些文件是 A 级、每个文件下不下（见「文件类型判断」）、是否有样本级注释（GSM 字段缺失可用文章补足）、为什么 download / lead / exclude / manual_review。明确无用（细胞系/动物/不可拆非目标癌种）记 exclude、不进线索；样本级排除清单见「默认不下载」。
