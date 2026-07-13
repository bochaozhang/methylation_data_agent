# GEO 数据检索注意事项 v4

适用对象：癌症早筛 AI Scientist 中负责 GEO 数据集可用性判定的 skill（geo_filter）。
本 skill 只做一件事：判断一个 GSE 数据集是否可用于癌症早筛液体活检方法开发。

## Scope（做什么 / 不做什么）

- Does: 判定单个 GEO 数据集 → keep / exclude / manual_review / article_only。
- Does NOT:
  - 检索/构建检索式 → 由 search 层（`config/cancer_synonyms.yaml` + `build_geo_search_string`）负责。
  - 下载 → 由 download 流程负责。
  - 跨数据集排序 → 由下游 ranking 负责（本 skill 只判定单个数据集，不排序）。
- 检索目标是找到“可用于癌症早筛液体活检方法开发的数据”，不是找到所有癌症甲基化相关数据。
- GEO 是主要数据来源但不是唯一；CFEA、EWAS Data Hub、TCGA/Xena、ArrayExpress/ENA、NGDC/GSA/OMIX、SRA、文章补充表等由其他流程另行检索，不在本 skill 范围。

## Procedure（按序执行 = reasoning chain）

判定每个 GSE 时按以下顺序回答（即 reasoning 字段要写出的链），gate 步骤可短路：

1. 数据类型：是不是 DNA 甲基化？（不是 → exclude）
2. 物种：是不是人？（不是 → exclude）
3. 样本类型：是不是细胞系/类器官/体外处理？（是 → exclude）
4. 癌种：是不是目标癌种？
5. 样本类型：是不是请求的样本类型（plasma/血清 = cfDNA）？是否有病例 + 对照？
6. 证据充分性：GSM 字段不够时，靠关联文章（abstract/Methods/Data availability）确认。

→ 给出 recommended_action，并记录为什么保留/排除（不能只输出 accession）。

贯穿上述步骤的核心判断原则：

- 标题命中不代表数据可用，必须看 Summary、Overall design、Sample characteristics 和关联文章。
- GEO 判断不能只看 GSE 页面；信息不够必须点开 GSM sample 页面，检查 Sample type、Source name、Characteristics、Extracted molecule、Molecule、Treatment protocol、Extract protocol、Data processing、Supplementary file 等字段。很多关键判断只在 GSM 页面出现。
- GEO 页面信息经常不完整，必须追踪 PMID/DOI，到原文 Methods、Data availability、Supplementary tables 中确认样本类型、病例对照、处理状态和文件说明。
- 文章或 GEO Summary 中提到 cfDNA/血液验证，不代表 GEO 公开了 cfDNA 数据；有些研究只公开组织发现数据，血液验证只在文章中以 qPCR、panel 或 marker 结果呈现。必须看 GSM 样本类型，确认 GEO 实际提供的是什么数据。
- 最终判断必须理解上下文语义：样本实际来自哪里、测了什么、公开了哪些文件、哪些样本可用。不能靠关键词命中直接决定纳入或排除。
- 每条候选数据都要记录为什么保留或为什么排除，不能只输出 accession 列表。

## Hard gates（任一命中 → exclude，无例外）

- 细胞系，例如 HCT-116、A549、NCI-H3122、CL1-5 等。
- 类器官、PDX、xenograft、动物模型、小鼠样本。
- 体外培养体系、药物处理、DNMTi 处理、辐射处理、基因编辑处理后的样本。
- 治疗后样本、复发样本、耐药模型、药物疗效预测队列；如果能单独提取治疗前 baseline 样本，可以只保留治疗前部分。
- 转移灶样本，除非任务明确需要；早筛数据整理中一般只保留原发癌、癌前病变和对照。
- 腹水、胸水、肺泡灌洗液等非血浆/血清样本，除非任务明确需要；不能和血浆 cfDNA 混用。
- 非目标癌种且无法拆分的数据。
- 只有极少样本、且没有特殊参考价值的数据。
- 只有 marker list、显著位点表，没有全量矩阵或样本级数据的数据。
- 数据锁定、无法下载、需要申请但当前无法获得的数据；可以记录为线索，但不进入可用数据清单。
- 物种非人（小鼠、动物模型）默认排除。

## Match criteria（怎样算“符合请求”）

候选数据必须能回答/满足：

- 样本类型：cfDNA、ctDNA、plasma、serum、whole blood、WBC、PBMC、tumor tissue、adjacent tissue、normal tissue、ascites、pleural effusion、BALF 等要分清。
- 疾病类型：目标癌种、癌前病变、良性疾病、健康人、其他癌种是否能区分。
- 样本量：病例数、健康对照数、良性/癌前病变数、癌旁/正常组织数。
- 分期和处理状态：是否早期、是否术前、是否治疗前；治疗后样本不能和治疗前混用。
- 是否为原发癌：转移灶、复发样本、耐药样本要单独标记，默认不用于早筛数据整理。
- 技术类型：450k、850k/EPIC、27k、RRBS、WGBS、MCTA、MeDIP、panel、qMSP 等。
- 文件类型：normalized beta matrix、IDAT、average beta、raw intensity、detection p-value、bed、bigWig、fastq、marker list 等。
- 是否有样本级注释：必须能把每个 GSM 对应到疾病状态、样本类型、处理状态；只有总样本数不够。
- 数据是否可直接使用：优先保留 beta matrix、处理后的 bed/bigWig；只有 marker list 不能作为全量数据。
- GEO 注释和文章是否一致：如果不一致，以原文 Methods、补充表、样本编号说明为准，并记录冲突。

## Ranking（⚠️ CONDITIONAL — 仅当 Match 通过后才用，用于排序/优先级，不用于 gate）

- 血浆/血清 cfDNA 或 ctDNA，且有癌症病例和健康人/良性病变/癌前病变对照。
- 早期、术前、治疗前样本。
- 癌组织 vs 癌旁/正常组织，适合寻找肿瘤来源甲基化标志物。
- 健康人 cfDNA、全血、WBC、PBMC、正常组织，可用于判断血细胞背景和组织特异性。
- 450k、850k/EPIC 数据，尤其是已有 normalized beta matrix 的数据。
- 样本量较大、样本注释清楚、有关联文章的数据。
- 同一个研究中能明确拆分目标癌种、正常对照、良性病变、癌前病变的数据。

## Edge cases → flag manual_review

- pooled cfDNA：可以作为线索或参考，但不能当成普通样本级数据。
- 只有病例没有健康对照的 cfDNA：可以记录，但不能单独做病例 vs 健康比较。
- 只有组织没有血液：可用于发现候选标志物，但不能证明血液中可检测。
- 只有全血/WBC：不是健康 cfDNA 对照，但可用于排除血细胞背景高的位点。
- 27k 数据：位点少、平台旧，可作为补充，优先级低于 450k/850k。
- RRBS/WGBS/MCTA/MeDIP：有价值，但与 450k/850k 平台不一致，样本量少时应单独标记，不要默认混合分析。
- 只有 fastq：不是完全不可用，但需要从头分析；如果样本量小、临床标签差，一般不优先。
- 泛癌或混合癌种数据：不要整套纳入，必须确认能否按样本编号拆出目标癌种。
- GEO 中 sample type 写得很笼统时，必须去文章或补充表查清楚。

## Evidence gathering（判定依据 + 证据不足时的处理）

**本 skill 判定所依据的证据**：GEO 元数据（title / summary / overall_design / platform / sample_count）+ 代表性 GSM 详情 + PubMed abstract（有 PMID 时）。**本 skill 自身不获取全文或补充表。**

基于手头证据，尽量确认以下关键点（这些是 keep/exclude 的依据；abstract 里没有的部分，就是证据不足之处）：

- 样本是否治疗前、是否早期、是否原发癌、是否血浆/血清 cfDNA。
- 文章中的 cfDNA/血液验证是否真的对应 GEO 公开数据；如果 GEO 只公开组织数据，必须标记为组织数据或文章标志物线索。
- 确认 GEO 中的样本数量是否和文章一致。
- 如果文章列出了具体样本编号，如哪些是腺瘤、哪些是转移灶、哪些是健康人，要记录这些编号。
- 如果 GEO 标签和文章标签冲突，要标记“需人工复核”，不能直接进入可用清单。
- 如果文章只有标志物结果，没有全量矩阵，也要记录为文章标志物线索，不要误认为有可用 GEO 数据。

**证据不足时**：若 abstract 不足以确认样本类型/对照/治疗状态等关键点 → `recommended_action=manual_review`，并在 reason/notes 写明缺什么。本 skill 到此为止，不再自行取证。

> **边界说明（给 orchestrator / 人工，不在本 skill 内）**：自适应取证（abstract 不够 → 取全文/补充表 → 重新判定）属于 orchestrator 的职责，不在本 skill。现实边界：全文仅 PMC Open Access 可得；补充表在期刊网站或 GEO supplementary；无 PMID 时可用 GSE 编号/标题在 PubMed（E-utilities）搜，仍找不到 → `manual_review`；Google Scholar、期刊网站无可靠 API，仅人工兜底。

## Output

结构化 JSON 的 schema 定义在代码（`skills/geo_filter/skill.py` 的 `_OUTPUT_CONTRACT`），本文件不重复。
语义约定：

- recommended_action: `keep`（可用，进审批）/ `exclude`（不进）/ `article_only`（文章提到但 GEO 没公开）/ `manual_review`（存疑/冲突/无法确认）。
- usable: `yes` / `partial` / `no` / `unclear`。
- plasma / 血清样本即 cell-free DNA（cfDNA）。
- 不要只输出“找到了 GSEXXXX”；必须说明样本是什么、为什么可用或不可用（写在 reason + reasoning）。

## Reference（非 procedure — 仅存疑时查阅；可裁剪 / RAG）

### 典型案例

- GSE122126：有健康 cfDNA 和部分癌种 cfDNA，可用于 cfDNA 背景或参考；但单个癌种样本数少，不能当作大规模癌种数据。
- GSE110185：结直肠癌 pooled cfDNA 数据，需要关注；但 pooled sample 不能当作普通独立样本。
- GSE79277：cfDNA RRBS 数据，包含 CRC、肺癌和健康人，可作为液体活检相关数据单独记录。
- GSE97932：血清 ctDNA 样本量看起来大，但如果没有健康对照和清楚文章信息，只能先标记为关注/人工复核。
- 治疗反应相关数据，例如抗 PD-1、铂类耐药、CC-486/durvalumab 等，只保留明确治疗前样本；治疗后或药物处理样本排除。
- 泛癌组织数据可以拆出目标癌种原发癌、癌旁、正常组织；转移灶、标注混乱样本要剔除或人工复核。
- EWAS Data Hub 中的样本标签可能和原文不一致；虽然本文主要针对 GEO，但如果引用 EWAS 结果，仍要回查原始 GEO 和文章。

### 文件类型优先级

450k/850k/EPIC 数据优先级：

1. 多样本 normalized beta matrix。
2. IDAT 原始文件，有完整样本注释。
3. raw intensity、detection p-value 等可重新处理的原始芯片数据。
4. 每个 GSM 页面单独的 average beta 或 normalized beta，可用但整理成本高。
5. 只有 marker list、差异位点表或文章图，不能作为全量数据。

测序类数据优先级：

1. 处理好的 methylation calls、region methylation table、bed、bigWig、coverage table。
2. BAM/CRAM，且有清楚 pipeline 和样本注释。
3. fastq only。

panel/qMSP 类数据要单独记录基因、探针、坐标、区域序列、样本类型、病例/对照数量和文章性能指标，不要标记为 genome-wide 数据。
