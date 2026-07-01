# GEO 数据检索注意事项 v3

适用对象：癌症早筛 AI Scientist 中负责 GEO 公共数据检索的 agent。

本文档只针对 GEO 检索。GEO 是目前主要数据来源，但不是唯一来源；CFEA、EWAS Data Hub、TCGA/Xena、ArrayExpress/ENA、NGDC/GSA/OMIX、SRA、文章补充表等也可能有重要数据，需要另行检索或补充。

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
- 每条候选数据都要记录为什么保留或为什么排除，不能只输出 accession 列表。

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
- 文件类型：normalized beta matrix、IDAT、average beta、raw intensity、detection p-value、bed、bigWig、fastq、marker list 等。
- 是否有样本级注释：必须能把每个 GSM 对应到疾病状态、样本类型、处理状态；只有总样本数不够。
- 数据是否可直接使用：优先保留 beta matrix、处理后的 bed/bigWig；只有 marker list 不能作为全量数据。
- GEO 注释和文章是否一致：如果不一致，以原文 Methods、补充表、样本编号说明为准，并记录冲突。

## 优先保留的数据

- 血浆/血清 cfDNA 或 ctDNA，且有癌症病例和健康人/良性病变/癌前病变对照。
- 早期、术前、治疗前样本。
- 癌组织 vs 癌旁/正常组织，适合寻找肿瘤来源甲基化标志物。
- 健康人 cfDNA、全血、WBC、PBMC、正常组织，可用于判断血细胞背景和组织特异性。
- 450k、850k/EPIC 数据，尤其是已有 normalized beta matrix 的数据。
- 样本量较大、样本注释清楚、有关联文章的数据。
- 同一个研究中能明确拆分目标癌种、正常对照、良性病变、癌前病变的数据。

## 默认排除的数据

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

## 需要谨慎标记的数据

- pooled cfDNA：可以作为线索或参考，但不能当成普通样本级数据。
- 只有病例没有健康对照的 cfDNA：可以记录，但不能单独做病例 vs 健康比较。
- 只有组织没有血液：可用于发现候选标志物，但不能证明血液中可检测。
- 只有全血/WBC：不是健康 cfDNA 对照，但可用于排除血细胞背景高的位点。
- 27k 数据：位点少、平台旧，可作为补充，优先级低于 450k/850k。
- RRBS/WGBS/MCTA/MeDIP：有价值，但与 450k/850k 平台不一致，样本量少时应单独标记，不要默认混合分析。
- 只有 fastq：不是完全不可用，但需要从头分析；如果样本量小、临床标签差，一般不优先。
- 泛癌或混合癌种数据：不要整套纳入，必须确认能否按样本编号拆出目标癌种。
- GEO 中 sample type 写得很笼统时，必须去文章或补充表查清楚。

## 文件类型判断

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

## 文章反向追踪

GEO 信息经常不够，必须找文章。至少做以下检查：

- GEO 页面有 PMID/DOI 时，打开原文和补充材料。
- 没有 PMID/DOI 时，用 GSE 编号、标题、作者在 PubMed、Google Scholar 或期刊网站搜索。
- 在文章中查 Methods、Cohort description、Data availability、Supplementary table、Sample information。
- 确认样本是否治疗前、是否早期、是否原发癌、是否血浆/血清 cfDNA。
- 确认文章中的 cfDNA/血液验证是否真的对应 GEO 公开数据；如果 GEO 只公开组织数据，必须标记为组织数据或文章标志物线索。
- 确认 GEO 中的样本数量是否和文章一致。
- 如果文章列出了具体样本编号，如哪些是腺瘤、哪些是转移灶、哪些是健康人，要记录这些编号。
- 如果 GEO 标签和文章标签冲突，要标记“需人工复核”，不能直接进入可用清单。
- 如果文章只有标志物结果，没有全量矩阵，也要记录为文章标志物线索，不要误认为有可用 GEO 数据。

## 输出格式要求

每条候选数据至少输出以下字段：

```json
{
  "source": "GEO",
  "accession": "GSE/GSM/GPL",
  "title": "",
  "pmid": "",
  "doi": "",
  "cancer_type": "",
  "sample_type": "",
  "disease_groups": "",
  "sample_size": "",
  "stage_or_treatment_status": "",
  "technology": "",
  "platform": "",
  "available_file_type": "",
  "sample_level_annotation": "yes/no/unclear",
  "usable": "yes/no/partial/unclear",
  "recommended_action": "keep / exclude / article_only / manual_review",
  "reason": "",
  "notes": ""
}
```

不要只输出“找到了 GSEXXXX”。必须说明样本是什么、为什么可用或不可用。

## 已有经验中的典型判断

- GSE122126：有健康 cfDNA 和部分癌种 cfDNA，可用于 cfDNA 背景或参考；但单个癌种样本数少，不能当作大规模癌种数据。
- GSE110185：结直肠癌 pooled cfDNA 数据，需要关注；但 pooled sample 不能当作普通独立样本。
- GSE79277：cfDNA RRBS 数据，包含 CRC、肺癌和健康人，可作为液体活检相关数据单独记录。
- GSE97932：血清 ctDNA 样本量看起来大，但如果没有健康对照和清楚文章信息，只能先标记为关注/人工复核。
- 治疗反应相关数据，例如抗 PD-1、铂类耐药、CC-486/durvalumab 等，只保留明确治疗前样本；治疗后或药物处理样本排除。
- 泛癌组织数据可以拆出目标癌种原发癌、癌旁、正常组织；转移灶、标注混乱样本要剔除或人工复核。
- EWAS Data Hub 中的样本标签可能和原文不一致；虽然本文主要针对 GEO，但如果引用 EWAS 结果，仍要回查原始 GEO 和文章。

## 给 agent 的短规则

检索 GEO 时先宽搜，再严格筛。关键词只用于召回候选数据，不能替代语义判断。每个 GSE/GSM 必须回答：样本是什么、是不是人、是不是目标癌种、是不是 cfDNA/血液或可用于标志物发现的组织、病例和对照有多少、是否治疗前、文件是什么、是否有样本级注释、为什么保留或排除。凡是细胞系、类器官、动物、体外处理、治疗后、转移灶、无样本级注释、只有 marker list 的数据，默认不能进入可用数据清单。
