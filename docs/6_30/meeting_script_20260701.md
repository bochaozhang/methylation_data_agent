# 周会发言稿 — 2026-07-01（约 4 分钟）

配套 PPT：`docs/6_30/6_30_report_merged.pptx`（5 张：封面 → 本周进展 → Bug 修复详情 → Pipeline → 智谱咨询对照与下一步）

---

**封面**

大家好，我汇报一下MethyAgent本周的进展。

---

**本周进展**

我主要要做了三件事：debug 修复了 3 个 NCBI 抽取错误中的 1 个；补全结构化输出字段并加了可观测性日志；搭了一个独立 的prototype， 跑通了搜索 → GEO 评估 → registry 的pipeline

---

**Bug 修复详情**

之前人工核对 PubMed 原文时发现了三个bug： 
AUC 误判、组织和 cfDNA 混用、参考数据集误标
一开始按 prompt 规则改，可是复测还是错。我发现模型在照抄示例里的占位符数字，不是真的在读摘要。所以 Bug 1 改成代码级强制校验：摘要里没"AUC"就直接置空，已复测修复。Bug 2、3 我还没修好，但是准备先判断 AUC 和 sample_type 是否在同一句、accession 附近是否有"参考/背景"等词，直接代码处理，不依赖模型自觉。

notes:
- LLM reviewer as next step as well, verification/review agent
    prompt角色设定
- manual reviewing

---

**Pipeline**

prototype就是把这两周写的tools都串起来：拿乳腺癌的真实论文跑了一遍，找到两个候选数据集，一个是小鼠的研究，模型正确排除了；另一个是真的相关但元数据不全，标了需要人工复核。两条都成功写进了 registry。说明这条链路本身是可以跑通的。

---

**智谱咨询对照与下一步**

上周智谱给的几条系统性诊断，我对照了一下：

评估体系已建 gold_standard.py，2/10 已核验，下周补齐。

可观测性这块，给搜索 pipeline 加了 Stage 1/Stage 2 通过率和字段完整度日志——字段完整度就是每个结构化字段（比如 AUC、样本量、data_availability 这些）真实被填上、而不是留空的比例。我发现 data_availability 这个字段——就是标注这篇论文的数据能不能公开下载、要不要申请、还是完全没有——在两轮查询里完整度都是 0%，一条都没填上，所以得修复。

LLM 靠 regex 兜底这条，Bug 1 的代码级校验就是这个思路的雏形，下周推广到更多字段。

修复后就可以尝试把search_and_extract 和 evaluate_geo_dataset 接入 orchestrator了

现在还是两条独立路径没打通。在您决定之前我没有直接改 orchestrator 或 LiteratureAgent 的代码，先用独立 prototype 把整条链路验证了一遍，确认逻辑没问题。如果思路可以，下周就照这个方向推进，具体怎么接可以再聊。汇报完毕，欢迎讨论。


从文献里获取其他信息
标志物/biomarker
数据 -> 本质还是找标志物
change prompt: cancer biomarker for methylated cfDNA
    reason first, then decide whether 纳入或者不纳入
    后面的话很大程度根据之前已经说的话

Look into claude science: anything to 借鉴
    end-to-end performance
    literature search ability


