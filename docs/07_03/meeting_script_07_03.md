# 周会发言稿 — 2026-07-03（约 4 分钟）

配套 PPT：`docs/07_03/weekly_report_07_03.pptx`（6 张：封面 → 本周进展 → LLM Reviewer 详情 → 新 Orchestrator 架构图 → Claude Scientist 对照 → 下一步）

---

**封面**

大家好，我汇报一下 MethyAgent 本周的进展。

---

**本周进展**

这周主要做了三件事：按您的建议把 Bug 2、3 的修复思路从正则改成了 LLM Reviewer；搭了一个新的、更 agentic 的 orchestrator；给两条 orchestrator 加了部署时可切换的开关。

---

**LLM Reviewer 详情**

上周汇报的时候，Bug 2（组织/cfDNA AUC 混淆）和 Bug 3（参考数据集误标）我准备按"拆句 + 关键词"这种代码级规则去修。周一您反馈说不要用纯正则，让我加一个 LLM reviewer——draft 抽取出来之后，再让模型看着摘要原文复核一遍。

我先按原计划把正则版本写完了，跑起来发现确实有问题：它要求 AUC 数值和 sample_type 关键词必须落在同一句话里，摘要里稍微隔一句表达就会被误删，参考数据集判断也只能靠"reference/background"这几个固定词，本质上还是没有真正理解上下文。

所以按您说的，加了第二次 LLM 调用做复核。用 PMID 40860669 实测：draft 里 sample_type 是 plasma_cfdna，但 auc_validation 报的是 0.922——那其实是摘要里 TCGA 组织队列的 AUC，真正的 cfDNA AUC 是 0.728，reviewer 把 0.922 正确置空了，并标记人工复核；dataset_ids 里的 GSE50132，摘要原文写的是"仅用于过滤背景甲基化噪音的白细胞参考集"，reviewer 也正确把它剔除了，保留了真正做验证的 TCGA 和 GSE69914。

这个思路也算是回应了两周前智谱说的"没有 multi-turn reasoning、没有 reflection loop"——现在等于加了一轮反思。代价是每篇论文多一次 LLM 调用，报告里也注明了这个成本权衡。

本地环境没有配 API key，所以这次测试用的是一个返回固定"标准答案"的假模型，验证的是代码的合并/解析逻辑没问题；真实场景下模型的推理质量，还需要您在服务器上用真实 key 跑一遍确认。

---

**新 Orchestrator 架构图**

按您的要求新建了 agents/orchestrator_v2.py，没有改动现有的 orchestrator.py。

现有的 v1 是固定顺序：parse_query → run_database_agent → run_literature_agent → generate_report，节点顺序写死在代码里。v2 是把三个能力包成 tool——search_papers、evaluate_geo_dataset_tool、write_to_registry——交给顶层的 LLM 自己决定调用哪个、调用几次、什么顺序，用的是 LangGraph 自带的 create_react_agent。这个链路其实和上周的 pipeline_prototype 很像，区别是那边的调用顺序是脚本写死的，这边是模型自己决定的。

同样受限于没有 API key，这次也是用脚本模拟了一个会按预期顺序调用三个 tool 的假模型，跑通了"搜索论文 → 发现 GSE50132 → 评估为背景参考集应排除 → 写入 registry"整条链路，registry 写入是真实的（临时数据库），只有最上层的 LLM 决策是模拟的。运行日志存在 docs/07_03/orchestrator_v2_mock_run_log.json。

另外加了个部署开关：ORCHESTRATOR_VERSION 环境变量，或者 settings.yaml 里的 orchestrator.version 字段，v1/v2 可以在部署时切换，默认还是 v1，不影响现在生产在用的链路。

---

**Claude Scientist 对照与下一步**

Claude Scientist 那边是并行测试的，这周的结果还没整理完，先占个位，下周补上对比数据。

下一步：这周做的两块——LLM Reviewer 和新 Orchestrator——核心逻辑都写完测完了，但都是用模拟的 LLM 响应验证的，需要您在服务器上用真实 API key 和网络跑一遍，确认真实场景下的推理质量；biomarker 召回率测试因为同样的原因这周没跑，也放到了下周；gold_standard 剩余的人工核验和 data_availability 完整度排查照常结转。如果 v2 orchestrator 这条路您觉得可行，下周可以讨论要不要往生产切、或者往您之前提到的完整技能化方向继续推进。

汇报完毕，欢迎讨论。
