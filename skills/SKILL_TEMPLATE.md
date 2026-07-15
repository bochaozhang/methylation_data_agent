# Skill 模板（LangGraph 适用）

一个 skill = 一个 graph node = 一件事。本模板用于 `skills/` 下所有 skill 的 SPEC
（如 `skills/geo_filter/SPEC.md`）。复制本文件、填入领域内容即可。

## 核心原则

- **单一职责**：一个 skill 只做一个判定/动作。搜索、下载、跨样本排序等是别的 node。
- **Procedure 与 Reference 分离**：Procedure（每次必发）要紧凑；Reference（案例、查表）
  只在存疑时查阅，可裁剪或 RAG，不必每次全发 → 省 token。
- **条件段要标注**：标注 `CONDITIONAL` 的节（如 Ranking）只在主判定通过后才用，
  未来可拆成独立下游 node。
- **输出 schema 单一真源在代码**：SPEC 不重复 JSON schema，只讲语义约定。
- **首行 `# ` 标题 = 文档名 + 版本**：被 `SPEC_NAME` 自动解析，写进 per-query CSV 日志。
  改版本只改这行。

## 模板正文

```markdown
# <文档名> v<版本>

适用对象：<这个 skill 服务谁、做什么>。
本 skill 只做一件事：<一句话职责>。

## Scope（做什么 / 不做什么）
- Does: <本 node 的唯一职责>
- Does NOT:
  - <明确划出去的> → 指向哪个 skill/node（search? rank? download?）
  - ...
- I/O: inputs (<…>) → outputs (<…>)

## Procedure（ALWAYS 按序执行 = reasoning chain）
按以下顺序回答（即 reasoning 字段要写出的链），gate 步骤可短路：
1. <步骤>
2. …
N. → verdict（并记录为什么保留/排除）

贯穿步骤的核心判断原则：
- <领域原则 bullets>

## Hard gates（任一命中 → exclude，无例外）
- <…>

## Match criteria（怎样算"符合请求"）
- <…>

## Ranking（⚠️ CONDITIONAL — 仅当 Match 通过后才用，用于排序，不用于 gate）
- <…>

## Edge cases → flag manual_review
- <…>

## Evidence gathering（元数据稀疏时如何确认）
- <…>

## Output
结构化 JSON 的 schema 定义在代码（<skill.py 路径> 的 _OUTPUT_CONTRACT），本文件不重复。
语义约定：
- <recommended_action 各值含义>
- <usable / 其他字段的口径>
- <必须填 reason + reasoning，不能只输出 ID>

## Reference（非 procedure — 仅存疑时查阅；可裁剪 / RAG）
### 典型案例
- <case_id>：<…>
### <查表/层级>
- <…>
```

## 各节 → LangGraph 映射

| 模板节 | 在图里对应 |
|---|---|
| Procedure | node 的 system-prompt 主体（= reasoning 链）|
| Hard gates / Match / Edge cases | prompt 内联规则；gate 可在链里短路 |
| Ranking（CONDITIONAL） | 可拆成下游独立 node（对通过的数据集排序）|
| Reference | 不必每次发：按需注入或 RAG（存疑时才给）|
| Output | 代码里的 structured-output schema（单一真源）|
| Scope / Does NOT | 让 orchestrator 知道何时不该调本 skill（→ 条件路由）|

## 写 skill 时的检查清单

- [ ] 标题行是 `# <名> v<版本>`（被 SPEC_NAME 解析）
- [ ] Scope 明确写了 Does NOT（划清与相邻 skill 的边界）
- [ ] Procedure 是有序步骤，能短路，且 = reasoning 链
- [ ] 没有"条件内容"混进 always-on Procedure（ranking/strategy 单独成节）
- [ ] 没有 search 层内容混进来（检索词/同义词属于 search skill）
- [ ] 没有重复 output JSON（schema 只在代码）
- [ ] 案例/查表放进 Reference，不放 Procedure
