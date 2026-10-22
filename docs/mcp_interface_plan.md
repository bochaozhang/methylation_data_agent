# MethyAgent MCP 接口计划

日期：2026-10-16
状态：**Phase 1 + Phase 2 已落地（2026-10-16，MCP 分支）**；Phase 3-4 待实施
关联：docs/caching_audit.md（缓存与去重）、registry GSM 粒度表、api/main.py（现有 REST）

---

## 1. 目标

让外部 agent（Claude Code、其他 LLM agent）以标准化方式使用本 data agent：
提交检索任务、轮询状态、取回结构化结果（判定 + 元数据 + 文件路径），
并支持**复合查询**（主检索/辅检索分开，训练数据与验证数据不混）。

## 2. 决策记录（为什么这样做）

| 决策 | 理由 |
|---|---|
| **MCP，不把 pipeline 做成 skill** | 本系统是有状态长时服务（任务队列 + 双 worker + SQLite 单写者 + 异步下载 daemon，单任务 ~55 min、~136 次 LLM 判定）。skill 意味着消费方在自己进程里跑代码，要求代码库 + 密钥 + 代理 + daemon，且 registry 写冲突、token 花费失控 |
| **MCP 是薄适配层，一切经 REST API** | api/main.py 已有完整接口（/query、/tasks、/datasets、/review 等）。MCP server = stdio→HTTP 适配 + 少量新端点，不直接写 DB |
| **submit + poll，不做阻塞式全流程 tool** | 全流程 ~55 min；MCP tool 必须立即返回 task_id，消费方轮询 |
| **task 边界 = sub-task 边界**（复合查询拆成多个 task） | 注册表已按 task_id 记录归属（query_dataset_map / query_sample_map / sample_metadata.csv 的逐 task 判定列）。实测合并 query 会丢辅检索（合并检索串召回不了 GEO tissue；tcga_direct.py:40 因 sample_type=cfdna 跳过 TCGA） |
| **结构化查询走 intent_json 直通，不经 LLM parser** | 消除 parser 随机性/信息丢失；intent_json 存入 task_queue，parse_node 命中时跳过 parse_query_with_llm |
| **训练/验证等角色语义留在主 agent** | data agent 只按 sub-task（label）分组返回，不理解"训练/验证" |

**非目标**：parser 内拆 `List[SearchIntent]`（改动面大且 task 队列已是现成隔离单位）；
MCP 传输文件内容（GB 级，本地 agent 直接读共享目录）；远程文件分发（Phase 3 后再议）。

## 3. 架构

```
外部 agent (Claude Code / 其他)
   │  MCP (stdio 本地 / streamable HTTP 远程, Phase 3)
   ▼
MCP server  (mcp_server/, 宿主机 venv 运行)
   │  httpx → localhost:80
   ▼
FastAPI webui (api/main.py)  ──现有──▶ task_queue (SQLite) ──▶ agent1-query worker
   │                                        ▲                        (parse→search→filter→register)
   ├─ GET /tasks /datasets /query-datasets  │
   ├─ 新增 POST /search/dry-run（parse+search，不落库）          agent1-download worker
   └─ MCP 层做路径翻译: /app/data/** → <宿主机>/data/**          （异步下载）
```

**路径翻译（必须）**：registry 存的是容器路径 `/app/data/...`（容器内 worker 写入），
宿主机真实位置经 `./data:/app/data` 挂载对应。MCP 返回前统一翻译，
宿主机根路径取 env `MCP_DATA_HOST_PATH`（默认 `<repo>/data` 绝对路径）。

## 4. 工具面（MCP tools）

| Tool | 背后 | Phase |
|---|---|---|
| `search_datasets(query或structured)` | 新端点 POST /search/dry-run：parse + geo_search，返回候选 GSE（秒级、便宜、无副作用） | 1 |
| `get_task_status(task_id)` | GET /tasks/{id} | 1 |
| `list_datasets(status?, cancer_type?...)` | GET /datasets | 1 |
| `get_dataset(accession)` | GET /datasets/{accession}（含 local_path / sample_metadata_path，翻译后） | 1 |
| `submit_query(query 或 structured, label?)` | POST /query（扩展） | 2 |
| `submit_compound_query({main, auxiliary})` | 内部两次 POST /query，label=main/auxiliary，返回两个 task_id | 2 |
| `get_task_result(task_id)` | GET /tasks/{id} + /query-datasets/{id}（+ /samples）：verdict 清单 + 逐数据集路径（翻译后） | 2 |
| `cancel_task(task_id)` | POST /tasks/{id}/cancel | 2 |
| `approve_dataset / reject_dataset` | POST /review/...，默认关，`MCP_ALLOW_WRITES=1` 才注册 | 3 |
| `health` | GET /health | 1 |

结构化查询字段（与 SearchIntent 对齐）：`cancer_type{display, tcga_code}`、
`sample_types[]`、`platform`、`data_type`、`year_start/end`、`extra_keywords[]`、`notes`。
MCP 层校验后序列化为 intent_json 随任务提交；同时生成一句人读 query 存 task_queue.query（可审计）。

## 5. 分期实施

### Phase 1 — 只读 + dry-run（零风险，先跑通）✅ 2026-10-16 落地

1. **API 侧**
   - [x] `POST /search/dry-run`：入参 query 或 structured intent；进程内跑 parse + SearchSkill（只 search，不 filter/register/download）；返回 intent + search string + 候选列表（截断 max_candidates≤200 + 总数）。同步 handler（FastAPI threadpool，慢查询不阻塞事件循环）
   - [x] docker-compose webui 服务补 `NCBI_PROXY: "socks5h://172.17.0.1:1080"`（dry-run 要出网查 NCBI；ncbi_search.py:71 读此 env）
   - [x] dry-run 超时保护：实测 structured 48s / free-text 52s（LLM parse + esearch），同步可接受（MCP_API_TIMEOUT 默认 300s 兜底）；超预期再改 BackgroundTask + 轮询
2. **MCP server**
   - [x] 新建 `mcp_server/`（server.py / api_client.py / paths.py / \_\_main\_\_.py），stdio transport
   - [x] `.venv` 安装 `mcp` 2.3.0（requirements.txt 固定 `mcp>=2.3,<3`）。**注意：mcp 2.x 把 FastMCP 改名为 `MCPServer`**（`from mcp.server.mcpserver import MCPServer`）
   - [x] 实现 search_datasets / get_task_status / list_datasets / get_dataset / health；list 视图裁剪大字段（llm_evidence 等），get_dataset 返回全量；全部经路径翻译
3. **注册使用**：`claude mcp add methyagent -- <venv>/bin/python <repo>/mcp_server/__main__.py`（绝对路径脚本形式，任意 cwd 可启动）— 已注册，✓ Connected
4. **测试**：tests/test_mcp_paths.py（11→实际 6 用例：前缀替换/env 覆盖/未知前缀透传/record 翻译）+ tests/test_api_dry_run.py（4 用例：422 / structured 直通无 LLM / 截断标记 / 检索串回显）；全量 suite 3F+20E 与基线一致，零新增

验收（已达成）：dry-run 结构化 136 候选（与生产召回一致）；free-text parse_method=llm, COAD+cfdna；`get_dataset("GSE124600")` 返回宿主机路径且文件真实存在。

### Phase 2 — 提交 + 复合查询 + 结构化查询（核心价值）✅ 2026-10-16 落地（MCP 分支）

1. **registry（迁移已跑在生产库上）**
   - [x] `task_queue` 新增 `label` / `intent_json`（CREATE + ALTER 迁移，registry.py）
   - [x] create_task / claim_task 带新列；新增 `list_active_tasks()`（去重查询用）
2. **API**
   - [x] QueryRequest 扩展：`intent`（结构化）、`label`、`force`；query 变为 intent 存在时可选（自动生成 "structured: ..." 摘要句供审计/literature 分支）
   - [x] POST /query：intent 先经 normalize_intent 再落库；**提交侧去重**——等价 pending/running 任务（规范化 intent 的 canonical JSON 相等，或自由文本 query 全等）返回既有 task_id（deduplicated=true, HTTP 200），`force` 可绕过
   - [x] TaskResponse 增加 label / deduplicated
3. **pipeline**
   - [x] parse 逻辑提取为模块级 `_parse_query_intent(raw, json_llm, intent_json)`（优先级 intent_json → LLM → rules；structured 路径不触碰 LLM）；parse_node 调用之
   - [x] Agent1State 加 intent_json 字段；run_agent1_pipeline / run_database_agent / daemon claim 链路传参；legacy 分支忽略 intent_json
4. **MCP tools**（共 9 个）
   - [x] submit_query / submit_compound_query（main+auxiliary 默认 label，各接受 query 或 intent）/ get_task_result（verdict 分桶 download/manual_review 全量、exclude 默认只计数，local_path 翻译）/ cancel_task
5. **护栏**
   - [x] 去重（见 API）；compound 重放等价 intent → 两个 task_id 均命中去重
   - [~] max_candidates 评估：filter_node 对全部候选逐个判定（无截断点）；截断需要 pre-filter 排序依据，**暂缓**（注：verdict cache 已确定不做——SPEC 更新必须即时反映到判定，见 §6 #4）
6. **测试**（新增 15 个：test_registry_task_columns / test_api_query_submit / test_pipeline_structured_parse；全量 206 = 191+15，3F+20E 与基线一致）
   - 结构化 intent 幂等性有专门测试（API 存 normalized，parse_node 再 normalize 一次，结果不变）

**验证事故记录（已恢复）**：test_api_query_submit 首版把 env 隔离放在 setUpModule（unittest 在模块体执行后才调用它）→ 测试打到生产 registry，7 条测试任务被 live daemon claim。处置：停 worker、删 7 行（register 未到达，datasets/map 零污染；仅留下无害 gsm cache 文件）、测试隔离改为模块 import 时生效。教训已写入测试文件 docstring。

验收（已达成）：compound 提交返回 main/aux 两个 task_id + label；worker 日志 `Claimed ... label='main' structured=True` + `parse: method=structured cancer=COAD`；filter 用的 intent 正确（cfdna/plasma/serum）；等价重放 deduplicated=true 返回原 id；cancel_task 取消 pending 任务生效。

### Phase 3 — 写操作 + 远程消费方

- [ ] approve/reject tool（`MCP_ALLOW_WRITES` 开关，默认只读）
- [ ] streamable HTTP transport：mcp_server 入容器或独立 service，bearer token 鉴权（compose 增服务）
- [ ] 提交侧成本护栏：并发任务上限 / 每 token 配额（复用去重机制）
- [ ] 远程文件分发需求此时再评估（只读共享盘 / 对象存储 / 按需打包端点）

### Phase 4 — 配套 SKILL.md（可选）

- [ ] 一份使用技能文档（"先 dry-run → submit → 轮询 → approve → 拿路径"工作流 + 字段说明），
      供 Claude Code 用户 `claude mcp add` 后即插即用；skill 只教流程，能力全在 MCP

## 6. 风险与开放问题

| # | 事项 | 处置 |
|---|---|---|
| 1 | dry-run 在 webui 容器内执行时长（NCBI esearch 经代理） | ~~实测 ~50s~~ → **2026-10-22 已优化到 ~10-15s**（低于 CC 默认 MCP tool 超时 30s）：`_get` 改全局线程安全限速器（顺带修复 filter 3 线程并发下限速被放大的潜在违规）+ batch 元数据批次并行 + supp 目录列表批内 8 路并行；CC 消费方超时兜底 `MCP_TOOL_TIMEOUT` 已写入使用说明 |
| 2 | 136 verdict 的 result_json 体量大，MCP 响应超限 | MCP 层默认摘要（每 verdict 截 reason），`full=true` 才全文 |
| 3 | 同一 GSE 被主/辅两个 task 同时判 download | 已天然解决：datasets 按 accession 去重 + dest-exists 跳过（caa6939），样本级归属看 query_sample_map / sample_metadata.csv 逐 task 列 |
| 4 | ~~verdict cache（caching_audit backlog #4）~~ **确定不做（设计决定，2026-10-20）**：每次重新 LLM 判定是故意设计——判定标准在 skill 的 SPEC.md 里，SPEC 更新后 verdict 必须跟着变，缓存违背"SPEC 是唯一事实源" | 无需行动；重复查询的 token 成本为固有代价 |
| 5 | mcp SDK 版本更迭快 | 锁 requirements.txt 版本；Phase 1 只依赖最小特性（tool + stdio） |
| 6 | **检索失败语义**（2026-10-19 事故）：GEO search 网络失败（如代理死）被静默吞掉 → pipeline"成功"完成 0 候选空任务（aux 首次重跑即中招）。应改为 search 失败 → 任务 fail（区分"真 0 结果"与"检索出错"） | **非 MCP 待办**（涉及 pipeline/SearchSkill 语义，用户拍板暂缓）；当前专注 MCP。缓解措施已做：隧道改 autossh + systemd（methy-proxy.service，跳板 34.26.3.123，断线自动重连 + 开机自启），降低复发概率 |
| 7 | **filter LLM 截断无重试**（2026-10-22，审批队列清理时发现）：3/17 待审项的 reason 是 `filter_error: Could not parse response content as the length limit was reached`——LLM 输出被 completion 上限截断，JSON 解析失败，判定直接降级 manual_review（不是真实判定）。 | **待办（skill 层）**：geo_filter 捕获 length-limit 截断后重试一次（如压缩 reasoning 要求/提高 max_tokens 再问）；重试仍失败才降级 manual_review 并在 reason 里区分"截断失败"与"证据不足" |
| 8 | **提交者自编码标题 → cancer 列 unclear**（2026-10-22，GSE97923 实例）：`_gsm_cancer_map` 是确定性关键词映射，只读 GSM 结构化字段（characteristics/source_name）；当提交者把病种写进自己发明的 sample-title 编码（如 `COAD.ct-511`）而 characteristics 只填 "cell-free DNA" 时，773 行全部落 `unclear`——样本粒度病例/对照标签缺失（Tier-2 工作时已识别的残留缺口，GSE97923 首次大规模撞上）。另发现 gsm_metadata_cache 里 title 字段为空（efetch 路径未带回），解码的原始材料也缺。 | **待办**：(a) resolve 时把 sample title 存进 gsm cache（原始材料）；(b) 每高价值 GSE 调一次 LLM 解码标题模式 → 映射规则套用全部 GSM（对齐 Tier-2 做法）；(c) 高价值数据集（如 GSE97923, n=773）可先人工映射。注意：不阻塞下载与逐任务判定，只影响样本粒度分析选样 |

## 7. 建议排期

Phase 1 先行独立交付（~半天）；Phase 2 是核心（registry 迁移 + parse 分支 + compound，~1–2 天含测试）；
Phase 3/4 视真实消费方出现再排。
