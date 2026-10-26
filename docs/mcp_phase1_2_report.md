# MCP 接口 Phase 1+2 实施报告

日期：2026-10-19（10-20 上游反馈迭代补记）
分支：`MCP`（已推送 origin/MCP；main 未动）
提交链：`2bbff73`（Phase 1）→ `ec97629`（Phase 2）→ `2f63c21`（daemon 自愈修复）
　→ `5970bc5`（series_matrix OOM 根因修复）→ `53feaff`（文档入库）
　→ `feb447c`（字段暴露·阻断项）→ `c92ff35`（字段暴露·全量清查）→ `9ea2848`（协作式取消）
　→ `21c778a`（TCGA 判定语义）→ `bd76ce2`（审批 tool）→ `dcccd73`（dry-run 提速）
设计文档：docs/mcp_interface_plan.md ｜ 使用说明：docs/mcp_usage_guide.md

---

## 1. 一句话总结

data agent 现在是一个标准 MCP 服务：外部 agent 可以 dry-run 预览召回（秒级、免费）、
提交结构化或自由文本检索任务、用 label 区分复合查询的子任务（训练/验证数据不混）、
轮询状态并取回判定分组与宿主机文件路径。9 个 tool，REST API 全程薄适配，pipeline 语义
零改动（唯一 pipeline 改动是 parse 的结构化直通分支）。

## 2. 架构与数据流

```
外部 agent ──MCP(stdio)──▶ mcp_server/ ──httpx──▶ FastAPI(webui:80) ──▶ task_queue(SQLite)
                                                   │                        │
                                                   │            agent1-query worker（claim→parse→search→filter→register）
                                                   │                        │
                                                   └──GET /datasets 等      agent1-download worker（异步下载）
```

- MCP server 不直接写库、不跑 pipeline；一切经 REST（唯一新端点 `POST /search/dry-run`）
- **路径翻译**：注册表存容器路径 `/app/data/**`（容器内 worker 写入），MCP 返回前翻译为
  宿主机路径（`./data:/app/data` 挂载对应），消费方拿到即可读
- **结构化直通链**：`POST /query {intent}` → normalize_intent → `task_queue.intent_json`
  → daemon claim → `parse_node` 直接采用（`parse_method=structured`，**不调 LLM**）→
  确定性检索串

## 3. 落地明细

### Phase 1（只读 + dry-run，2bbff73）

| 层 | 改动 |
|---|---|
| api | `POST /search/dry-run`：query（LLM parse）或 intent（确定性）→ parse + geo-search，无判定无写入；同步 handler 跑 FastAPI threadpool（初测 ~50s；2026-10-22 并行化后 ~10-15s，低于 CC 默认 MCP tool 超时）|
| compose | webui 服务补 `NCBI_PROXY`（dry-run 出网） |
| mcp_server/ | 新包：server.py / api_client.py / paths.py / \_\_main\_\_.py；stdio；`mcp>=2.3,<3`（注意 mcp 2.x 将 FastMCP 改名 `MCPServer`） |
| tools | search_datasets / get_task_status / list_datasets（裁剪 llm_evidence 等大字段）/ get_dataset（全量）/ health |
| 注册 | `claude mcp add methyagent -- <venv>/python <repo>/mcp_server/__main__.py`（绝对路径，任意 cwd）→ ✓ Connected |

### Phase 2（提交 + 复合查询 + 去重，ec97629）

| 层 | 改动 |
|---|---|
| registry | `task_queue` + `label` + `intent_json`（CREATE + ALTER 幂等迁移，已跑在生产库）；create/claim 带新列；`list_active_tasks()` |
| api | `/query` 接受 `intent`（normalize 后落库；query 可选，自动生成 "structured: ..." 摘要句）、`label`、`force`；**提交侧去重**：等价 active 任务（canonical intent 相等 / 自由文本全等）返回原 task_id（`deduplicated=true`，HTTP 200）；TaskResponse + label/deduplicated |
| pipeline | parse 链提取为 `_parse_query_intent`（intent_json → LLM → rules fallback）；`Agent1State.intent_json`；daemon → run_database_agent → run_agent1_pipeline 传参；legacy 分支忽略 intent_json |
| mcp_server | +submit_query / submit_compound_query（main+aux 默认 label）/ get_task_result（verdict 分桶 + exclude 默认只计数 + 路径翻译）/ cancel_task |

### 自愈修复（2f63c21，事故驱动，详见 §5）

- `Registry.requeue_orphaned_tasks()`：claim 模式 worker **启动时**把遗留 `running` 任务拉回
  `pending`（download 模式绝不调用——会误杀在跑任务）
- `heartbeat_*` 行从 requeue 与 claim_task 双向排除（见 §5 事故 3）

## 4. 测试结果

新增 5 个测试模块共 **30 个用例**，全部通过：

| 模块 | 数量 | 覆盖 |
|---|---|---|
| tests/test_mcp_paths.py | 7 | 路径翻译：前缀替换、env 覆盖根、未知前缀/None 透传、record 级翻译 |
| tests/test_api_dry_run.py | 4 | 422、structured 直通（不碰 LLM）、截断标记、检索串回显 |
| tests/test_api_query_submit.py | 6 | 结构化落库回读、等价 intent 去重、不同 intent 不去重、force 绕过、done 任务不是去重目标 |
| tests/test_pipeline_structured_parse.py | 5 | 优先级链：structured 零 LLM 调用、LLM 失败回退 rules、**normalize 幂等性**（API 存 normalized + parse 再 normalize = 不变） |
| tests/test_registry_task_columns.py | 8 | 新库建表、旧库迁移、label/intent_json 回读、list_active_tasks、孤儿 requeue、心跳行双向排除 |

全量 suite：**210 个测试，3 failures + 20 errors——与 MCP 工作开始前的基线完全一致**
（既有问题：test_geo_download_skill 癌种子集漂移 3F；test_llm_extractor 硬编码容器路径 20E），
零新增失败。运行方式：`.venv/bin/python -m unittest discover -s tests`。

## 5. 事故记录（全部当日发现当日修复）

| # | 事故 | 根因 | 处置 |
|---|---|---|---|
| 1 | **测试泄漏到生产库**（10-16）：7 条测试任务入队，live daemon claim 开跑真 pipeline | test 把 REGISTRY_PATH 隔离写在 `setUpModule`——unittest 在模块体执行**之后**才调它，env 没生效绑到了生产库 | 停 worker → 删 7 行（register 未到达，datasets/map 零污染）→ 隔离改到模块 import 时；教训写入测试 docstring |
| 2 | **孤儿任务**（10-16→10-19）：aux 任务跑一半遇容器重启，永远停在 `running`，worker 空转 2.5 天 | claim 只认 pending，重启杀死的任务无人回收 | `requeue_orphaned_tasks()` 启动自愈（2f63c21），已实测生效 |
| 3 | **心跳行竞态**（10-19）：自愈把 `heartbeat_database` 也 requeue 成 pending，短暂可被 claim | 心跳行存在 task_queue 里且 status=running | requeue 与 claim_task 双向排除 `heartbeat_%`；`update_heartbeat` 的 REPLACE 30s 内自愈了现场 |
| 4 | **检索静默失败**（10-19）：隧道死后重跑 aux，0 候选"成功"完成 | 代理死 → esearch 异常被吞 → 空结果；且隧道本身是无人看管的 `ssh -f` | 隧道改 **autossh + systemd**（`methy-proxy.service`，跳板 34.26.3.123，断线重连+开机自启）；**检索失败语义**（失败应 fail 任务）记为待办，暂缓（专注 MCP） |
| 5 | **filter 阶段 13GB 内存炸弹 → 宿主机 OOM**（10-19，**已修复** `5970bc5`）：355 候选的组织类任务四轮全量重跑，每轮都在**第 303-307 个判定**附近被全局 OOM-killer 处决（dmesg：anon-rss 13.4GB；正常过滤全程仅 ~500MB）。自愈机制每次正确回收重跑 → 形成"崩溃→重跑→再崩溃"的**西西弗斯循环**（每轮浪费 ~300 次 LLM 判定） | **根因**：`fetch_series_matrix_sample_info` 把整个 series_matrix 读入内存（`resp.content` + `gzip.decompress` + `.decode` + `splitlines` ≈ 4 份并发副本）。芯片系列的 series_matrix 内嵌全量 β 矩阵——**GSE51032** 的达 3.1GB 压缩 / ~10GB 解压 → 13.4GB 峰值。它死于 resolve 中途 → 缓存永不落盘 → 每轮重跑必撞。**修复**：流式 + 增量 gunzip，读到 `!series_matrix_table_begin`（元数据都在矩阵表前）即停 + 64MB 解压上限兜底。**验证**：GSE51032 从必死变为 1.4s / 76.6MB 拿到 845 样本完整标注并正确判定 exclude（WBC）；aux 第五轮 2h03m 零崩溃跑完 355 候选。附带收益：所有大系列 resolve 只读文件头 | ~~防御性待办：requeue 加次数上限~~ **已修（10-26）**：`requeue_count` 超 3 次标 failed，杜绝崩溃循环被自愈放大（见 §7.5）|

## 6. 生产验证（compound 查询端到端）

场景即上游主 agent 的真实用例：主检索 = CRC 血浆/血清 cfDNA（病例 vs 健康对照）；
辅检索 = 肿瘤 vs 癌旁组织（TCGA COAD/READ，标志物空间）。

| 验证点 | 结果 |
|---|---|
| compound 一次提交 | `{main: e1baeefd…, auxiliary: f2555e73…}`，label 落库 ✓ |
| worker 日志 | `Claimed … label='main' structured=True` → `parse: method=structured cancer=COAD` ✓ |
| 检索串分叉 | main=液体活检串（136 候选）；aux=组织串（355 候选）——合并查询做不到 ✓ |
| filter 上下文 | 请求中样本类型与各自 intent 一致（cfdna/plasma/serum vs tumor/adjacent）✓ |
| 去重 | 等价 intent 重放 → 两 task 均返回原 id（含 running 中的），零新建 ✓ |
| cancel | pending 任务取消生效（顺带清理了误提交）✓ |
| main 结果 | 48 min 完成：126 exclude / 5 manual_review / 0 新下载（历史已下载数据保持 done，零重下载——缓存修复生效）✓ |
| aux 结果 | **完成（第五轮，修复后）**：2h03m 零崩溃（前四轮均死于 §5 #5 内存炸弹）。355 候选 → 63 download / 4 manual_review / 288 exclude；GSE51032（前炸弹）845 样本正确判定 exclude(WBC)；tcga-direct 首次触发（1 个 GDC 数据集）；62 个 pending 转异步下载。结果经 MCP `get_task_result` 取回，label=auxiliary ✓ |

## 7. 上游反馈迭代：字段暴露补齐（feb447c + c92ff35，2026-10-20）

上游平台 agent 联调反馈：**MCP 模式下资格判定到不了 eligible**——其硬规则要求从
`disease_groups` 文本解析 `reported_control_count`（缺失 → `COMPARATOR-NOT-EVIDENCED` →
候选最多 inconclusive），而下载/物化门要求 eligible。旧 CLI 模式能跑通 solely 因为直接读
SQLite；MCP 只能走接口 → 全部候选卡死。10 候选 smoke：0 eligible / 8 inconclusive，主因
即字段缺失。

**第一轮（feb447c）**：`disease_groups` + `sample_level_annotation` 加入
`DatasetResponse`、`get_datasets_by_task_id` SELECT（不加它任务桶永远是 None）、MCP
list/桶条目。

**第二轮全量清查（c92ff35）**：datasets 31 列逐一比对，再补 1 个 bug + 6 个缺口：

| 字段 | 库内行数 | 性质 |
|---|---|---|
| `sample_metadata_path` | 76 | **bug**：MCP 自 Phase 1 宣称返回但 DatasetResponse 从未携带 → 静默 None。三处修复（含路径翻译），与 `sample_level_annotation=yes` 配对 |
| `stage_treatment` | 56 | 与 disease_groups 同类的资格输入（SPEC 分期/治疗标准） |
| `recommended_action` + `reason` | 84 | 最后判定 verdict + 一句话理由，数据集粒度端点原先拿不到 |
| `notes` | 84 | SPEC 警示（pooled、泛癌需拆子集、样本数与文章不符）——直接影响平台资格 |
| `available_file_type` | 6 | phase-2a 文件形态证据 |
| `TaskResponse.intent` | — | 解析后的 intent_json——复合查询消费方核对实际运行的结构化意图 |

**有意不暴露**（留档）：`checksum_md5`（0 行）、`usable`（可从 recommended_action 派生）、
`datasets.task_id/raw_query`（任务作用域溯源，走任务端点）、`samples.characteristics_json`
（批量明细，样本级详情在 sample_metadata.csv / gsm 缓存）。

实测三路全通：`get_dataset`、`get_task_result` 桶、`get_task_status`（label + intent）。

### 7.1 协作式任务取消（第二轮反馈）

上游反馈：running 任务不可取消——错误提交只能烧完整个 ~1h / ~136 次 LLM 判定的运行。
实现协作式取消：`cancel_task` 对 running 任务置 `cancel_requested`（API 立即返回）；
pipeline 在**每个候选判定边界**检查标志（一次索引 SELECT，µs 级）并抛
`TaskCancelledError`；daemon 捕获后落 `cancelled`。启动自愈同步升级：孤儿回收时
`cancel_requested` 直接落 cancelled（不复活已被取消的任务）。

实测：18 个判定后取消 → **20 秒内** finalized `cancelled`。

排障记录（smoke 中抓到）：`TaskCancelledError` 初版继承 `RuntimeError`，被
`_run_concurrent` 的 `except RuntimeError` 回退分支捕获并在新事件循环里重跑 gather →
Semaphore 跨循环 → 二次异常 → 任务误落 `failed`。改为继承 `Exception` 解决。
另修复一处环境事故：协作者用自己账号（uid 1002）直接开过 registry 库，留下的
`-wal`/`-shm` 文件归属 1002，容器进程（uid 1000）写不了 → API 全部 readonly——
chown 回 1000 恢复。**教训：不要绕过服务直接开生产 SQLite；必须开时用容器同 uid。**

### 7.2 TCGA 行的 unknown 桶（第三轮反馈，2026-10-20）

上游反馈：`recommended_action=null` 的行（实测 TCGA-COAD）落进 `get_task_result` 语义不明的
unknown 桶，消费方只能保守跳过。**属实**：全库唯一 NULL 行就是 TCGA-COAD——tcga-direct 设计上
绕过 geo-filter（"Level-3 β 值是标准可用数据，直接信任"），但 register 时没给判定字段。

修复（语义补全，非简单填值）：

- `files_to_dataset_records` 为 TCGA 记录盖**设计信任章**：`recommended_action='download'` +
  reason（"trusted by design, bypasses geo-filter"）+ `sample_level_annotation='yes'`
- GDC 查询补取 `cases.samples.sample_type` → 聚合出**真实**的 tumor/normal 计数填入
  `disease_groups`（平台解析对照数直接可用）：TCGA-COAD 实测
  `Primary Tumor 437 / Solid Tissue Normal 61 / Metastatic 1 / Recurrent Tumor 1`
  （跨 450K/EPIC 两个平台组聚合）
- 存量行已用真实 GDC 数据回填；验证 `get_task_result`：TCGA-COAD 现落 **download 桶**（64 项）

已知小瑕疵留档：同 accession 多平台组 upsert 时 disease_groups 会被后写组覆盖（本次回填用聚合
绕过）；pipeline 侧聚合留作小改进。测试 +4（`test_tcga_records.py`），共 228，基线不变。

### 7.3 审批队列 tool（第四轮反馈，bd76ce2）

上游反馈：manual_review 数据集只能走 REST/Web UI 审批，MCP 消费方在审批门前是死路。加 3 个
tool（共 12 个）：`list_awaiting_approval`（队列 + LLM 证据）/ `approve_dataset`（→ 入下载队列）/
`reject_dataset`（→ skipped）。写门控 `MCP_ALLOW_WRITES` **默认开**（与 submit_query 同权限面；
与原计划"默认关"的偏差及理由记录于 §8）。首次批量实战：17 项待审一次清零（3 批准全部下载成功，
其中 GSE97923 命中 773 样本已处理 ctDNA 数据）。

附带事故（已恢复）：该测试模块首版在 **import 时**重绑共享 `api.main`，discover 全量导入时与
其他模块的 tearDownModule 重载碰撞，4 行合成数据写进生产库——已删除，测试改为 setUpClass 绑定。
同族第二次事故；规则固化：**共享模块的重绑只能在 setUpClass，绝不在 import 时**。

### 7.4 dry-run 超时（第五轮反馈，dcccd73）

上游反馈：`search_datasets` 耗时超 CC 默认 MCP tool 超时（30s）。插桩定位（136 accessions）：
大 esearch 4s + eutils 批次调用**单次 ~5s**（代理延迟）+ 136 个 supp 目录列表批内串行（0.44s/个）。

修复三处（`GEOClient`）：`_get` 改**全局线程安全限速器**（顺带修复 filter 3 线程并发下限速被
线程数放大的潜在 NCBI 违规）；批次间 4 路并行（`executor.map` 保序，检索排名不变）；supp 列表
批内 8 路并行（GEO FTP，非限速 eutils）。**实测：search 阶段 60s → 7-9s（候选逐一致）；端到端
dry-run 14.4s**。生产任务同路径免费提速。CC 消费方兜底 `MCP_TOOL_TIMEOUT` 已写入使用说明。

### 7.5 可靠性双修（2026-10-26，自主排项）

**截断重试（backlog §6-7）**：filter LLM 响应被端点总 token 预算截断（prompt+completion
合计 ~8-11k；`max_tokens` 参数解不了）→ JSON 解析失败 → 静默降级伪 manual_review。修复：
截断时**压缩输入重问一次**（8 个代表 GSM、摘要截 1000 字、显式精简 directive——语义与
SPEC 不变），仍败才降级且 reason 标 `filter_error_truncated`（区别于 `filter_error`）。
真实验证：曾截断的 GSE289742 现产出真实 exclude 判定（64 COPD + 64 对照全血 cfDNA，非目标
疾病）。测试 +4（重试恢复带 `_truncation_retried` 标记、压缩消息更短且含 directive、持续
截断区分标记、非截断错误仍单次调用）。

**requeue 次数上限（§5 #5 遗留防御项）**：自愈 + 周期性崩溃 = 无限重跑烧钱（OOM 事故四轮
~1200 次浪费判定的教训）。`task_queue` 加 `requeue_count`（迁移已跑生产库）；孤儿回收超过
3 次即标 `failed`（error 说明疑似崩溃循环、修因后可重提），不再复活。测试 +3。

## 8. 已知限制与下一步

- ~~filter 内存炸弹~~ **已修复**（`5970bc5`，见 §5 #5；355 候选任务验证通过）
- **Phase 3（部分完成）**：~~approve/reject 写操作 tool~~ ✅ 已加（2026-10-22，第四轮上游需求）：
  `list_awaiting_approval` + `approve_dataset` + `reject_dataset`（12 个 tool）；写工具门控
  `MCP_ALLOW_WRITES` **默认开**（与 submit_query 同权限面——同一无鉴权 localhost API，门控供远端
  暴露时收紧）。剩余：远程消费方（streamable HTTP transport + token 鉴权）、提交侧并发/配额护栏
- **Phase 4（未做）**：配套 SKILL.md（教 agent "dry-run→submit→poll→取路径" 工作流）
- **verdict cache：确定不做（设计决定，2026-10-20）**——每次重新 LLM 判定是故意设计：
  判定标准的主战场在 skill 的 SPEC.md（用户维护），SPEC 更新后 verdict 必须跟着变，缓存会
  违背"SPEC 是唯一事实源"的原则。重复查询/重跑的 token 成本为此设计的固有代价
- **检索失败静默**：见 §5 #4，语义修复待拍板
- get_task_result 的 exclude 默认只计数（防上下文溢出），需要全量时 `exclude_limit`
