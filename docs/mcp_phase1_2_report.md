# MCP 接口 Phase 1+2 实施报告

日期：2026-10-19
分支：`MCP`（已推送 origin/MCP；main 未动）
提交：`2bbff73`（Phase 1）→ `ec97629`（Phase 2）→ `2f63c21`（daemon 自愈修复）
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
| api | `POST /search/dry-run`：query（LLM parse）或 intent（确定性）→ parse + geo-search，无判定无写入；同步 handler 跑 FastAPI threadpool（实测 ~50s 不阻塞事件循环） |
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
| 5 | **filter 阶段 13GB 内存炸弹 → 宿主机 OOM**（10-19，**已修复** `5970bc5`）：355 候选的组织类任务四轮全量重跑，每轮都在**第 303-307 个判定**附近被全局 OOM-killer 处决（dmesg：anon-rss 13.4GB；正常过滤全程仅 ~500MB）。自愈机制每次正确回收重跑 → 形成"崩溃→重跑→再崩溃"的**西西弗斯循环**（每轮浪费 ~300 次 LLM 判定） | **根因**：`fetch_series_matrix_sample_info` 把整个 series_matrix 读入内存（`resp.content` + `gzip.decompress` + `.decode` + `splitlines` ≈ 4 份并发副本）。芯片系列的 series_matrix 内嵌全量 β 矩阵——**GSE51032** 的达 3.1GB 压缩 / ~10GB 解压 → 13.4GB 峰值。它死于 resolve 中途 → 缓存永不落盘 → 每轮重跑必撞。**修复**：流式 + 增量 gunzip，读到 `!series_matrix_table_begin`（元数据都在矩阵表前）即停 + 64MB 解压上限兜底。**验证**：GSE51032 从必死变为 1.4s / 76.6MB 拿到 845 样本完整标注并正确判定 exclude（WBC）；aux 第五轮 2h03m 零崩溃跑完 355 候选。附带收益：所有大系列 resolve 只读文件头 | 防御性待办（未做）：requeue 加次数上限（如 3 次后标 failed），避免任何崩溃循环被自愈放大 |

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

## 7. 已知限制与下一步

- ~~filter 内存炸弹~~ **已修复**（`5970bc5`，见 §5 #5；355 候选任务验证通过）
- **Phase 3（未做）**：approve/reject 写操作 tool（`MCP_ALLOW_WRITES` 开关）、远程消费方
  （streamable HTTP transport + token 鉴权）、提交侧并发/配额护栏
- **Phase 4（未做）**：配套 SKILL.md（教 agent "dry-run→submit→poll→取路径" 工作流）
- **verdict cache**（caching_audit #4）：本次事故中它的缺失被放大四倍——四轮重跑重复判
  定 ~1200 次；有了它，OOM 重跑可从断点近似恢复。建议最优先排期
- **检索失败静默**：见 §5 #4，语义修复待拍板
- get_task_result 的 exclude 默认只计数（防上下文溢出），需要全量时 `exclude_limit`
