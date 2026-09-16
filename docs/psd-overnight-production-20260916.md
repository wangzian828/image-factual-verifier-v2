# PSD 夜间正式推进（2026-09-16）

## 2026-09-17 02:35 接续检查及备用恢复入口

- v3采集持续推进到338/3200，3普通错误、0待解基础设施槽，并发40；v3累计6次
  模型超时走原预算重试，仍无新增NaN。collector、v32存储助手和GPU guard均身份匹配存活。
  02:30四卡瞬时100/100/100/88%；最新空间准入约15.0GiB，准入开放。
  无新的阶段完成或待用户处理问题，不因例行检查反复通知。
- 已预先补齐**历史5次预算槽在下一代恢复中不能复用**的兼容缺口，未为此停止当前采集。
  备用不可变快照`/volume/ybo/wza/training-artifacts/psd-completed-recovery-20260917-v33/code`，
  **522项服务器回归通过**、冻结Agent未变；本地62项定向测试通过。
  v33继承v32全部读取能力，只用于后续需要恢复/接续的新进程；活动collector仍v30、
  存储助手仍v32。没有启动v4，也没有重新采任何完成槽。
- 仅允许**已完成**的5次预算槽复用：校验原allowance、原3次失败ledger及重启/64次
  回放证据，再检查前3次历史逐项未改、总尝试4或5、最后completed；按原样传递allowance，
  原失败仍计数。下一代`retry_episode`只读缓存，不调用模型。支持gzip缓存并记录archive SHA。
  新增统计将“本次扩大预算”和“继承既有额度”分开；继承不会再送两次机会。
  缺allowance、证据变化、抹掉旧尝试、running/耗尽或未解决NaN均在写目的目录前拒绝。
- 4个真实历史扩展槽已只读核验：全在累计第4次完成、原证据有效、前三次历史未改、
  缓存与canonical一致。真实检查没有调用模型或创建新的结果；反复迁移且不重采的行为由
  定向测试验证。不要把该备用修复说成正式训练已经启动。
- 朋友Pro继续原串行队列：最近20成功、22失败、394未尝试；Flash436完成不重跑。
  旧judge不重交，最终目标构建仍待3200源槽齐全。

## 2026-09-17 02:14 存储助手已换 v32，正式采集未中断

- 当前 v3 249/3200 canonical；3 普通模型错误保留，0 待解基础设施槽，另40槽在途。
  v3 内累计4次模型请求超时已按预算重试，尚无新增 NaN；历史 v1/v2 NaN 不混计。
  GPU 单次观测100/99/100/100%，四后端均健康。正式 optimizer 仍0，不冒充训练启动。
- **后续 reader 改用 `/volume/ybo/wza/training-artifacts/psd-storage-spool-20260917-v32/code`**，
  515项服务器回归通过、冻结 Agent 未变；本地14项存储/恢复测试通过。
  v32完整继承 v31 压缩缓存读取语义；原 v30 collector/PID1225918完全未改、未停。
- v31存储助手在一轮完成后的休眠窗口正常停止，原receipt保存在
  `RUN/storage-compaction-v31-process.json`。新v32已真实完成一轮、进程身份已核对；
  权威receipt仍为`RUN/storage-compaction-process.json`（初始PID1234814仅线索），
  另留`storage-compaction-v32-process.json`；新日志`storage-compaction-v32-controller.log`。
  不要照旧段落重启v31，也不要同时运行两个存储助手。
- 两轮累计无损回收估计8,436,441,881 bytes（约7.86GiB）。每轮明细/回收量在
  `storage-compaction/*.json`；`storage-compaction-latest.json`仅表示最近一轮，不是累计值。
  02:13空间准入检查约10.91GiB、准入开放，128GiB上限不变；不是个人剩余配额。
- 存储助手的gzip/硬链接/JSON临时文件全部移到经边界及同文件系统检查的
  `/volume/ybo/wza/tmp/psd-compact-*`专属目录，再原子发布到持久目标路径。
  这避免助手临时文件消失导致collector的`du`扫描ENOENT；已用测试覆盖临时源位于RUN外。
  不声称消除了原collector自身并发写临时文件的一切扫描竞争，也未观察到因此停采。
  已压缩且native/canonical同inode的槽不重复解压扫描，真正消费者仍逐次校验缓存SHA。
- 完整原缓存字节、图片、工具/请求历史、各代run和原权重均保留。
  当前阶段完成后按新reader接checker/repair/raw teacher/最终四卡global32恢复验收和5epoch。

## 2026-09-17 02:01 存储读取版本更新（不要热改采集进程）

- v3仍由**原v30 controller/PID1225918**持续采集；一次观测201/3200、3普通错误、
  0待解基础设施槽。当前run新增2次模型超时按预算重试，尚无新增NaN。
  四卡服务/guard正常。未开始checker或optimizer。
- 01:51作业约11.3GiB，原样保存的增长有撞128GiB准入上限风险。发现完整native trace与
  canonical trace字节相同，却各占空间；结果缓存又保存一整份相同payload。
  共享df仍不是个人配额，机器没有quota/mmlsquota工具，不抬高准入上限掩盖问题。
- 后续**读取、审核、恢复、目标构建必须用**
  `/volume/ybo/wza/training-artifacts/psd-storage-reader-20260917-v31/code`或其后继，
  **514项服务器回归通过**，冻结Agent src未动。v31包含v30全部功能，并使`load_bound`
  支持`ifv-psd-bound-gzip-v1`结果缓存。不要用v30重新启动/恢复已经压缩的run。
- 独立`compact_psd_completed_storage.py --follow`只处理v3内**已完成且加锁**的槽：
  canonical与native trace在SHA与原字节核对后改为硬链接，路径/内容不变；原`result.json`
  的完整字节保存在同槽`result-original-<SHA>.json.gz`，原路径成为带SHA/长度/identity/
  payload哈希的小索引。新loader解压验证后返回完全相同的payload，测试确认不会重采。
  不压缩/删除images、events、contexts、snapshots；不碰v1/v2/旧权重，也不改活动CODE30。
  gzip不是只存摘要；如需要回退旧reader，可从该gzip无损恢复原result.json字节。
- 独立进程receipt为`RUN/storage-compaction-process.json`（初始PID1233239仅线索），
  日志`storage-compaction-controller.log`，每轮结果`storage-compaction/*.json`及
  `storage-compaction-latest.json`；每300秒继续处理新完成槽，采集结束/离开采集阶段后退出。
  读取要求写在`storage-reader-requirement.json`。不重复启动，不与新run混用。
- 活动collector只对未完成槽生成、返回后不再读取该槽cache，最终采集统计只查cache存在及
  retry-state；所以无需热改或打断它。后续新阶段由接续任务明确改用v31。
  原始结果全部可恢复，实际节省以每轮receipt为准，不以逻辑文件大小冒充物理占用。

## 最新接续：2026-09-17 01:07，服务恢复后补四槽

本节覆盖下文 v2 在途快照；优化器仍 0 step，不把采集称为训练。

**01:14验收更新：**v3首40已齐，四个耗尽槽全部在累计第4次完整尝试恢复，未用第5次。
`first40-gate.json passed=true`；37完整报告的strict audit全部通过（24个warning，
0 scheduler/protocol/route rejection），3个普通unusable失败保留，0待解infra。
`state.json`已进入`collecting_remaining_3160`，观察到四副本各8–9个在途请求。
当前是正式3200槽采集，不是优化训练已启动。代码提交`3215e0f`已本地push。

- v2 首40结束：36份可进入后续审核的原始结果（33正常报告、3 unusable输出），
  **4槽三次基础设施尝试全部 HTTP400/NaN 耗尽**，controller 已保护退出，未发剩余3160。
  16个原NaN槽中12个恢复；不能说重试已经消除了数值问题。
- 全量后端日志中0/2号无同类错误；1/3号分别14/13个序列化错误，且两者遗留
  `VLLM_COMPUTE_NANS_IN_LOGITS=1`，另外两者没有。这只是关联。
  安装版源码显示此开关在采样后计数，并非数值修复；统计位置不是原始层级logits证据。
- `SERVICE/nan-exhausted-before-v1`：四个耗尽槽的最后失败请求，分别在四卡重建完整
  请求并流式诊断，共16次；两次在首token即token0/null概率（GPU1/3）。没有执行返回动作，
  没有把诊断输出当作训练源；不是原始wire逐字节回放，不裁剪图像/上下文/预算。
- `SERVICE/nan-metric-restart-ab-v1`只重启身份确认的GPU1/3：GPU1将NaN统计关为0，
  GPU3保留1作为重启对照；原命令、模型、JIT缓存位置不变，GPU0/2未动。
  重启后`SERVICE/nan-exhausted-metric-ab-v1`四请求×四GPU×四轮 **64/64完成**，
  所以**不能证明开关导致异常，更不能声称永久根治**；当前只是有真实验收的恢复措施。
- 新快照`/volume/ybo/wza/training-artifacts/psd-service-recovery-20260917-v30/code`，
  **505项服务器回归通过**，包含v27全部下游准入修复，冻结src未改。
  同时拒绝选中/候选概率null（合法top-k负无穷仍允许）；旧活动快照不热改。
- 当前输出 **`/volume/ybo/wza/runs/psd-production400x8-20260917-v3`**；控制器是v30父目录
  `run_psd_production_collection.py`，初始PID1225918仅线索，核process.json及准确命令。
  新run保留旧36份结果，只给4个耗尽数值槽各额外最多两次，**总上限5、旧三次仍记账**。
  每槽`recovery-allowance.json`绑定原ledger SHA及重启/64次回放证据；不能对5次预算再次扩展。
  其余3196槽仍默认3次。正常模型错误不重采挑成功。v1/v2所有原文件未改。
  01:07控制器已启动，仍须核首40真实产出/strict audit，不能据启动宣称首批通过。
- 首40验收通过自动并发40接剩余3160；完整3200后继续source checker/slate修复/
  preservation/raw teacher/最终DP4 global32短步和恢复门槛，再独立5epoch优化。
  下游用v30与最终完整v3，不误用v1/v2残缺manifest。原SFT权重全保留。

诊断和恢复代码：`probe_psd_failed_slots.py`、`psd_nan_metric_ab.py`；诊断部署v29，
v28在选取native错误receipt时已拒绝、未发请求。服务/守护的权威receipt仍在SERVICE，
不能照抄历史PID。朋友Flash已436/436结构成功；Pro最近12成功/19失败/405未尝试，
继续串行流程，不把这个快照当最新完成量。旧judge均已结束，不重交。

## 最新接续状态：2026-09-17 零点巡检后的恢复

以下覆盖后文首次启动的路径/状态，不重启 v1：

- v1 首40最终全部落盘，其中22份正常报告、2份 unusable policy output、16份
  **HTTP400/NaN serialization**。原 v1 controller 已在40份结果全部缓存后停止；
  `drain-watch.json` 证明原40槽全部落盘，没有开启剩余3160槽。
- 重试缺口不是泛指400：旧分类器遗漏了 vLLM 的精确
  `BadRequestError/code400/Out of range float values are not JSON compliant: nan`。
  修复版本 v2 同时在原生响应解析之前检查非有限概率；普通400、空工具动作、答案错误
  不会仅因此重试。旧 NaN 的每份 native context error receipt 已独立核对。
- 本地修复提交 `74cacfd` 已推送。采集使用独立不可变
  `/volume/ybo/wza/training-artifacts/psd-numerical-recovery-20260917-v25/code`，481回归通过。
  活动控制器是其父目录的 `run_psd_production_collection.py`。
- **当前正式输出为 `/volume/ybo/wza/runs/psd-production400x8-20260917-v2`**，初始PID1220105
  只作线索，查 `process.json` 及当前命令。复用原24份完整结果，16份 NaN 仅使用剩余
  两次预算；`slot-recovery.json` 绑定所有原记录哈希，原 v1 没有被改写。
- 恢复后一次观察32/40已落盘（含复用24），3份 unusable output、0耗尽基础设施槽；
  余8份在运行。不能把这解释为3200完成，优化仍0step。首40通过后自动接3160，
  仍并发40、同图/seed/生成参数，不换服务/不缩减预算。
- 下游独立代码为
  `/volume/ybo/wza/training-artifacts/psd-completion-admission-20260917-v27/code`，**493回归通过**。
  v26仅因测试夹具依赖不齐而失败，保留作为历史，不使用它，也未热改活动collector。
  修正 review/postprocess/round 对 completed_with_errors 的拒绝：仅当完整8槽覆盖、
  每槽持久化ledger/result/canonical一致、无未解决基础设施错误时承认采集已结束；
  原manifest及失败结果不改。已核对真实3份 unusable output 的 native错误receipt和
  有限capture前缀，可作为待修复源；这不是preservation准入，也不是宣称根因已查明。
  未知错误、NaN、缺轨迹仍拒绝/隔离，不用宽泛“任何RuntimeError都正常”规则。
- source checker/repair/raw teacher/最终DP4batch32及5epoch仍在采集之后，尚未开始。
  准备阶段使用v27，指向v2完整episodes及v2 snapshot，不能误用v1 running manifest。
- 朋友Flash组已436/436结构成功；不重复运行这组、不将结构通过当视觉质量验收。
  Pro组仍继续，曾观测12成功/18失败/wave0，不以这个旧快照替代最新状态。

未知个人配额及128GiB新run准入上限仍保持。原v1约2.2GiB，恢复没有复制旧runtime图片树；
新旧源结果都保留，后续按实际增长评估存储，不能用共享df当个人额度。

用户已授权离开后自主推进到完整 PSD 优化训练启动。固定 400 张 × 8，
基于三 epoch SFT step3084；不扩大到 4,000 张，不覆盖原权重。

## 顺序与验收

1. 正式采集 3,200 槽，四卡、并发 40；首 40 槽在线工程验收，完整产物直接保留。
   每槽最多三次基础设施尝试，正常模型失败不重采挑成功。全量逐槽核对身份、seed、
   原图与完整 native capture。首批通过后仍保持连续 40 并发，不按 40 一批反复等长尾。
2. 全部源轨迹 checker，失败任务按上游规则选择源、生成提示并真实完整重跑；每题
   6 次完整修复/12 提案，跨题并发 40。preservation 保留真实通过轨迹。
3. 同一冻结 Qwen 的 exact IDs/pixels 原始 top20 教师评分；服务采样 logprobs
   不冒充无掩码教师目标。私有答案不进入 Agent/提示生成器。
4. 用最终数据做四卡 global batch32 短步验收：有限 loss/梯度、参数真实变化、
   原生全状态保存恢复。此前 199 目标单 GPU raw gate 与旧 DP4 gate 不替代这一步。
5. 独立输出启动 5 epoch PSD：LoRA32/alpha32/dropout0，LR4e-5，Adam(.9,.95)，
   eps1e-12，wd0，top20，目标权重1，token loss SUM。多个 optimizer step 正常
   推进才报告“正式训练已启动”，不能把采集/服务启动算成训练。

完成耗时取决于源调查和真实修复；用首批及后续完成率估算，不能保证今晚一定进入优化。
未达到目标也继续自主推进，不因为一个阶段结束而停止。

## 当前执行入口

- 不可变已验证采样代码：`/volume/ybo/wza/training-artifacts/psd-infrastructure-retry-20260916-v23/code`。
- 正式控制器：本仓库 `scripts/server/run_psd_production_collection.py`。
- 正式输出：`/volume/ybo/wza/runs/psd-production400x8-20260916-v1`。
- 网关 `http://127.0.0.1:19025`，四 stock worker 19002–19005。
- 控制器仅调度并缩小内存中的结果摘要，不删除磁盘完整轨迹、改变 Agent 或生成参数。
- 状态文件 `state.json`、`collection-progress.json`、`first40-gate.json`、`storage.json`，
  权威进程凭证 `process.json`。复查活进程和 `controller-lock`，不可重复启动。
- 后续使用 `scripts/run_psd_round.py prepare` 的完整输入绑定，repair mode slate、
  task-source-selection longest_failed、expected-rollouts-per-case8、case-concurrency40、
  defer-topk。正常模型错误与未解决基础设施失败须明确区分，不能改 manifest 掩盖错误。
- 最终 raw teacher 与 DP4 launcher 需绑定本次最终 bank；旧 canary 的硬编码入口不能直接运行。

## 无人值守与保护

恢复已有 `gemini-3-1-pro-agent` heartbeat，每15分钟检查并推进，不新建重复任务。
本机/Codex 要保持运行才能继续 AI 判断与修复；服务器 collector/GPU guard 独立存活。
正式训练验收后恢复每小时监控训练及朋友实验；旧已完成 judge 不重新付费提交。

所有服务器写入仅 `/volume/ybo/wza`，禁止服务器 Git，凭据不输出。
保留所有旧 SFT1/2/3epoch 权重和原生 checkpoint；新 PSD 单独目录。
GPU guard 继续真实计算脉冲，阶段交接排空身份明确服务，不能抢训练显存。

个人 GPFS 配额仍未知，45TiB 是共享余量而非个人配额。新 source run 设置128GiB
准入上限（不是用户配额承诺），每60秒检查新目录实际占用和共享盘16GiB最低余量。
达到上限仅停止新槽准入，让在途请求完成并保留证据，由接续任务诊断空间，不盲删旧数据。
网关重复 wire 归档关闭，native 完整 token/图片/工具轨迹不缩减。首40后估算总增长，
及时处理而非等写满。上限可能被已有在途请求越过少量，不宣称硬磁盘配额。

本文件为执行计划；实际是否启动及完成量以服务器上述状态与进程为准。

## 首次启动实况

- 代码 `fe5522d` 已从本地推送；本地及服务器6项新增定向测试均通过。
- 部署入口 `/volume/ybo/wza/training-artifacts/psd-production-collection-20260916-v24/run_psd_production_collection.py`。
- 正式控制器已启动（初始PID1216870只是线索），40个持久化槽与40个真实runtime已存在，
  四副本同时有在途请求；一次观测 GPU 利用率为100/89/100/100%。
- 首条已结束轨迹 `main-02754--d5fdfd7f--r003` 是 unusable tool response，
  `finish_reason=tool_calls,content_chars=0,reasoning_chars=1`。原始错误保留；
  不能凭该文本认定 NaN，也不能为了得到正确答案自动重采。终端输出是
  `deterministic_segment_boundary=true`，没有模型capture是预期的异常终止记录，
  在线capture检查跳过该机械段而不跳过真正模型输出。
- 后续接续要重点核对既有下游 `review_psd_sources.py`、`postprocess_psd_training.py`
  和 `psd_round.py` 的 `status == completed` 门槛：正常模型错误会让 native manifest
  写成 `completed_with_errors`。需要用完整槽位/基础设施ledger证据区分结束状态与
  模型答对率，不能篡改 manifest 为 completed、丢失败槽或重采到全部无错误来过门槛。
  这是待完善的下游接续点，尚未修改或声称已通过。
