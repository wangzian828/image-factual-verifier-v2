# PSD 夜间正式推进（2026-09-16）

## 2026-09-17 16:06 PSD评分优先，朋友Pro在当前样本后暂时排空

- PSD collector仍健康，16:04为3020/3200、0 normal error、0 pending infrastructure，
  四卡均100%利用率；在线source review为2905条有效、70 pending、5 exhausted transport。
  新增耗尽项仍是HTTP 429，不是轨迹或policy错误。
- 朋友Pro与PSD source reviewer共用同一外部Gemini额度；固定样本`sample-0220-fa084f97`
  的实时日志也连续返回quota 429。继续同时压测会增加PSD评分的共享冷却和有限attempt耗尽，
  与当前“PSD优先”冲突。
- 已在核验receipt与完整cmdline后仅向朋友Pro控制器PID1063898发送SIGTERM。其signal handler
  只设置STOP并排空当前样本，不粗暴杀sandbox、不覆盖结果；当前attempt继续到正常有界结束，
  随后状态应为`stopped_after_drain`。Flash旧结果不动。
- 这只是为PSD source review排空而暂停，不取消朋友实验。PSD评分和后续外部repair/judge不再受
  当前Pro请求竞争后，再从既有campaign状态恢复；成功样本不重跑，失败仍按原最多3个补跑wave处理。

## 2026-09-17 15:31 采集继续，识别出2条评分传输预算耗尽

- collector仍为CODE41/v6且精确进程身份正常；2878/3200完整、0 normal error、
  0 pending infrastructure，四张H20实时利用率均为100%。按14:37至15:30的实际增量估算，
  剩余322槽纯采集约需1.5小时；这只是动态估计，不把它写成完成承诺。
- CODE42在线评分可见2871、已审核2777，active16、queued27；1887 fail、864 pass、
  26 unresolved、56 pending error。评分仍在跟随新增轨迹，没有停。
- `exhausted_transport=2`已定位为两条在4次外层预算（每次内部最多2次）全部收到HTTP 429，
  不是policy fail、解析错误或轨迹错误。原记录与计费attempt必须保留；当前不重启活跃reviewer、
  不清预算无限重抽。待全量排空时与其余pending一起按CODE44正式切换门槛做有界补齐，
  未补齐前不得宣称full collection admitted或进入最终bank。
- compactor持续无损处理且每轮报告原始缓存可恢复、轨迹字节不变。朋友Pro独立串行任务为
  166成功、51失败、219未尝试，仍在运行；不抢PSD的GPU采集。
- 正式PSD optimizer仍为0 step；采集完成不等于训练启动，后续source review排空、
  prepare/repair、raw teacher、最终bank DP4 global32恢复验收与5 epoch顺序不变。

## 2026-09-17 14:05 AVIF真实评分验收通过，采集继续

- collector/reviewer/compactor/guard精确身份均正常；v6采集2546/3200、0normal_error、
  0pending_infrastructure，32个durable running槽，四卡实测100%利用率。存储测量已从超时自动恢复，
  admission_open=true、按inode去重约120.3GiB；不把共享free当个人quota。
- CODE42评分2490条有效、44pending、10在途、queued0、exhausted_transport0；有新轨迹即继续，
  不代表全部3200已评分。朋友Pro161成功/49失败/226未尝试，仍正常串行。
- 在活跃评分只有10/16时使用1路空闲额度，首次对固定真实AVIF轨迹main-05936 r003执行CODE44完整
  Gemini3.1ProPreview source judge；不是视觉简化probe。返回合法`fail`并通过全部trace/gold/image/request
  binding及literal证据校验，artifact/原trace/原AVIF哈希写入CODE44父目录
  `live-avif-canary/{artifact,result}.json`，原文件不改，active reviewer/collector均未重启。
  这证明PNG传输适配可被真实接口接收且产出可验收artifact；`fail`是评分结果，不是工程失败。
  不为追求pass重抽同一轨迹。其余AVIF在正式流程按相同确定性转换与精确缓存处理即可。
- 正式prepare/repair/raw teacher/最终bank继续以CODE44为当前候选；19条JSON转义引用可从原缓存机械恢复，
  2条真实改写仍pending。在线CODE42先自然跑完，不能热替换活动进程；optimizer_steps仍0。

## 2026-09-17 13:36 找到并修复JSON引用误报，CODE44已做真实缓存验收

- 活跃collector仍CODE41/v6、prefetch仍CODE42/v2-auto-retry，不重启、不热改；2476/3200完整，
  0normal_error/0pending_infrastructure。评分2101有效、39pending、16worker、0传输额度耗尽，
  原错误自动重试在继续。存储admission_open=true、按inode去重117.77GiB，个人quota仍未知。
  13:27朋友Pro160成功/45失败/231未尝试继续串行；模型服务与guard精确身份正常。
- 将旧21条引用错误逐条禁网重放，原来27处bad quote均不是索引错位；进一步检查发现其中19条
  只是**工具结果中的完整JSON字符串未解码**（如真实引号存成反斜杠转义），或quote自身是JSON编码的
  带外引号子串。并非19条都在编造证据。证据在CODE43父目录literal-evidence-diagnostic.json和
  literal-json-encoding-diagnostic.json，未发新provider请求。
- CODE44=`/volume/ybo/wza/training-artifacts/psd-literal-json-20260917-v44/code`，含CODE43 AVIF修复；
  代码`2940ff8`已本地push，89本地定向/510服务器PSD回归通过，Agent逐字不变，父目录stage-state及
  code-binding齐全。新增显式`evidence_encoding=ifv-psd-literal-json-strings-v1`，仅在原严格路径和
  单次纠正后的最终决定仍因literal错误被拒时，允许机械读取完整JSON的字符串值（最多3层）及
  quote的单层JSON字符串编码。原step_index不变，仍要求同一字段内逐字/原有显式省略规则匹配；
  不折叠空格/大小写、不改数字、不换标点、不跨字段拼接、不模糊匹配，不修改Gemini判决或引用原文。
  新解释标记可复验；旧有效artifact及原严格纠正链完全保持，不重新选择之前已被拒的首个判决。
- CODE44父目录`real-json-evidence-canary.json`：21条真实轨迹中19条验收恢复、2条仍pending，
  0新provider调用；恢复的artifact另存offline-source-reviews，旧记录不改。4条plain+4条原corrected
  有效对照逐字相同。后续正常formal review会从相同原缓存重现这些artifact，不要手改活跃prefetch记录。
- 剩余2条确有词语改写：一个把原文“the athlete's bib displays”写成“The bib shows”，另一个漏掉
  “visual”。仍拒绝，不用最长相似子串补证据；需要后续有边界的引用修复方案，不能伪装已全部解决。
  新增轨迹可能有其他格式错误，应按相同严格方法诊断，不能假定所有39pending都由这19条解释。
- 当前不要为升级评分而中断在途调用。安全排空/正式prepare时使用CODE44（替代候选CODE43），
  仍指向v6/source-review-prefetch-v2-auto-retry以复用缓存；AVIF固定轨迹真实Gemini验收尚待空闲额度，
  不额外叠加第17路。最终bank DP4 deployment改为psd-literal-json-20260917-v44且必须显式ready。
  源评分/完整bank/冻结raw教师/global32恢复/5epoch门槛全部不变，optimizer_steps仍0。

## 2026-09-17 13:10 常规推进；AVIF评分传输修复已备好，未热切换

- 实时collector仍CODE41/v6，2377/3200完整、0normal_error/0pending_infrastructure；四副本健康。
  prefetch仍CODE42/v2-auto-retry，1189有效、28pending、16worker，429共享冷却已触发，
  exhausted_transport=0。有效旧缓存重新绑定进度不是全局已评分总数，不能说旧评分丢失。
  13:02存储admission_open=true、当前及恢复祖先按inode去重约111.8GiB；之后独立du复测成功且stderr空，
  先前CalledProcessError具体原因未复现。collector/reviewer/compactor/guard receipt精确身份通过。
- 朋友Pro继续原串行：13:02为159成功/42失败/235未尝试，未中断；Flash已完成不重跑。
- 单独修复评分AVIF传输，不改原始图片、Agent或活跃快照。新增`psd_judge_media.py`，
  仅单帧RGB/RGBA AVIF转PNG，逐像素、尺寸、通道、EXIF与ICC往返校验，原sha用于任务/轨迹身份，
  PNG wire sha及原sha映射进入绑定的media包；不缩图、不有损重压、不丢alpha/动画。
  原JPEG/PNG/WEBP/GIF字节、media结构与缓存key完全不变；未知格式/多帧AVIF仍拒绝。
  依据[Gemini官方图片输入文档](https://ai.google.dev/gemini-api/docs/image-understanding)：PNG受支持，
  AVIF不在其内联支持列表，不能只改MIME字符串冒充支持。
- 代码`90373e6`已本地提交push。候选CODE43=`/volume/ybo/wza/training-artifacts/psd-avif-transport-20260917-v43/code`，
  79本地定向/500服务器PSD回归通过，Agent逐字未变；父目录有stage-state.json及code-binding.json。
  `real-media-canary.json`证明原先8条AVIF轨迹全部完整组包成功、原文件不变；另外一条既有有效评分
  在禁网client下逐字复用成功，0provider调用。**尚未做真实Gemini转码请求验收，不宣称8条评分补齐。**
- 不中断CODE42在途评分、不为这一项再次重启collector。后续在16路预算允许的安全窗口，先用CODE43
  在独立可复用cache对上述固定AVIF轨迹做真实完整source judge，不添加视觉简化probe、不反复抽label。
  通过后正式prepare/repair/raw teacher/最终bank DP4可使用CODE43，deployment为
  `psd-avif-transport-20260917-v43`，仍显式传v6/source-review-prefetch-v2-auto-retry与最终ready。
  切换前必须核当前输出/缓存绑定与进程，不能照历史段落重启旧reviewer。21条非逐字证据问题仍需
  逐条诊断，不能放宽或绕过；本次AVIF改动不解决这类证据错误。正式优化step仍0。
- 已通过自动化工具刷新同一15分钟heartbeat的说明，清除过时v4/v5启动指令；没有新建重复任务，
  不改变安静监控意图。这次按OpenAI Docs查阅官方定时任务说明后更新，频率/目标任务保持不变。

## 2026-09-17 12:45 自动补评分已运行，实际40槽采集恢复（当前入口）

- 用户要求失败评分自动重跑，并追问为什么GPU只有guard在计算。真实原因是CODE41恢复采集时
  `import_prior_slots`串行解压、哈希、完整性核验全部旧结果，之后才导入并启动首批；不是GPU故障，
  也不是等待评分。12:33左右完成恢复：2296旧槽中2289完整结果原样复用、7不完整槽待补，
  原预算与4个历史allowance继承，0新增额度；所有旧原文保留。不要再重启collector重复这段核验。
- 当前collector仍CODE41/v6，receipt `v6/process.json`初始PID1318931；12:41 first40 gate已passed，
  已进入collecting_remaining_3160，完整2292/3200、normal_errors=0、pending_infrastructure=0，
  其中2289旧结果复用、3条真实补采成功。之后短暂被storage measurement CalledProcessError挡住，
  12:44:38自动成功重测、admission_open=true；12:45:29逐槽确认running=40，非仅配置值，
  gateway四副本各9在途模型请求，GPU利用率100/99/99/92%。不是只剩guard，未绕过存储门禁。
  临时测量错误具体stderr仍在只读追查，不能据此说个人quota已知或存储曾真的满。
  CODE41父目录原`handoff-process.json`/`completion-policy-handoff.json`负责接compactor，勿双开。
- 新CODE42=`/volume/ybo/wza/training-artifacts/psd-review-autoretry-20260917-v42/code`，
  本地提交`cbcf5ba`已push；43项本地定向测试、490项服务器PSD回归通过，`src/**/*.py`相对CODE41
  全部逐字相同。**未重启collector/vLLM/guard，未改变Agent或评分prompt/schema。**
- 新助手CODE42父目录`handoff-process.json`（初始PID1321562），入口
  `scripts/server/advance_psd_source_review_retries.py`，状态`review-handoff.json`、日志`handoff.log`。
  当前`v6_reviewer_autoretry_running`：已等v5评分active=queued=0且waiting、v6恢复证明存在、
  v5 collector已退出后安全完成切换。新评分PID1322009，12:45实际progress retry_errors=true，
  retrying=7、exhausted_transport=0。助手已完成，不要重启或手工双开评分。
- 唯一新输出为`v6/source-review-prefetch-v2-auto-retry`，receipt仍`v6/source-review-prefetch-process.json`，
  日志`v6/source-review-prefetch-v2-auto-retry.log`；精确复用v5/source-review-prefetch-v1缓存，
  Gemini3.1ProPreview/high/out8192、16worker及HTTPgate16不变。
- 自动补HTTP429/500/502/503/504、网络超时/传输错误与未完成响应；每条最多4次外层评分尝试，
  每次内部仍max_retries2。旧尝试计入总预算，先落盘计费attempt再发请求；60秒指数退避+jitter，
  429共享冷却；重启不清预算。有效pass/fail/unresolved不重评；改变后的新轨迹按新hash评分。
  通用ValueError/已缓存的非法模型决策不盲重抽，pending_error需诊断，不能冒充政策fail或跳过gate。
  新progress含retry_errors、retrying、exhausted_transport、cooldown_until；active含退避worker，
  不能据此说始终有16个HTTP请求。
- 12:40对29个ValueError做禁网缓存重放：21条为一次允许纠正后仍非逐字证据引用，8条为
  unsupported PSD judge image type；0次外部调用。它们不是429，需分别解决评分证据契约和媒体
  传输支持，不能宣称普通网络重试已修好。抽查unsupported的main-05936：原任务图AVIF/RGB/1000x760，
  归档policy请求用JPEG；需保证原图身份/内容和评分transport对应，不能删图假装评分成功。
  旧评分最终2296条中1288fail/569pass/18unresolved/421pending_error；新进度重建绑定时从0计数，
  **旧有效评分没有丢弃、不会重复付费**，仍逐条读精确缓存并验hash。
- 后续正式prepare/repair/teacher/DP4统一用CODE42，prefetch参数显式指向上述v2输出，
  不再照旧段落启动v6/source-review-prefetch-v1。采集3200完整、评分排空、所有正式gate仍不可跳过；
  最终bank四卡global32短步及恢复验收、5epoch不变。训练optimizer_steps仍0。

## 2026-09-17 11:44 用户要求不完整轨迹也自动重跑，CODE41有序接续已启动

- 用户明确扩大恢复范围：没有完整可用轨迹就补跑，不能只覆盖NaN/超时。新模块
  `psd_source_completion.py`检查terminal success、完整最终报告schema/最终输出回合、完整有限token capture；
  不读取gold/judge、不以答案正确率选择。完整但答错、正常完成调查中保留的失败/空工具结果不重抽。
  无最终报告、异常终止、不可用工具动作引起的中断等会记`trajectory_failed`，总预算仍3次，
  旧attempt照记、失败原文保留；耗尽仍pending而非伪装完整，不擅自新增额度。
- 对检查时2235条已完成源轨迹全量检查：2228完整可用、7条中断；包含用户指出的两条。
  其余5条为main-02757 r001、main-07325 r003、main-07778 r002、main-08866 r002、main-09228 r002。
  完整清单/原trace哈希在v40/source-completion-inventory.json，CODE41目录有绑定引用。
  v40仅未上线测试候选，**不要启动v40控制器**；它与v41的completeness validator字节一致。
- CODE41=`/volume/ybo/wza/training-artifacts/psd-complete-source-20260917-v41/code`；
  本地代码`1eadc91`、`0a57a1b`已push，142本地定向测试、573服务器回归通过，冻结Agent逐文件不变。
  v38旧compactor曾因prefetch持有completed-slot锁而以BlockingIOError退出（未损坏内容）。
  v41改为跳过busy槽、下一轮再处理，有冲突/释放后继续压缩的回归测试。不要复活未修复的旧helper。
- 自动排空助手已启动：CODE41父目录`drain-process.json`（初始PID1316620仅线索），
  入口`drain_psd_v5_for_source_completion.py`，进度`drain-progress.json`。只暂停精确owned du
  子进程阻止新槽，已发出的Agent照常完成；0在途且ledger/canonical/progress一致后才停止旧v5。
  所有模型副本与guard不动，失败会恢复被暂停的测量，不杀在途Agent。
- 自动接续助手也已启动：CODE41父目录`handoff-process.json`（初始1316621仅线索），
  `completion-policy-handoff.json`与`handoff.log`；等待排空证明`source-drain.json`，
  跳过已退出的v38 compactor，以CODE41启动新RUN=`/volume/ybo/wza/runs/psd-production400x8-20260917-v6`
  并`--reuse-run ...v5`，重新核验全部旧缓存；只把不完整槽标为待补，完整结果硬链接原样复用。
  已失败1次的槽从attempt-002续预算，不覆盖旧run。助手活动期间勿重复launch/启动compactor。
  新first40通过后自动接CODE41 compactor，并核新采样进展。**当前是有序切换，不能称7条已重跑完成。**
- 原16路评分暂继续处理v5已完成轨迹，不杀在途API。v5排空后，当原prefetch进程
  `active=0, queued=0`且进入waiting_for_completed_sources，按精确receipt停止它。
  随后用CODE41在v6启动新的prefetch：原benchmark/train/gold/model不变，`--concurrency16 --follow`
  `--output v6/source-review-prefetch-v1 --cache-source v5/source-review-prefetch-v1`，
  trusted helper环境CODE41/GATEWAY19025，CUDA_VISIBLE_DEVICES为空，显式HTTP gate16；
  receipt存v6/source-review-prefetch-process.json，独占新日志。不能让两个prefetch同时使用共享cache。
  新入口验证恢复祖先链、checkpoint/输入/评分配置，按精确请求缓存复用旧完整评分；重跑后变了的轨迹
  自然产生新key，绝不把旧残缺轨迹的unresolved挂到新轨迹上，也不删除旧评分。
- 后续全部3200完整后，CODE41正常prepare并传v6的`--source-review-prefetch`，16路评分排空后
  进入原slate/preservation/raw teacher/global32最终bank验收/5epoch；DP4 deployment改为
  `psd-complete-source-20260917-v41`并显式传本轮ready。旧SFT/fullstate只读保护，不重交旧评测judge。
- 存储核查：11:38 PSD当前run及五级恢复祖先按inode去重约104.4GiB（不含SFT权重），
  当时2245条canonical的逻辑大小44.09GiB，中位样本18.89MiB、p90样本31.99MiB；
  metadata含大量逐token概率及重复policy_input，还有原生请求归档、图片和重试缓存。
  压缩历史93轮累计估计回收82.24GiB；这是无损压缩/共享副本，不是删除轨迹、图片或权重。
  个人quota仍未知，不拿共享49TB free冒充个人可用空间，不恢复任意128GiB门槛。

## 2026-09-17 11:15 外部 source checker 16路与采集并行（当前最高优先级）

- 用户批准16路外部评分。CODE39=`/volume/ybo/wza/training-artifacts/psd-source-review-prefetch-20260917-v39/code`，
  代码`04c17ff`已从本地push；38项本地定向测试、561项服务器回归通过，冻结Agent源码逐文件不变。
  当前采集RUN仍为`/volume/ybo/wza/runs/psd-production400x8-20260917-v5`；collector及compactor
  **继续使用CODE38，不重启、不热换**。旧drain/handoff已经完成，禁止按下文历史段落重启。
- 新入口`CODE39/scripts/prefetch_psd_source_reviews.py`只读取durable completed source槽：核验
  seed、训练成员、policy/snapshot、task image、ledger/result/canonical一致性，使用同一原有
  `judge_source`完整轨迹/图片/transport-ID投影。不会宣布采集完成、提前筛修复任务或开训。
  Gemini模型`gemini-3.1-pro-preview`、high/out8192未变；16 worker和显式HTTP gate16，
  timeout240/max_retries2；不是仅修改外层并发却仍卡在客户端默认4路。
- 首批16已真实完成（约127秒）：14 fail、2 unresolved、0 transport/schema error；这是前两题
  的8次采样，不代表全体质量比例。两个unresolved已核实均为原轨迹termination=error、确实无最终报告，
  **不是漏传材料，不可当pass或改标签**；留待正式source-resolution处理。
  4条真实轨迹的缓存复用验证（含unresolved）在禁止网络的客户端下逐字一致、0新provider调用，
  证明`RUN/source-review-prefetch-v1/cache-reuse-canary.json`。
- 持续评分已启动，独立receipt=`RUN/source-review-prefetch-process.json`（初始PID1313480仅线索），
  日志`RUN/source-review-prefetch-follow.log`；状态`RUN/source-review-prefetch-v1/progress.json`，
  `--follow --concurrency16`，CUDA_VISIBLE_DEVICES为空。先核精确cmdline/pgid，不双开。
  首批canary receipt另存`RUN/source-review-prefetch-canary-process.json`（已退出），不要重启。
- `RUN/source-review-prefetch-v1`为私有缓存：按输入/模型/prompt/schema/全部图片精确hash复用
  Gemini响应，包括无效已完成响应；不按标签反复评分直到通过。records的pending_error和unresolved
  与普通policy fail分开，缓存/缺口不得擅自删掉重抽。worker存储异常会终止而非卡死queue.join。
  source采集发生intervention时停止新评分调度、排空已排队工作；不影响source进程本身。
- 后续实跑出现少量HTTP429（客户端内部2次重试已用完）及响应解析错误，独立pending_error，
  未伪装成policy fail。持续监控限流/吞吐，最终正式review会尝试补未缓存成功的请求，不无限付费重抽。
  用户追问两条无报告的重跑情况：`main-02754--d5fdfd7f--r003/r006`均只有1次attempt，
  原生异常为`finish_reason=tool_calls, content_chars=0`，reasoning_chars分别1/332。
  当前规则把它们识别为`unusable_policy_output_with_captured_prefix`而非基础设施异常，**没有自动重跑**。
  仅凭此错误无法区分模型生成格式与服务解析问题，不能宣称已证明是模型自身错误；需原始响应证据。
  也不能把pending的source review说成已进入实际修复执行；正式repair尚未开始。
- 11:10源轨迹已2117/3200，四卡40路继续；不是开始optimizer。全部3200后先等prefetch排空并退出，
  用**CODE39**正常`run_psd_round.py prepare`，原参数不变，额外显式传
  `--source-review-prefetch RUN/source-review-prefetch-v1 --source-review-concurrency16`。
  完整采集/全部成员校验照常；正式review重新生成完整bank绑定，复用相同请求缓存，避免再付一轮费用。
  禁止绕过pending gate、伪造completed manifest或只拿已评分子集开训。
- 此后的slate/preservation/raw teacher和最终bank DP4使用CODE39；DP4参数deployment为
  `psd-source-review-prefetch-20260917-v39`，仍必须显式传本轮最终`--ready`。
  原三epoch SFT及完整state保留、repair每题最多6次/12提案、最终global32验收及5epoch均不变。
  修复实耗取决于入选任务数及成功前尝试次数，最多400任务×6，并非3200×6；不能用source吞吐
  直接承诺修复ETA。16路评分与40路GPU采集重叠仅减少可并行等待，不降低质量/预算门槛。

## 2026-09-17 10:17 v5 接续完成，无人为容量上限的新采样已验证

- 自动接续已实际完成，`v38/storage-policy-handoff.json`为
  `verified_new_sampling_without_run_cap`；一次性drain/handoff助手正常结束，不再启动它们。
- 当前RUN仍是`/volume/ybo/wza/runs/psd-production400x8-20260917-v5`、CODE38不变。
  旧1860槽全部原样复用、0新增重试额度、4个既有allowance继承，first40再次passed。
  已完成1898/3200，其中38条为v5真实新采样，40在途、7历史普通错误、0待解infra。
  截至本次，v5新增基础设施失败为0；并非保证未来没有NaN/超时。
- `binding.storage_ceiling_bytes`和最新`storage.ceiling_bytes`均为null，
  `admission_open=true`、`measurement_status=complete`。当前run及全部恢复祖先去重后
  约87.5GiB；仍测量占用和shared free16GiB保护，**不得恢复128GiB或另一任意run上限**。
- 当前collector receipt为RUN/process.json（初始1301181），compactor receipt为
  RUN/storage-compaction-process.json（初始1302668），已核精确身份活动，不双开。
  新compactor实际处理25个v5 native槽，该轮无损回收831590748 bytes；旧图片/轨迹/权重不动。
- 新近4份v5原生轨迹strict audit全部通过：64个完整有限capture、57次成功工具调用。
  三epoch export/fullstate保护核验通过；四卡单次利用率100/100/100/87%，守护仍在。
- 后续继续v5的3200全槽采集，再CODE38完成source checker/slate修复/保留目标/raw教师评分，
  最终bank四卡global32与原生恢复验收后5epoch。**现在仍是采集，optimizer 0步**。
- 朋友Pro正常继续141成功、28失败、267未尝试。旧judge不重交。下文“仍在排空/等待复用”
  均为历史状态，不能按旧段落重复启动任何助手。

## 2026-09-17 09:53 自动排空完成，v5 正在校验复用1860槽

- 排空助手已自动完成：v4的1860槽全部completed、0在途，canonical/progress一致，
  `v38/source-drain.json`保存冻结后二次验证。旧collector及旧compactor均已退出，无在途轨迹被杀。
  最后1槽在原预算第2次完成，普通模型错误共7条仍原样保留，不按正确性重采。
- 接续助手已自动启动v5，当前RUN=`/volume/ybo/wza/runs/psd-production400x8-20260917-v5`；
  `RUN/process.json`初始PID1301181仅线索。CODE38不变。binding的`storage_ceiling_bytes=null`。
- 当前仍在逐条读取、校验1860份旧结果，**尚未恢复新的采样**，不把进程启动当作接续完成。
  09:53进程RSS约0.31–0.45GiB；3秒内累计读取增加约200MB，持续工作，不是卡死。
  旧数据全部预校验通过后才发布v5 ledger/slot-recovery；此时ledger为空是预期。
- `handoff-process.json`对应助手仍活动，`storage-policy-handoff.json`阶段为
  `validating_and_reusing_completed_sources`。它会自动等复用/首40通过后接新compactor，
  再验证`storage.ceiling_bytes=null`及真实新采样。**不要手动重复启动v5或compactor。**
- 最新2份旧源轨迹strict audit通过，30个完整有限capture、28次成功工具调用；三epoch
  export/fullstate保护核验通过。四个模型副本及实际计算guard保持活动，空档有成功计算脉冲。
- 朋友Pro继续135成功、28失败、273未尝试；旧judge不重复提交。PSD optimizer仍0步。

## 2026-09-17 用户取消人为 128GiB 门槛：v38 安全接续中（优先于下文）

- 用户明确要求移除该人为采集上限；`STORAGE_CEILING=None`，不能再恢复128GiB或改成另一个
  任意run上限。保留占用统计及shared free低于16GiB保护，个人GPFS quota仍未知。
- 本地代码`0da3ae3`已push，44项定向测试及新不可变CODE38的549项服务器回归通过，
  冻结Agent未改。CODE38=`/volume/ybo/wza/training-artifacts/psd-uncapped-storage-20260917-v38/code`。
- v4暂仍活动；一次性排空助手receipt为CODE38父目录`drain-process.json`，初始PID1299053
  仅线索，进度`drain-progress.json`。旧控制器没有drain接口，因此只暂停其精确owned `du`
  子进程，利用原准入重测等待阻止新槽，已派发Agent继续；绝不杀在途Agent/扩大重试预算。
  只有ledger全部completed、canonical和progress一致、冻结后再核验才停止旧collector；
  成功证据`source-drain.json`。测量TimeoutExpired在此维护窗口是预期，不是新的磁盘故障。
  若助手失败，其finally会恢复精确owned测量子进程；诊断后再做接续，不能盲目重采。
- 排空后在v4 compactor的300秒sleep窗口按receipt精确停止，避免恢复校验时改缓存。
  **接续助手已启动**：CODE38父目录`handoff-process.json`（初始PID1299396仅线索），
  入口`finish_psd_uncapped_handoff.py`，阶段文件`storage-policy-handoff.json`、日志`handoff.log`。
  它等待排空证据后自动停旧compactor、启动v5、等复用验证/首40通过、接新compactor并验证
  无上限的新采样。助手活动时不要手动执行下面命令或双开compactor；若失败先读其阶段和日志。
  代码`4b70c8b`已本地push；它不杀在途轨迹、不重启模型服务/guard、不删数据。
  下一RUN=`/volume/ybo/wza/runs/psd-production400x8-20260917-v5`，从v4原样复用，
  CODE38父目录`run_psd_production_collection.py launch --run-name psd-production400x8-20260917-v5
  --code-directory CODE38 --reuse-run /volume/ybo/wza/runs/psd-production400x8-20260917-v4`。
  不修改旧run、权重或模型服务；保留原样本/seed/预算/所有失败记录。已有目标RUN时先检查，勿双开。
- 待v5 import和first40通过、恢复新采样后，以CODE38启动其独占compactor；v5 scope严格绑定v4。
  后续checker/repair/raw教师/final-bank DP4及5epoch均使用v5与CODE38，DP4 deployment改为
  `psd-uncapped-storage-20260917-v38`，仍必须传本轮最终ready。当前未开始optimizer。
- 下文所有128GiB规定都是已撤销的历史记录，旧v37/v4接续指令不能覆盖此节。

## 2026-09-17 08:08 v4 已恢复新采样，压缩助手已接上

- 原1538槽全部通过恢复校验并原样复用；`slot-recovery.json`记录1538 reused、0新增预算、
  4个历史allowance继承，首40在线门槛再次通过。v4仍是当前RUN，CODE37仍是当前代码。
- 已实际产出2条**新生成**轨迹，合计1540/3200、40在途、6普通错误、0待解infra。
  两条新轨迹strict audit均通过，9个完整有限native capture、7次成功工具调用、0warning；
  原三epoch export/fullstate保护检查通过。四卡恢复模型请求，08:06单次利用率均100%。
- v4压缩助手已启动并完成首轮，receipt=`RUN/storage-compaction-process.json`，
  初始PID1289349仅线索；日志`storage-compaction-v37-controller.log`。不要重复启动。
  首轮处理1个新完成槽，无损回收8,839,712 bytes；原1538复用槽/native旧目录不重处理。
- 空间准入已实测`measurement_status=complete`、`admission_open=true`，约71.9GiB，
  这是v4及所有恢复祖先合并去重后的口径，不能与旧v3单目录66GiB当作新增占用比较。
  修复没有抬高128GiB限额，继续关注增长。仅采集恢复，source checker/repair/optimizer仍未开始。
- 朋友Pro最近107成功、27失败、302未尝试，原串行队列继续；旧judge不重交。

## 2026-09-17 07:50 慢磁盘扫描修复与 v4 接续（当前权威入口）

- v3在1538条已完成、0在途时安全停止，6条普通模型错误照常保留；没有重采成功样本。
  此前`du`独立实测48.387秒，超过控制器45秒超时，空间状态反复过期、GPU请求队列出现空档。
  `gather(return_exceptions=True)`还会把准入测量异常变成未采集槽，不能放着等最后补齐。
  这不是新增NaN、不是磁盘已满。停机前短暂冻结准确owned进程组再次核对ledger/进度/
  canonical数量一致后TERM；证据在v36部署目录`source-drain.json`。原轨迹、尝试次数未改。
- **当前RUN为`/volume/ybo/wza/runs/psd-production400x8-20260917-v4`**，不要复活v3。
  新控制器receipt=`RUN/process.json`，初始PID1287468仅线索；原1538槽在逐条预校验，
  全部通过后才发布新ledger/canonical硬链接与`slot-recovery.json`，再复用首40验收并接余下槽。
  07:50仍是预校验、尚未恢复新模型采样，尚无优化训练；单进程约0.25–0.33GiB RSS，
  持续读取原压缩缓存，不能把暂时没有目标ledger当死锁，也不能把PID启动称为恢复完成。
- **CODE37=`/volume/ybo/wza/training-artifacts/psd-recovery-compactor-20260917-v37/code`**，
  545项服务器回归、82项本地定向测试通过，冻结Agent仍未变。活动controller在该code父目录；
  修复提交`4eaf96e`、`e25867e`、`23fe94e`已本地push。v35/v36只作中间验证候选，不运行。
  原CODE34的最终bank DP4门槛、v33历史5次预算兼容和gzip读者均完整继承。
- 空间准入扫描改180秒，失败先关闭准入并10秒后重测，不跳过槽、也不花模型尝试额度；
  真超空间阈值则60秒重查，不假装完成。128GiB/共享free16GiB门槛不变，扫描范围包含
  当前run与`binding.reuse_run`完整祖先链，一次`du -c`按inode去重，**换目录不绕过限额**。
  shared free仍不是用户配额，个人配额未知。完整3200后可能仍接近/达到限额，继续评估。
- 恢复改为逐个验证完整缓存，仅保留小结果记录，避免把1538个多模态payload一起放内存。
  同版本的原result/gzip/canonical采用硬链接，字节与原路径保留、不展开缓存；原失败次数、
  四个历史max5 allowance原样继承，无新预算。校验前后核SHA，复用槽不再调用模型或重写trace。
- 原v32 compactor已在休眠窗口停止，并对v3做完最后一次压缩：1502个本地生成槽，
  另外36个更早复用槽不触碰。旧receipt/PID1234814仅历史，不能复活或并发启动。
  **v4 compactor尚待启动**：等v4进入`collecting_remaining_3160`且import完成后，用CODE37
  的`compact_psd_completed_storage.py --run RUN --reader-snapshot CODE37 --follow`，
  PYTHONPATH=CODE37:CODE37/training，owner.spawn新独占日志，存`RUN/storage-compaction-process.json`。
  新helper只允许v3及binding严格指向v3的当前v4；不处理旧run的native文件。
- 四个vLLM服务与原计算guard未重启，空档有真实计算脉冲；不得用显存占用冒充利用率。
  下一检查先核v4的import/首40/新采样实际产出，再启动上项compactor；不要重复launch。
  全部3200后用CODE37、v4 episodes/snapshot接source checker、slate修复、raw教师评分，
  最终真实bank的DP4 global32恢复验收通过后才启动5epoch优化。原SFT权重及全部旧run保留。

## 2026-09-17 04:03 最终bank的四卡验收入口已准备（尚未执行）

- 采集继续到693/3200、3普通错误、0待解基础设施槽；四卡、存储助手正常。
  v3截至03:59累计14次模型超时，未见新增NaN。作业约30.4GiB，128GiB准入保护未触发。
  这仍是源采集，正式训练0步；不因为夜间时间目标而绕过完整源/目标验收。
- 新备用代码`/volume/ybo/wza/training-artifacts/psd-final-bank-gate-20260917-v34/code`：
  531项服务器回归通过、冻结Agent未变，本地29项门槛/配置测试通过。
  继承v33的gzip及历史预算恢复支持。**没有切换活动collector/存储助手，也没有拿走GPU。**
- `scripts/server/run_psd_dp4_resume_gate.py`新增显式`--ready`：先通过正常`load_ready`
  校验最终bank哈希和datum，再从它选择datums、snapshot及19025当前网关，不再误用199-target
  canary的硬编码bank、旧snapshot及19019。仅接受ROOT内路径、epoch3原模型、无先前PSD adapter
  的本轮初始化，记录ready/datums/snapshot绑定；原生2步baseline/第1步恢复的证明机制不变。
  默认不传`--ready`仍是旧工程入口，**正式bank验收必须传，不能直接照抄旧launch命令**。
- 未来用法：v34的该脚本`launch --deployment psd-final-bank-gate-20260917-v34
  --output-name dp4-resume-gate-production-v1 --ready <最终本轮ready.json>`。
  此路径现在尚不存在，不运行。先完成3200源、checker、slate repair、preservation和raw教师评分；
  准备阶段的`--round-index`应为1（这是第一个PSD round，不是SFT epoch或0-based编号）。
  最终数据以DP4、每rank2、累积4的global32执行2步及原生恢复对照；通过后另起5epoch优化。
- 上述531测试只证明代码/输入边界；最终bank上的GPU loss/梯度/更新及恢复仍未实测。
  旧199-target单卡更新、旧masked bank四卡恢复都不替代这个待完成组合门槛。

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
