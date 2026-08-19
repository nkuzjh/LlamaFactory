# 实验进度与结果汇总

更新时间：2026-08-19 +08:00

本文档是 BrickNet-MM 三仓库**唯一人工维护的实验结果账本**，统一记录 LlamaFactory PT/SFT、BrickNet
Stage5–8 agentic inference/SFT 和 ms-swift GRPO。`validated/frozen` 结果可用于当前结论；`diagnostic` 只用于定位；
`invalidated` 只保留故障 provenance，不进入模型或系统排名。输出目录中的自动生成 `results.md` 是证据 artifact，
不是第二份结果账本。

具体训练、推理与评测命令仍分别保留在 [LlamaFactory record](record.md)、
[BrickNet Stage 6–7 评测说明](../BrickNet/BrickNet-MM%20Agentic%20LEGO%20Planner/Stage%206-7%20Agentic%20Evaluation.md)
和 [ms-swift record](../ms-swift/record.md) 中。

## 当前项目进度

- Stage 0–1 已完成；mixed PT-exp1 final 已作为 Stage2 的共同初始化，66,456 条 Control/Thinking-Hard 数据与
  processor gate 已通过。
- Stage2 的 `exp4`–`exp4_3` 以及 Lean-State V2 `exp4_3_1` 训练、VAL512 推理和统一评测均已完成。旧
  Thinking-Hard 未优于 NonThinking-Control；Lean-State 恢复部分退化但主指标仍未超过 `exp4_2`，两条 Thinking
  路线均不扩 50k/all。
- Stage3 v002 Pilot 机器标注已完成 252/252，人工审计仍未批准；Stage3 与 PT-exp2 保留为研究支线，不是当前
  action-only 主线的前置条件。PT-exp2 数据与配置已就绪，但 text8m/MM e1/e2/e3 均未训练；外部 PT-exp2-100k
  adapter 尚未同步到规范路径。
- Stage5 full-pool replay 已通过：66,456/66,456 reference、1,751,435 个 GT action、0 failure。
- Stage6–7 的模板/EOS/RNG/framing 修复仍有效。2026-08-18 定义审计发现旧 V1/V2
  raw first-choice 被 verifier rejection 擦除、V2/A1 的 DFS 不完整、A0/A1 首轮使用了替换 exp4_2 prompt 的
  `stage8-act`。实现已修复，并冻结 inference-contract v1；B1/V1/V2/A0/A1 五组等待 VAL512 重跑，旧五组结果不得引用。
  新协议保留被拒 candidate-0 文本后立即终止诊断前缀，不运行 shadow rollout；V2→A1 只在
  `final_system` 层做 paired 主比较。推理输出已改为无 resume 的逐样本三路 partial 流式持久化，正常结束以 audit-last
  提升正式 artifact；Stage6/Stage8 preflight 拒绝任何残留 partial。
- Stage8 的 R1-S protocol/window 实现和历史 64-source 切窗测试已验证；当前重新绑定 Stage5 full report 的 smoke
  artifact 仍等待 supervised-token/token-mix audit，状态为 `training_eligible=false`。`stage8-act` 比较器已独立命名为
  `S8-ZS-Greedy` / `S8-ZS-DFS`，但尚无正式 VAL512 分数。下一步是刷新 token gate，再物化并
  训练 R1-S 10k。R1-C 等待 R1-S policy rejection，R1-B 仍等待至少 1,000 rollback transitions/100 sources。
- Stage9–10 尚未进入 gate；只有通过统一 VAL512 gate 的最佳 R1 checkpoint 才能进入 multi-turn GRPO。

## 结果状态与异常登记

| 对象 | 状态 | 是否进入当前比较 | 说明 |
| --- | --- | --- | --- |
| Stage6–7 contract-v1 B1/V1/V2/A0/A1 | `implementation + streaming fixed / five-run rerun pending` | 否 | 完整 inference contract、新诊断层、完整 DFS、逐样本持久化与 exact-reconstruction preflight 已实现；等待五组原路径覆盖重跑 |
| Stage6–7 2026-08-18 旧 B1 | `superseded by contract-v1` | 否 | 曾通过上一版 preflight，但 audit 未嵌入当前完整 contract；不能作为当前 B1 |
| Stage6–7 2026-08-18 旧 V1/V2/A0/A1 | `invalidated` | 否 | raw-gating、非完整 DFS、A0/A1 base-prompt 替换；旧 artifact 只保留故障 provenance |
| Stage6–7 2026-08-15 B1/V1/V2/A0/A1 | `invalidated` | 否 | native Qwen 空 think、错误 EOS；B1 另有 framing 空串写回。只保留故障 provenance |
| PT-exp0 固定 `"a"` 无条件推理 | `invalidated` | 否 | 与 MM-PT 的 image+inventory 训练协议不匹配，0% 不能解释为模型能力 |
| exp4_3_1 Lean-State | `validated/hold; trace warning` | 是 | path 指标齐全，但 strict trace-format 仅 38/512；无主指标优势，不推广或扩容 |
| Stage8 R1-S 64-source | `protocol validated / artifact blocked` | 仅协议证据 | 当前 artifact 等待 supervised-token/token-mix gate；不是正式训练结果 |

默认情况下异常数字留在历史记录中且不与有效实验混排；本轮 Stage6–7 按用户批准的临时例外直接覆盖本地五组
controller/eval artifact，账本仍保留事故说明，直到新结果写回本节。

## 统计口径

- 后续统一对比实验均在 512 条 `BrickNet-MM-VAL` 上调参、选择搜索预算并报告指标。已有 non-thinking 实验使用
  `max_new_tokens=4096`、`top_k=20`、`top_p=0.95`、`temperature=1.0`；阶段 2 Thinking-Hard 保持同一
  sampling，但为完整 trace 使用 `max_new_tokens=16384`。
- 该集合会被反复观察，因此结果属于 repeated-use VAL，不是未触碰的 locked test 或无偏泛化估计。
- `Train loss` 是整个训练过程的平均 next-token cross-entropy loss，不是最后一步 loss。
- GRPO 的 `Reward`、五个分量和权重统一见[统一指标词典](#统一指标词典)；训练表结果为 1,000 个训练更新步骤的日志均值。
- `exp0`、`exp1`、`exp1_1` 直接使用 BrickNet-MM-VAL 训练，只用于验证训练/推理链路和
  过拟合能力，不能代表泛化性能。
- 图文指标仅在成功解析、转换为 LDR 并完成八视图渲染的样本上计算；`Adj.` 指标将未成功
  渲染的样本按 0 分计入固定的 512 条分母。

## 统一指标词典

本节统一解释后文 `exp4_2 Agentic Stage 5–7 结果`、`GRPO 在线训练结果` 和
`Condition Generation 结果` 的同名指标。除非表格另有说明，验证集比例均以固定的 512 条样本为分母；百分数只是
把 `[0, 1]` 比例乘以 100 后展示。几个最容易混淆的字段先统一如下：

- `Parsable` 与 `Connectivity` 计算相同，前者只显示比例，后者同时显示成功条数和比例。
- `Parse` 与 `Parse Prefix` 计算相同，是“失败前完成了多少”的连续分数；它们不等于只看完整成功与否的
  `Parsable/Connectivity`。
- `Length` 与 `Length Score` 计算相同；`VQA` 与 `VQAScore`、`SigLIP2` 与 `SigLIP 2` 也只是表头写法不同。
- `Reward` 与 `Dense Reward` 使用同一加权公式，但 `Reward` 是在线训练期间模型临时生成结果的训练日志均值，
  `Dense Reward` 是固定验证集最终预测的评测均值，因此不能直接把两者当成同一批数据上的前后变化。

### 生成文本、结构安全与图文指标

| 英文名 | 中文对应与计算 | 含义、用途和限制 |
| --- | --- | --- |
| `Parsable` / `Connectivity (Num, %)` | 可完整读取率。预测从头到尾都能解析记为成功；`Connectivity` 展示成功数及其占 512 条的比例，`Parsable` 只展示比例。 | 表示文本格式和连接树完整，是最低可用性要求；不保证零件清单正确、无碰撞或形状正确。 |
| `Clean (Num, %)` / `Clean` | 无碰撞完整率。完整可解析且碰撞列表为空的样本数除以 512；带 `Num` 的表同时展示成功数。 | 表示基础结构安全；仍不要求与目标三维形状一致。 |
| `Collision` | 首次失败前平均放置数。单条样本取总动作数、首次解析失败位置和首次碰撞位置中的最小值，无失败时取总动作数，再对 512 条求平均。 | 越高表示通常能连续正确放置得更久。它不是碰撞次数，且受目标长度影响；不要与归一化的 `Collision Prefix` 混淆。 |
| `BLEU-4` | 四阶局部文字重合分。生成路径与参考路径按字符比较连续一至四个字符片段，使用 NLTK 的第 3 种平滑方法，逐样本计分、乘 100 后取平均。 | 反映局部文本模仿程度；文字相似不代表结构合法或形状正确。 |
| `ROUGE-1` / `ROUGE-2` / `ROUGE-L` | 顺序文字重合分。使用 jieba 分词后，分别比较单个词、连续两个词和最长保持顺序的公共部分，取兼顾准确与覆盖的分数，逐样本乘 100 后求平均。 | 用于观察生成顺序和参考文本的重合；同一结构可能有不同合法搭建顺序，因此只作辅助诊断。 |
| `PE` | 图文外观相似分。使用 `PE-Core-bigG-14-448` 把文字描述和八个渲染视角转成归一化数值特征并计算余弦相似度；每条样本取八视角最高分，再对成功渲染样本求平均。 | 衡量可见外观与描述是否接近；只覆盖可成功解析和渲染的子集。 |
| `SigLIP2` / `SigLIP 2` | 第二种图文语义相似分。使用 `google/siglip2-giant-opt-patch16-384`，把每个视角的图文匹配输出转换到 0 至 1；每条样本取八视角最高分，再对成功渲染样本求平均。 | 提供另一种图文语义证据；同样受成功渲染子集的选择影响。 |
| `VQA` / `VQAScore` | 问答式图文相符分。向 `Qwen/Qwen2-VL-7B-Instruct` 询问渲染结构是否符合描述，以第一个回答单位为 `Yes` 的概率计分；八视角取最高值后求平均。 | 只说明图像与描述是否看起来相符，不证明连接、碰撞或精确姿态正确。 |
| `PE Adj.` / `SigLIP 2 Adj.` / `VQA Adj.` | 覆盖率修正图文分。对应图文均值乘以成功渲染样本数，再除以 512，等价于把未成功渲染样本记为 0 分。 | 防止少量可渲染样本的高分掩盖低覆盖率；必须与未修正图文均值一起解释。 |

### 目标对齐与共享奖励分量

设预测和目标零件数分别为 `P/T`，按“零件种类与颜色”统计的重合零件数为 `O`。以下逐样本分数均在
`[0, 1]` 内，表中展示 512 条均值或成功比例。

| 英文名 | 中文对应与计算 | 含义、用途和限制 |
| --- | --- | --- |
| `Parse` / `Parse Prefix` | 可读取进度分。完整解析得 1；否则为已经成功解析的预测零件数除以目标零件数，上限为 1。 | 区分“一开始就失败”和“接近完成才失败”；它是连续完成度，不是完整可读取率。 |
| `Inventory F1` | 零件清单综合分，计算为 `2O/(P+T)`；零件种类、颜色和重复数量都参与比较。 | 同时惩罚多放、少放、用错零件和用错颜色；不检查连接与姿态。 |
| `Length` / `Length Score` | 长度分，计算为 `min(P,T)/max(P,T)`；预测数和目标数相同得 1，两者都为空时也定义为 1。 | 只衡量零件总数接近程度，不能说明使用了正确零件。 |
| `Collision Prefix` | 无碰撞进度分。没有碰撞得 1；否则为首次碰撞位置除以目标零件数，上限为 1。 | 是按目标长度归一化的安全进度。空结果也可能因“未检测到碰撞”得到 1，必须与可读取进度等其他分量共同看。 |
| `Pose Match` | 姿态匹配分。先消除预测与目标整体起点和方向的差异，再按相同零件种类和颜色配对；位置误差不超过 0.5、旋转误差不超过 5 度的目标零件数除以目标零件数。 | 衡量是否真正还原目标三维形状，只在有参考答案的评测或训练中使用，不能作为部署时检查程序偷看的信息。 |
| `Dense Reward` | 连续综合分：`0.20×Parse Prefix + 0.20×Inventory F1 + 0.10×Length Score + 0.20×Collision Prefix + 0.30×Pose Match`。 | 最终成功较少时用于比较整体进步和定位失败原因；不能代替 `Strict Success`。 |
| `Reward` | 在线训练总奖励，使用与 `Dense Reward` 相同的五项和权重；表中是 1,000 个训练更新步骤的日志均值。 | 统计的是训练时每题生成八个答案的表现，不是固定 VAL512 的最终质量。训练还会在每题内部比较八个答案的相对高低。 |
| `Strict Success` | 最终任务成功率。必须同时满足完整解析、预测与目标零件数相同、零件种类/颜色/数量完全一致、无碰撞且 `Pose Match=1`。 | 第一主指标，回答“是否完整还原了目标结构”；不能用文本相似分或仅合法的输出率替代。 |

### Stage 6–7 外层纠错机制与计算成本

| 英文名 | 中文对应与计算 | 含义、用途和限制 |
| --- | --- | --- |
| `Controller hard-valid success` | 外层程序合法完成数。外层程序返回成功，并经复核满足完整解析、零件清单和长度完全正确、无碰撞。 | 回答“是否得到一个合法、安全、用料正确的完整结构”，但不要求姿态与目标一致，因此只是安全上限，不能代替 `Strict Success`。 |
| `Recovery rate` | 恢复率：至少出现过一次拒绝、但最终仍合法完成的样本数，除以至少出现过一次拒绝的样本数。 | 衡量重试、分支尝试和回退救回了多少原本遇到错误的样本。 |
| `Mean expansions` | 平均展开数：每条样本实际取得的备选动作总数，再对 512 条求平均。 | 近似表示外层程序尝试了多少条路；越大通常表示计算成本越高。 |
| `Mean generated tokens` | 平均生成量：每条样本所有模型调用产生的基本文字单位总数，再对 512 条求平均。 | 比较不同重试和搜索方案消耗的生成预算。 |
| `Mean latency` | 平均墙钟用时：从单条样本开始到结束的实际经过时间，再对 512 条求平均。 | 反映用户等待时间；它包含模型生成和外层检查，不等于纯显卡计算时间，机器同时有其他任务时也会受影响。 |

### 成对差值与不确定范围

`A → B` 的差值统一计算为“B 减 A”。两种方法必须使用相同的 512 条样本和相同顺序，先逐样本相减，再用固定
随机起点 42 对这 512 个差值进行 10,000 次有放回重复抽取。

| 英文名 | 中文对应与计算 | 判读方式 |
| --- | --- | --- |
| `Final strict delta` | 系统最终层的 `Strict Success` 差值，以百分点展示。例如 `+1 pp` 表示成功率绝对提高 1 个百分点。 | 正数有利于后一个方法，负数有利于前一个方法。 |
| `Final dense delta` | 系统最终层的 `Dense Reward` 均值差，直接使用 `[0, 1]` 分数单位，不是百分点。 | 正数表示后一个方法的连续综合分更高。 |
| `95% CI` | 95%不确定范围：10,000 次重复抽取所得平均差值的第 2.5% 和第 97.5% 位置。 | 整段大于 0 才称稳定提升；整段小于 0 才称稳定下降；跨过 0 表示当前样本不足以证明稳定差异。 |
| `判定` | 根据差值方向、不确定范围是否跨 0，以及实验是否通过协议检查写出的文字结论。 | 协议失效的结果即使区间不跨 0，也只能保留为事故诊断，不能成为项目结论。 |

### 在线训练专用字段

| 英文名 | 中文对应与计算 | 含义、用途和限制 |
| --- | --- | --- |
| `Zero-std` | 零差异答案组比例。每道训练题生成八个答案；若八个答案的总奖励完全相同，该题组记为零差异，再对训练步骤求平均。 | 零差异组无法告诉模型哪个答案更好，比例过高说明奖励缺乏区分能力；越低通常越有利，但不能单独证明训练有效。 |
| `Loss` | 在线训练优化目标的日志均值，由题内相对奖励、概率变化限制以及与参考模型的偏离约束共同形成。 | 不是监督训练中的逐字预测误差，数值可接近 0 或短暂为负；不能用其绝对大小跨实验判断生成质量。 |
| `状态` | 实验是否完成、完成多少训练步骤，以及结果是否通过检查或仍被暂停。 | 不是质量分数；只有状态允许进入比较的结果才能支持结论。 |

### 指标文件来源与决策顺序

- `BLEU-4` 和 `ROUGE-*` 来自逐样本预测与参考文本。
- `Parsable/Connectivity`、`Clean` 和 `Collision` 来自 BrickNet 解析与真实网格碰撞结果。
- 三种图文分数来自成功渲染的八视角图像，修正值再把未渲染样本计为 0。
- `Parse Prefix` 到 `Strict Success` 来自逐样本预测与目标结构的对齐结果。
- 外层程序成功率、恢复率、展开数、生成量和用时来自逐样本执行记录。
- 在线训练的 `Reward`、五个分量、`Zero-std` 和 `Loss` 来自训练状态日志。

项目决策先看 `Strict Success` 及其 95%不确定范围，再看 `Dense Reward`，随后确认合法完整输出覆盖率，最后比较
展开数、生成量和用时。结构安全、图文和文字重合指标用于解释原因，不替代第一主指标。

## 实验配置与进度

已有 non-thinking BrickNet-MM 实验均使用 `Qwen/Qwen3.5-0.8B`、`qwen3_5_nothink`、
`cutoff_len=4096`、LoRA target `all`、LoRA rank/alpha `64/128`。PT/SFT 使用 batch
size 2、gradient accumulation 8、learning rate `5e-5` 和 cosine scheduler。

阶段 2 Thinking-Hard 是明确例外：用户于 2026-08-06 冻结 `qwen3_5_nothink`、`enable_thinking=false`、
`cutoff_len=16384`、`packing=false`、`train_on_prompt=false`，推理 `max_new_tokens=16384`。显式 `<think>` 是 assistant
监督文本，不叠加 Qwen 原生 thinking 模板。无思考对照统一命名为 `NonThinking-Control`。Stage 0 mixed PT-exp1
final 已完成；`exp4`–`exp4_3` 四个训练和原始 VAL512 的 512/512 推理均已完成，`exp4_2/exp4_3` 全指标已完成。
Lean-State V2 `exp4_3_1` 也已完成训练、strict extraction 和统一 VAL512 评测。
原 Stage 2 的 50k/all 继续暂停。新增 PT-exp2 性能支线使用独立
`PT-exp2-text8m/mm-e1/e2/e3` 命名；下游不做 VAL511，版本从 `exp4_4` 10k 继续递增到 `exp4_5` 50k、
`exp4_6` all，详情见 [PT-exp2 runbook](bricknet-pt-exp2.md)。
数据、配置、gate 和执行命令见
[Stage 2 runbook](bricknet-stage2-thinking-hard.md)。

| Exp | 框架 | Train output | 初始化 | 数据 | 样本数 | Epoch | 主要 ablation | Train loss | 状态 |
| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |
| PT-exp0 | LlamaFactory | `train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64` | Qwen3.5-0.8B | BrickNet-MM-PT | 135,051 | 3 | MM-PT：图像+inventory → path | 0.1507 | 完成并评测 |
| PT-exp1 | LlamaFactory | `train_PT_exp1_qwen35_08b_bricknet_text270k_mmpt135k_ep1_bs2_ga8_lora64` | Qwen3.5-0.8B | BrickNet text PT + BrickNet-MM-PT | 405,153 | 1 | 固定曝光预算的 text+MM mixed PT | 0.0683 | 完成；final 已作为 Stage 2 共同初始化 |
| PT-exp2-text8m | LlamaFactory | `train_PT_exp2_text8m_qwen35_08b_path7698261_steps250k_bs4_gbs32_lora64_len6401_nopack` | Qwen3.5-0.8B | first-round + cross-pool exact-dedup text path PT | 7,698,261 | 250k steps（约 1.03919 epoch） | 单/双卡自适应；BS4；GA8/4；path+EOS、6,401-token non-packed full-sequence PT、global batch 32 | - | 未训练；source/shard/VAL1000、全量 parse 和 train view 完成；10k collision 94/10,000 仅作 provenance；执行前人工确认所选卡空闲 |
| PT-exp2-mm-e1/e2/e3 | LlamaFactory | `train_PT_exp2_mm_{e1,e2,e3}_qwen35_08b_text8m_mm135k_replay1to1_ep1_bs2_gbs16_lora64_len6400` | PT-exp2-text8m → e1 → e2 adapter | all MM-PT + 三组不重叠 1:1 text replay | 150,668 / 150,637 / 150,718 | 各 1 epoch | 单/双卡自适应、BS2、GA8/4、global batch 16；三次顺序 multimodal consolidation；每轮后独立 VAL512 推理 | - | 三组数据与 processor zero-error/zero-truncation gate 完成；e1 等待 text8m adapter |
| exp2 | LlamaFactory | `train_exp2_qwen35_08b_sft1w_ep3_bs2_ga8_lora64` | Qwen3.5-0.8B | BrickNet-MM-SFT | 10,000 | 3 | 无 PT，小规模 SFT | 0.2418 | 完成并评测 |
| exp2_1 | LlamaFactory | `train_exp2_1_qwen35_08b_sft5w_ep3_bs2_ga8_lora64` | Qwen3.5-0.8B | BrickNet-MM-SFT | 50,000 | 3 | 无 PT，扩大 SFT 数据量 | 0.2031 | 完成并评测 |
| exp2_2 | LlamaFactory | `train_exp2_2_qwen35_08b_sft_ep3_bs2_ga8_lora64` | Qwen3.5-0.8B | BrickNet-MM-SFT | 334,355 | 3 | 无 PT，全量 SFT | - | 中断于 20,660/62,694；可恢复 `checkpoint-20000` |
| exp3 | LlamaFactory | `train_exp3_qwen35_08b_pt_sft1w_ep3_bs2_ga8_lora64` | PT-exp0 adapter | BrickNet-MM-SFT | 10,000 | 3 | PT 初始化后新建 SFT adapter | 0.1701 | 完成并评测 |
| exp3_0_1 | LlamaFactory | `train_exp3_0_1_qwen35_08b_pt_sft1w_ep10_bs2_ga8_lora64` | PT-exp0 adapter | BrickNet-MM-SFT | 10,000 | 10 | exp3 的 epoch ablation | 0.1090 | 完成并评测 |
| GRPO-exp0 | ms-swift | `../ms-swift/output/bricknet_grpo/exp0_qwen35_08b_exp3_rl_n2000_g8` | PT-exp0 merged + exp3 adapter | BrickNet-MM-RL | 2,000 | 1 | GRPO，五项结构/几何 reward，G=8 | - | 完成并评测 |
| exp3_1 | LlamaFactory | `train_exp3_1_qwen35_08b_pt_sft5w_ep3_bs2_ga8_lora64` | PT-exp0 adapter | BrickNet-MM-SFT | 50,000 | 3 | PT + 50k SFT | 0.1673 | 完成并评测 |
| exp3_2 | LlamaFactory | `train_exp3_2_qwen35_08b_pt_sft_ep3_bs2_ga8_lora64` | PT-exp0 adapter | BrickNet-MM-SFT | 334,355 | 3 | PT + 全量 SFT | - | 未开始 |
| exp4 | LlamaFactory | `train_exp4_qwen35_08b_mixedpt_stage2_nonthinking_control_val511_ep3_bs1_ga16_lora64_len16384` | mixed PT-exp1 final | Stage2 NonThinking-Control VAL511 | 511 | 3 | 无思考 overfit 链路检查 | 0.1738 | 训练完成；VAL512 512/512 推理完成 |
| exp4_1 | LlamaFactory | `train_exp4_1_qwen35_08b_mixedpt_stage2_thinking_hard_val511_ep3_bs1_ga16_lora64_len16384` | mixed PT-exp1 final | Stage2 Thinking-Hard VAL511 | 511 | 3 | Thinking-Hard overfit 链路检查 | 0.0860 | 训练完成；VAL512 512/512 推理完成 |
| exp4_2 | LlamaFactory | `train_exp4_2_qwen35_08b_mixedpt_stage2_nonthinking_control_10k_ep3_bs1_ga16_lora64_len16384` | mixed PT-exp1 final | Stage2 NonThinking-Control 10k | 10,000 | 3 | 无思考正式 paired 对照 | 0.1727 | 训练、512 推理和全指标完成 |
| exp4_3 | LlamaFactory | `train_exp4_3_qwen35_08b_mixedpt_stage2_thinking_hard_10k_ep3_bs1_ga16_lora64_len16384` | mixed PT-exp1 final | Stage2 Thinking-Hard 10k | 10,000 | 3 | Thinking-Hard 正式 paired 实验 | 0.0434 | 训练、512 推理、strict extraction 和全指标完成 |
| exp4_3_1 | LlamaFactory | `train_exp4_3_1_qwen35_08b_mixedpt_stage2_thinking_hard_v2_lean_state_10k_ep3_bs1_ga16_lora64_len16384` | mixed PT-exp1 final | Stage2 V2 Thinking-Hard Lean-State 10k | 10,000 | 3 | 移除 GT-next-action 泄漏的短 state-before 诊断 | 0.1083 | 训练、512 推理、strict extraction 和全指标完成；未超过 exp4_2，保持 hold |
| exp4_4 | LlamaFactory | `train_exp4_4_qwen35_08b_PT_exp2_stage2_nonthinking_control_10k_ep3_bs1_gbs16_lora64_len16384` | final `PT-exp2` alias | Stage2 NonThinking-Control 10k | 10,000 | 3 | 单/双卡自适应、BS1、GA16/8、global batch 16；新 PT 初始化的首个下游候选；无 VAL511 | - | dormant；等待 PT-exp2 alias |
| exp4_5 | LlamaFactory | `train_exp4_5_qwen35_08b_PT_exp2_stage2_nonthinking_control_50k_ep3_bs1_gbs16_lora64_len16384` | final `PT-exp2` alias | Stage2 NonThinking-Control 50k | 50,000 | 3 | 单/双卡自适应、global batch 16；exp4_4 收益 gate 后扩容 | - | dormant；未物化 50k |
| exp4_6 | LlamaFactory | `train_exp4_6_qwen35_08b_PT_exp2_stage2_nonthinking_control_all66456_ep3_bs1_gbs16_lora64_len16384` | final `PT-exp2` alias | Stage2 NonThinking-Control all | 66,456 | 3 | 单/双卡自适应、global batch 16；exp4_5 收益 gate 后扩容 | - | dormant |
| exp4_4_1 | LlamaFactory | `train_exp4_4_1_qwen35_08b_PT_exp2_100k_stage2_nonthinking_control_10k_ep3_bs1_gbs16_lora64_len16384` | external PT-exp2-100k | Stage2 NonThinking-Control 10k | 10,000 | 3 | 外部 100k PT 权重的独立下游 | - | blocked；规范 PT-exp2-100k adapter 尚未同步 |
| exp4_7_1 | LlamaFactory | `train_exp4_7_1_qwen35_08b_PT_exp2_100k_stage2_thinking_hard_v2_lean_state_10k_ep3_bs1_gbs16_lora64_len16384` | external PT-exp2-100k | Stage2 V2 Lean-State 10k | 10,000 | 3 | 外部 100k PT 权重 + Lean-State | - | blocked；规范 PT-exp2-100k adapter 尚未同步 |
| Stage8 R1-S | LlamaFactory | `train_stage8_r1_s_act_success_10k_ep3_bs1_ga16_lora64_len16384` | PT-exp1 + exp4_2 adapters | GT success-only Act trajectory | 10,000 sources | 3 | 逐 placement observation 协议 cold start；accepted-action boundary window | - | 历史 64-source protocol/window 测试通过；当前重绑 Stage5 的 artifact 因 token gate fail-closed，正式 10k 未物化/训练 |
| Stage8 R1-C | LlamaFactory | `train_stage8_r1_c_act_correction_10k_token_matched_lora64_len16384` | PT-exp1 + exp4_2 adapters，新 LoRA | supervised tokens 80% success + 20% real rejection/correction | 10,000 sources | token-matched | 与 R1-S supervised action-token budget 匹配 | - | blocked；等待 R1-S checkpoint、paired gate 和 policy rejection |
| Stage8 R1-B | LlamaFactory | `train_stage8_r1_b_act_rollback_10k_token_matched_lora64_len16384` | PT-exp1 + exp4_2 adapters，新 LoRA | supervised tokens 70% success + 20% correction + 10% rollback | 10,000 sources | token-matched | 只接受真实 successful rollback branch | - | blocked；等待 ≥1,000 rollback transitions / 100 sources |

PT-exp0 虽然命名为 PT，但在 LlamaFactory 中使用 `stage=sft` 和 BrickNet-MM-PT，属于
多模态监督预训练式训练，不等同于原始 BrickNet 使用固定 `"a"` prompt 的无条件
text-only PT。

Stage 2 的 paired 全指标已齐。`exp4_2` 为 parsable `382/512 (74.61%)`、clean `93/512 (18.16%)`、dense
reward `0.58159`、strict success `16/512 (3.12%)`；`exp4_3` 为 parsable/trace-valid `360/512 (70.31%)`、
clean `101/512 (19.73%)`、dense reward `0.57395`、strict success `13/512 (2.54%)`。Thinking-Hard 提高 clean
`+1.56 pp` 和 collision-prefix `+0.0133`，但降低 parsable `-4.30 pp`、dense reward `-0.00765`、strict success
`-0.59 pp`，三项图文指标也较低；当前没有总体优势，T1-10k 人工推广 gate 未批准。

T1 trace 诊断显示：512 条输出都生成了成对 `<think>/<action>`，511 条正常结束，但 152 条 extracted path 无效；
主要错误是 connector family 不存在、index 越界或 subtype 不可配对。模型在这些无效 action 上仍自报
`parse/inventory/collision=pass`，说明全正例文字检查未形成真实 verifier 能力。Thinking-Hard 平均输出约
2,741.7 tokens，而 Control 为 612.5；推理耗时约为 Control 的 4.81 倍，训练 token/FLOPs 约为 2.41 倍。
这些诊断用于解释失败机制，不改变上面的统一 evaluator 主结论。

Lean-State V2 `exp4_3_1` 为 parsable `374/512 (73.05%)`、clean `95/512 (18.55%)`、dense reward
`0.57829`、strict success `16/512 (3.12%)`。它相对旧 `exp4_3` 恢复 parsable、dense 和 strict，但相对
`exp4_2` 仍降低 parsable `-1.56 pp`、dense `-0.00331`，strict 完全持平；未做 paired bootstrap，因此只能判定
“未显示点估计优势”，不能声称等价或更好。另有 `512/512` 非空 path prefix，但只有 `38/512 (7.42%)`
通过 Lean-State 内部一致性 strict trace-format；主要错误为 action 0/1 的 state 与已生成 path 不一致。因此 path
评测本身有效，但该 reasoning schema 的合规性很差，是不推广此路线的独立证据。Stage2 V2 诊断完成后保持 hold。

`exp4_3_1` 关键证据 SHA-256：`generated_predictions.jsonl=6432c817…75b50`、
`path_predictions.jsonl=c194a616…cced8`、`trace_extraction_report.json=44227acb…a0d`、
`evaluation_manifest.json=a52fc339…78f2`、`metrics.json=812dd8c4…3431`、
`alignment.jsonl=ff1930c8…84f`、`path_text_metrics.json=8c143182…2166f`。训练产物为
`train_results.json=1004fd6f…17df`、`trainer_state.json=e4320dcd…d7c0`、
`adapter_config.json=d65c0207…0f55`、`adapter_model.safetensors=ec526af8…a3d4`。

## exp4_2 Agentic Stage 5–7 结果

Stage5 使用真实 mesh 对完整 66,456 reference 做事务 replay，结果为 `processed=passed=66,456`、
`total_actions=1,751,435`、`failures=0`、`stage5_replay_gate_passed=true`。Stage6–7 固定使用
`Qwen3.5-0.8B + PT-exp1 + exp4_2`，同一 512 VAL、seed 42 和冻结预算；没有重新训练模型。

> **当前状态：inference-contract v1 与五组实现已冻结；B1/V1/V2/A0/A1 均待重跑。** 旧 B1 缺少当前完整
> contract metadata，旧 V1/V2 另有 rejected raw 擦除与 DFS 不完整，旧 A0/A1 另有首轮 prompt 漂移。下表十行
> 全部只作事故诊断，不进入当前比较或项目决策。本轮按临时覆盖策略在原目录生成新 provenance，并同名重建结果表。

`final_system` 是 controller 最终可以安全交付的 hard-valid path，失败 episode 记为空串。模型诊断层按模式分为
`full_path_raw`（B1 完整单次生成）、`greedy_observed_candidate0_prefix`（V1/A0）和
`search_observed_candidate0_prefix`（V2/A1 主 DFS 实际访问状态）。candidate 0 是生成 API 返回顺序中的第一个，
不声称是全局最高概率；首次拒绝后保留该文本并终止诊断前缀，不在非法 prefix 上继续生成，也不运行
shadow rollout。V2→A1 只在 `final_system` 上配对，search-observed prefix 不作为完整 raw trajectory 的主比较。
系统选择以 pose-aware task strict success 为第一指标、dense reward 为第二指标，不能用 controller hard-valid success 代替
task strict success。

下表质量字段统一见[统一指标词典](#统一指标词典)：`Parsable` 对应 `Connectivity` 的比例部分，`VQA` 对应 `VQAScore`；
`Inventory F1`、`Pose Match`、`Dense Reward` 和 `Strict Success` 与 Condition Generation 表使用完全相同的评测口径。

| Mode | Layer | Parsable | Clean | PE | SigLIP2 | VQA | BLEU-4 | ROUGE-L | Inventory F1 | Pose Match | Dense Reward | Strict Success |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 post-hoc（superseded） | historical final | 18.95% | 18.95% | 0.305568 | 0.917509 | 0.819251 | 17.7410 | 12.7263 | 0.189453 | 0.084633 | 0.320116 | 15/512 (2.93%) |
| B1 post-hoc（superseded） | historical full_path_raw | 72.85% | 19.14% | 0.279856 | 0.801712 | 0.760395 | 90.4902 | 55.7241 | 0.900121 | 0.142103 | 0.584527 | 15/512 (2.93%) |
| V1 silent retry（invalidated） | legacy final | 28.32% | 28.32% | 0.296717 | 0.876359 | 0.794801 | 26.3474 | 17.7328 | 0.283203 | 0.091264 | 0.368981 | 13/512 (2.54%) |
| V1 silent retry（invalidated） | legacy raw | 16.60% | 16.60% | 0.310031 | 0.910616 | 0.828570 | 15.6142 | 10.9608 | 0.166016 | 0.076760 | 0.306036 | 13/512 (2.54%) |
| V2 silent DFS（invalidated） | legacy final | 69.34% | 69.34% | 0.282680 | 0.795736 | 0.765012 | 64.1347 | 39.3129 | 0.693359 | 0.134614 | 0.587064 | 16/512 (3.12%) |
| V2 silent DFS（invalidated） | legacy raw | 18.36% | 18.36% | 0.309605 | 0.900475 | 0.813474 | 17.2275 | 11.7962 | 0.183594 | 0.088955 | 0.318483 | 15/512 (2.93%) |
| A0 explicit feedback（invalidated） | legacy final | 0.20% | 0.20% | 0.239990 | 1.000000 | 0.761952 | 0.1713 | 0.1149 | 0.001953 | 0.000977 | 0.201270 | 0/512 |
| A0 explicit feedback（invalidated） | legacy raw | 0.00% | 0.00% | - | - | - | 0.0000 | 0.0000 | 0.000000 | 0.000000 | 0.200000 | 0/512 |
| A1 feedback search（invalidated） | legacy final | 5.27% | 5.27% | 0.307337 | 0.892875 | 0.774451 | 4.8555 | 3.2588 | 0.052734 | 0.019809 | 0.232310 | 3/512 (0.59%) |
| A1 feedback search（invalidated） | legacy raw | 0.20% | 0.20% | 0.288818 | 0.777344 | 0.464743 | 0.1923 | 0.1172 | 0.001953 | 0.001674 | 0.201479 | 0/512 |

旧 B1 是同协议、同 seed/RNG 生命周期的新随机运行，但未绑定当前完整 inference contract，因此只作历史诊断。
完全消除采样差异的 post-hoc 因果检查另由冻结 exp4_2 prediction replay 提供。`Parsable/Clean` 检查原始
prediction；task strict 在统一换行规范化后计算，两者输入边界不同，不能互相倒填。

Controller 机制与成本指标：

以下全部成本来自 superseded/invalidated artifact，只用于估计旧实现开销，不能与效果组成当前 cost-effectiveness 结论。
`Controller hard-valid success`、`Recovery rate`、`Mean expansions`、`Mean generated tokens` 和 `Mean latency`
的计算与解释统一见[Stage 6–7 外层纠错机制与计算成本](#stage-67-外层纠错机制与计算成本)。

| Mode | Controller hard-valid success | Recovery rate | Mean expansions | Mean generated tokens | Mean latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1（superseded） | 97/512 | 0.00% | 1.00 | 610.84 | 12.381 s |
| V1（invalidated） | 145/512 | 14.05% | 20.44 | 465.41 | 12.437 s |
| V2（invalidated） | 355/512 | 64.48% | 384.25 | 10,948.73 | 96.392 s |
| A0（invalidated） | 1/512 | 0.20% | 6.59 | 140.44 | 3.966 s |
| A1（invalidated） | 27/512 | 5.27% | 295.81 | 8,647.52 | 69.890 s |

所有 controller 成功输出经 evaluator 复核均为 100% hard-valid；`gpu_time_seconds` 未记录，因此当前只能报告
wall-clock latency，不能宣称 GPU-time 优势。

以下 paired bootstrap 使用同一 ordered VAL512，对逐样本 `candidate-baseline` 有放回抽样 10,000 次，seed=42；
由于 candidate artifact 已 invalidated，整表只保留历史 provenance，不是当前统计结论：

`Final strict delta`、`Final dense delta`、两列 `95% CI` 和 `判定` 统一见[成对差值与不确定范围](#成对差值与不确定范围)；箭头方向
`A → B` 始终表示差值按 `B-A` 计算。

| Comparison | Final strict delta | 95% CI | Final dense delta | 95% CI | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| B1 → V1 | -0.391 pp | [-1.562, +0.586] pp | +0.048864 | [+0.030827, +0.067106] | strict 未见稳定差异；V1 dense 显著提高 |
| V1 → A0 | -2.539 pp | [-3.906, -1.172] pp | -0.167711 | [-0.191202, -0.144270] | 未训练显式反馈显著退化 |
| V2 → A1 | -2.539 pp | [-4.102, -1.367] pp | -0.354754 | [-0.378715, -0.330678] | feedback search 显著劣于 silent DFS |
| B0/exp4_2 → V2 final | 0.000 pp | [-0.977, +1.172] pp | +0.005471 | [-0.015181, +0.025735] | strict/dense 均未证明稳定提升 |

当前没有可用于排名的 Stage6–7 五组结果。B1 的 post-hoc 覆盖、V1/V2/A0/A1 的相对效果、V2 是否作为安全
fallback，以及是否据此直接进入 R1-S 均保持未决；必须先完成五组 contract-v1 重跑和新 paired bootstrap。

冻结证据：

- Stage5 report：`../BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/stage5/Stage5-full-replay-report.json`，
  SHA-256=`89556b1c2d0ad754548984c3350388f57787c69618e6b6a60d30468455112ae4`。
- Stage6–7 contract-v1 config：`../BrickNet/configs/agentic_stage67_exp4_2.json`，SHA-256=
  `606dcd5f5a691ef6237f4ef391ae1d72be35963fecff8e5135654ca32caccf47`。
- 待覆盖证据：五组 controller 写入 `../BrickNet/outputs_val/qwen35_08b/agentic_exp4_2_{b1,v1,v2,a0,a1}/`；统一评测同名
  重建 `agentic_exp4_2_stage67_manifest.json`、`agentic_exp4_2_stage67_statistics.json` 和
  `agentic_exp4_2_stage67_results.md`。当前同名文件仍是旧内容，不能用其 hash/分数补位。

## GRPO 在线训练结果

本表的 `Parse/Inventory F1/Length/Collision Prefix/Pose Match` 分别对应统一词典中的
`Parse Prefix/Inventory F1/Length Score/Collision Prefix/Pose Match`；`Reward` 使用与 `Dense Reward` 相同的
加权公式，但统计的是在线训练生成结果。`Zero-std`、`Loss` 和 `状态` 见[在线训练专用字段](#在线训练专用字段)。

| Exp | Reward | Parse | Inventory F1 | Length | Collision Prefix | Pose Match | Zero-std | Loss | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| GRPO-exp0 | 0.451437 | 0.707373 | 0.757767 | 0.702507 | 0.325427 | 0.076910 | 1.20% | 5.732e-7 | 1000/1000，完成 |

训练耗时 6h13m42s，平均 completion 长度 1,028.85，平均 KL 0.001597，显存记录峰值
75.85 GiB。最终权重为
`../ms-swift/output/bricknet_grpo/exp0_qwen35_08b_exp3_rl_n2000_g8/checkpoint-1000`。

训练前已验证：2,000 条数据均无 assistant prompt，图片存在，reference 可解析且 part
数与 `target_part_count` 一致；reference completion 的五项 reward 和总 reward 均为
1.0。policy 与 reference 均正确加载 exp3 rank-64 adapter。

## Condition Generation 结果

BrickNet-MM 和 BrickNet-MM-RL 均在同一组 512 条 `BrickNet-MM-VAL` 上评测。除新增的
`Exp` 列外，`Train Data` 至 `VQAScore` 八列沿用 BrickNet condition generation 的展示
方式，其余指标追加在最右侧。BrickNet-MM
SFT 的对齐指标均由保留的逐样本预测按当前 GRPO verifier 重新计算。`MM-PT-135k`
使用匹配训练协议的 `VAL image + empty caption + inventory` 输入；其图文指标仍使用保留的
原始 VAL caption 评价渲染结果，因此该行可比较输出质量，但与带 caption 的 SFT 行输入不同。
`Exp` 记录对应的实验版本；论文公布的基线统一标记为 `BrickNet-paper`。

| Exp | Train Data | Model | Training / Sampling | Connectivity (Num, %) | Collision | PE | SigLIP 2 | VQAScore | Clean (Num, %) | BLEU-4 | ROUGE-1 | ROUGE-2 | ROUGE-L | PE Adj. | SigLIP 2 Adj. | VQA Adj. | Parse Prefix | Inventory F1 | Length Score | Collision Prefix | Pose Match | Dense Reward | Strict Success |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BrickNet-paper | BrickNet | Qwen3-0.6B-SFT | - | 466 (91.02%) | 10.6504 | 0.2643 | 0.7023 | 0.6848 | 151 (29.49%) | - | - | - | - | 0.2406 | 0.6392 | 0.6233 | - | - | - | - | - | - | - |
| BrickNet-paper | BrickNet | Qwen3-1.7B-SFT | - | 469 (91.60%) | 10.9980 | 0.2672 | 0.7243 | 0.6885 | 134 (26.17%) | - | - | - | - | 0.2448 | 0.6635 | 0.6307 | - | - | - | - | - | - | - |
| BrickNet-paper | BrickNet | Qwen3-4B-SFT | - | 481 (93.95%) | 10.2598 | 0.2707 | 0.7339 | 0.7179 | 146 (28.52%) | - | - | - | - | 0.2543 | 0.6895 | 0.6744 | - | - | - | - | - | - | - |
| BrickNet-paper | BrickNet | Qwen3-8B-SFT | - | 483 (94.34%) | 11.2227 | 0.2696 | 0.7309 | 0.7153 | 149 (29.10%) | - | - | - | - | 0.2543 | 0.6895 | 0.6748 | - | - | - | - | - | - | - |
| BrickNet-paper | BrickNet | Qwen3-14B-SFT | - | 479 (93.55%) | 11.3047 | 0.2712 | 0.7369 | 0.7183 | 151 (29.49%) | - | - | - | - | 0.2537 | 0.6894 | 0.6720 | - | - | - | - | - | - | - |
| PT-exp0 | MM-PT-135k | Qwen3.5-0.8B-PT | lr=5e-5, ep=3; eval p=.95, k=20, t=1 | 310 (60.55%) | 5.2188 | 0.2823 | 0.8007 | 0.7604 | 78 (15.23%) | 91.2174 | 95.3796 | 65.7656 | 55.4362 | 0.1709 | 0.4848 | 0.4604 | 0.7881 | 0.8253 | 0.7836 | 0.4596 | 0.1418 | 0.5355 | 14 (2.73%) |
| PT-exp1 | text path PT + MM-PT | Qwen3.5-0.8B-PT | lr=5e-5, ep=1; eval p=.95, k=20, t=1 | 351 (68.55%) | 5.9551 | 0.2807 | 0.7859 | 0.7462 | 86 (16.80%) | 88.9179 | 94.5496 | 64.9356 | 54.6818 | 0.1924 | 0.5388 | 0.5116 | 0.8411 | 0.8486 | 0.8211 | 0.4683 | 0.1198 | 0.5497 | 1 (0.20%) |
| exp2 | MM-SFT-10k | Qwen3.5-0.8B-SFT | lr=5e-5, ep=3; eval p=.95, k=20, t=1 | 237 (46.29%) | 4.4941 | 0.2814 | 0.7933 | 0.7570 | 68 (13.28%) | 90.2491 | 94.9980 | 65.6216 | 54.8258 | 0.1303 | 0.3672 | 0.3504 | 0.6671 | 0.7216 | 0.6612 | 0.4789 | 0.1233 | 0.4766 | 7 (1.37%) |
| exp2_1 | MM-SFT-50k | Qwen3.5-0.8B-SFT | lr=5e-5, ep=3; eval p=.95, k=20, t=1 | 354 (69.14%) | 5.6426 | 0.2824 | 0.7996 | 0.7522 | 85 (16.60%) | 91.6652 | 95.7064 | 66.2222 | 55.2598 | 0.1952 | 0.5529 | 0.5201 | 0.8413 | 0.8716 | 0.8372 | 0.4487 | 0.1497 | 0.5609 | 17 (3.32%) |
| exp3_1 | MM-PT + SFT-50k | Qwen3.5-0.8B-PT-SFT | lr=5e-5, ep=3; eval p=.95, k=20, t=1 | 367 (71.68%) | 6.0332 | 0.2854 | 0.8190 | 0.7666 | 93 (18.16%) | 92.1417 | 96.0075 | 66.6486 | 56.0046 | 0.2046 | 0.5870 | 0.5495 | 0.8641 | 0.8932 | 0.8609 | 0.4521 | 0.1617 | 0.5765 | 20 (3.91%) |
| exp3 | MM-PT + SFT-10k | Qwen3.5-0.8B-PT-SFT | lr=5e-5, ep=3; eval p=.95, k=20, t=1 | 340 (66.41%) | 5.6895 | 0.2830 | 0.8218 | 0.7662 | 85 (16.60%) | 91.5485 | 95.5846 | 66.0981 | 55.2628 | 0.1880 | 0.5457 | 0.5088 | 0.8289 | 0.8608 | 0.8247 | 0.4702 | 0.1501 | 0.5595 | 17 (3.32%) |
| exp3_0_1 | MM-PT + SFT-10k | Qwen3.5-0.8B-PT-SFT | lr=5e-5, ep=10; eval p=.95, k=20, t=1 | 267 (52.15%) | 5.1797 | 0.2833 | 0.8225 | 0.7657 | 75 (14.65%) | 91.3399 | 95.4301 | 66.2890 | 56.0601 | 0.1477 | 0.4289 | 0.3993 | 0.7335 | 0.7851 | 0.7299 | 0.4827 | 0.1463 | 0.5171 | 18 (3.52%) |
| GRPO-exp0 | MM-PT + SFT-10k + RL-2k | Qwen3.5-0.8B-PT-SFT-RL | GRPO lr=5e-6, ep=1, G=8, t_train=.9, p_train=1; eval p=.95, k=20, t=1 | 324 (63.28%) | 5.8086 | 0.2831 | 0.8182 | 0.7549 | 78 (15.23%) | 91.6202 | 95.6808 | 66.2536 | 55.7319 | 0.1791 | 0.5178 | 0.4777 | 0.8301 | 0.8625 | 0.8262 | 0.4630 | 0.1486 | 0.5583 | 14 (2.73%) |
| exp4_2 | mixed PT-exp1 + NonThinking-Control 10k | Qwen3.5-0.8B-PT-SFT | lr=5e-5, ep=3, len=16384; eval p=.95, k=20, t=1 | 382 (74.61%) | 6.2012 | 0.2804 | 0.7896 | 0.7552 | 93 (18.16%) | 90.6938 | 95.3249 | 66.1692 | 55.6116 | 0.2092 | 0.5891 | 0.5634 | 0.8850 | 0.8995 | 0.8739 | 0.4610 | 0.1504 | 0.5816 | 16 (3.12%) |
| exp4_3 | mixed PT-exp1 + Thinking-Hard 10k | Qwen3.5-0.8B-PT-SFT | lr=5e-5, ep=3, len=16384; eval extracted path p=.95, k=20, t=1 | 360 (70.31%) | 6.1738 | 0.2799 | 0.7818 | 0.7486 | 101 (19.73%) | 90.8840 | 95.3189 | 65.7112 | 55.1668 | 0.1968 | 0.5497 | 0.5264 | 0.8663 | 0.8812 | 0.8603 | 0.4743 | 0.1452 | 0.5739 | 13 (2.54%) |
| exp4_3_1 | mixed PT-exp1 + Thinking-Hard V2 Lean-State 10k | Qwen3.5-0.8B-PT-SFT | lr=5e-5, ep=3, len=16384; eval extracted path p=.95, k=20, t=1 | 374 (73.05%) | 6.1719 | 0.2806 | 0.7815 | 0.7612 | 95 (18.55%) | 90.4117 | 95.2212 | 65.4844 | 55.3858 | 0.2049 | 0.5709 | 0.5561 | 0.8751 | 0.8853 | 0.8658 | 0.4848 | 0.1422 | 0.5783 | 16 (3.12%) |

### 指标含义、计算、来源与作用

所有重复字段已经合并到文档前部[统一指标词典](#统一指标词典)。本表中的字段对应关系为：

- `Connectivity` = `Parsable` 的成功数与比例版本；`Clean (Num, %)` = `Clean` 的成功数与比例版本。
- `SigLIP 2` = `SigLIP2`，`VQAScore` = `VQA`。
- `Parse Prefix` = GRPO 表中的 `Parse`，`Length Score` = GRPO 表中的 `Length`。
- `Dense Reward` 与 GRPO 表中的 `Reward` 公式相同，但前者统计固定 VAL512，后者统计在线训练生成结果。
- `Collision` 是首次失败前的绝对动作数；`Collision Prefix` 是按目标长度归一化的无碰撞进度，两者不能互换。

BrickNet 原始论文行的三项 `Adj.` 由表中已四舍五入的原始值和覆盖率推算，末位可能有舍入误差。

- `Trace-format Valid`（仅显式 trace 实验）：`<think>/<action>` 状态机闭合、每个 action 可提取，且 state-before 与
  已生成 path/inventory 内部一致的样本数和比例。它来自 `trace_extraction_report.json`，用于判断 reasoning 协议是否
  被模型遵守；不等同于 path 可解析，也不替代 evaluator 的 Strict Success。

指标来源和项目决策顺序也统一见词典末尾；汇总文件只能从逐样本证据聚合，不能根据表内均值反向补写。

本表各实验均无额外 LDR 转换或渲染失败，因此图文指标样本数等于
`Connectivity Num`。BrickNet 原始结果没有保存可与当前 VAL reference 对齐的逐样本
path，因此 BLEU/ROUGE 和七项对齐指标记为 `-`，不根据汇总值反推。

## 阶段性结论

- MM-PT-only 在匹配的 image+inventory 协议下达到 60.55% Connectivity、15.23% Clean
  和 2.73% Strict Success，说明 PT adapter 已学到有效的条件生成能力；由于评测输入清空
  caption，该结果不能与带 caption 的 SFT 行解释为严格的训练阶段 ablation。
- Direct SFT 从 10k 增至 50k 后，Connectivity 从 46.29% 提升到 69.14%，扩充数据有效。
- 在相同 50k SFT 数据和 3 epochs 下，加入 MM-PT 初始化后，Connectivity 从 69.14%
  提升到 71.68%，Clean 从 16.60% 提升到 18.16%，Collision 从 5.6426 提升到
  6.0332，Strict Success 从 3.32% 提升到 3.91%。PE、SigLIP2、VQAScore 和五项
  alignment reward 分量也全部提升，说明 MM-PT 对 50k SFT 仍有正向初始化收益。
- MM-PT + SFT-10k 达到 66.41% Connectivity 和 16.60% Clean，接近 Direct SFT-50k，
  说明 MM-PT 提高了小数据 SFT 的数据效率。
- 在 exp3 上继续进行 2k GRPO 后，Collision 从 5.6895 提升到 5.8086，BLEU/ROUGE
  略升；但 Connectivity、Clean、SigLIP2 和 VQAScore 均下降。当前 RL 改善了有效前缀
  长度，尚未改善完整结构成功率或整体视觉语义质量。
- exp3 从 3 epochs 延长到 10 epochs 后，Train loss 从 0.1701 降到 0.1090，但
  Connectivity 从 66.41% 降到 52.15%，Clean 从 16.60% 降到 14.65%，出现明显过拟合。
- Lean-State V2 `exp4_3_1` 相对旧 Thinking-Hard 恢复 parsable、dense 和 strict，但相对 exp4_2 的 strict 持平、
  dense 和 parsable 较低；同时 strict trace-format 仅 7.42%。它既没有主指标点估计优势，也没有 paired
  non-inferiority 证据，reasoning schema 合规性也不足，因此诊断完成后保持 hold。
- exp4_2 Stage6–7 当前五组均待 contract-v1 重跑；旧 B1/V1/V2/A0/A1 及 paired CI 均已撤回。新协议下
  V2→A1 只比较 `final_system`；V2 的
  hard-valid coverage、A0/A1 的显式 feedback 效果及主要瓶颈位置均须等待五组重跑，不能沿用旧结论。
- BLEU/ROUGE 较高不代表路径结构合法；任何 verifier/search 增益仍必须以 pose-aware task strict、dense、
  hard-valid safety 与成本的完整 paired 证据判断。

GRPO-exp0 使用基础 RL 2,000 子集，不是 policy-specific hard 子集。根据 2026-08-05 总推进决策，
policy-specific hard mining 暂停，不再为每个 SFT policy 运行 66,456×8 full mining。只有完成 SFT cold start、
至少一轮 RL 和基础数据扩容后仍明确有提升空间时，才允许新增独立 hard ablation，且 mining 输入最多 2,000 prompts。

## MM-VAL 过拟合与推理参数 ablation

exp1 公共配置为：BrickNet-MM-VAL 512 条、10 epochs、LoRA rank/alpha `16/32`、
learning rate `5e-5`、cosine scheduler、batch size 1、gradient accumulation 8。
推理统一使用 `max_new_tokens=512`，长标签可能被截断。

| Exp | Model / 主要 ablation | Train loss | Sampling | BLEU-4 | ROUGE-L | Parsable | Clean | Collision | 状态 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| exp0 | Qwen3-VL-2B，3 epochs，LoRA 32/64，训练链路 debug | 0.5432 | - | - | - | - | - | - | 训练完成，未正式评测 |
| exp1 | Qwen3-VL-2B，10 epochs | 0.3231 | k20/p0.95/t1.0 | 67.9555 | 51.8508 | 12.89% | 5.86% | 2.5938 | 完成 |
| exp1 | Qwen3.5-0.8B，10 epochs | 0.2980 | k20/p0.95/t1.0 | 69.1700 | 55.0429 | 17.38% | 9.18% | 3.0332 | 完成 |
| exp1 | Qwen3.5-2B，10 epochs | 0.2702 | k20/p0.95/t1.0 | 69.8198 | 56.7406 | 22.66% | 12.89% | 3.3203 | 完成 |
| exp1_1 | Qwen3.5-2B，20 epochs，BS4/GA8，LoRA 32/64，LR `1e-5`，constant-with-warmup | 0.3800 | k20/p0.95/t1.0 | 67.9795 | 51.8816 | 11.33% | 5.86% | 2.5859 | 完成 |
| exp1_1 | 同一 adapter，仅修改 sampling | 0.3800 | k50(default)/p0.9/t0.95 | 68.3941 | 52.3675 | 14.26% | 5.08% | 2.6836 | 完成 |

Qwen3.5-2B exp1 是当前 VAL 过拟合实验中表现最好的配置；exp1_1 延长 epoch、增大
LoRA/有效 batch 并降低学习率后没有提升。由于训练集和评测集相同，这部分结果仅用于
配置调试。

## 原始 BrickNet 无条件 PT 复现

官方 BrickNet PT+SFT adapter 的 caption-conditioned VAL 结果已经并入 Condition
Generation 主表。本节记录固定 `"a"` prompt、`stop_after_newlines=199`、每模型生成
2,048 条样本的无条件 PT 复现结果。

| Exp | Model / Adapter | Parsable | Clean | Collision | 状态 |
| --- | --- | ---: | ---: | ---: | --- |
| BrickNet-PT | Qwen3-0.6B + BrickNet-0.6B-PT | 85.89% | 1.81% | 15.1333 | 完成 |
| BrickNet-PT | Qwen3-1.7B + BrickNet-1.7B-PT | 89.40% | 2.10% | 16.2275 | 完成 |
| BrickNet-PT | Qwen3-4B + BrickNet-4B-PT | 92.43% | 1.32% | 16.1934 | 完成 |
| BrickNet-PT | Qwen3-8B + BrickNet-8B-PT | - | - | - | 启动过但无输出 |
| BrickNet-PT | Qwen3-14B + BrickNet-14B-PT | - | - | - | 启动过但无输出 |
| PT-exp0 | Qwen3.5-0.8B + 本地 MM-PT adapter | 0.00% | 0.00% | 0.0000 | 固定 `"a"` 输入协议不匹配，结果无效 |
| PT-exp0 | Qwen3.5-0.8B + 本地 MM-PT adapter | 60.55% | 15.23% | 5.2188 | MM-PT matched protocol，512 VAL 完成；完整指标见主表 |

本地 MM-PT adapter 训练时接收图像和 inventory 条件，不能使用固定 `"a"` prompt
评测。其 0% 结果不表示 MM-PT 训练失败。正确的 PT-only 推理协议为
`VAL image + empty caption + inventory -> path`。对应数据准备、预测和评测命令统一见 `record.md`，不在结果账本重复。
评测随后使用 ms-swift 的 BrickNet-MM `alignment-worker` 对同一 512 条 prediction/reference
补算 inventory、长度、pose、dense reward 和 strict success。完整产物位于
`../BrickNet/outputs_val/qwen35_08b/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20/metrics.json`
和同目录 `metrics.md`；310 个可解析结构均成功完成 LDR 转换和八视图渲染，无额外失败。

该任务不使用原始 BrickNet text-only PT 的固定 `"a"` prompt，也不使用
`stop_after_newlines=199`；生成应由 image/inventory 条件和模型 EOS 决定终止位置。

## 待完成实验

1. 按 inference-contract v1 将 B1/V1/V2/A0/A1 原路径覆盖重跑，完成增强 preflight、十套
   `final_system`/分模式诊断层评测、同名 manifest/statistics/results 和 paired bootstrap；V2→A1 仅做
   `final_system` paired。
2. Stage6–7 新基线冻结后，重新执行 Stage8 preflight，按 seed-42 10k manifest 物化、token audit 并训练 R1-S 10k。
3. Stage8 comparator 命名已冻结：R1-S/R1-C 与 `S8-ZS-Greedy` 成对，R1-B 与 `S8-ZS-DFS`
   成对；两个比较器使用 `stage8-act` 但不是 Stage6 A0/A1。当前待生成正式 VAL512 final/raw artifact。
4. 即使 R1-S 通过 `S8-ZS-Greedy` 因果 gate，未在 strict/dense/成本上超过重跑后有效 V2 前也不替换
   inference fallback。
5. R1-B 必须等待至少 1,000 个真实有效 rollback transition、覆盖 100 个不同 source；旧 A1 证据已失效。
6. Stage9 multi-turn GRPO 只接收通过上述统一 VAL512 gate 的最佳 R1 checkpoint；不得提前启动。
7. PT-exp2、Stage3 和 `exp2_2` 恢复均保留为独立研究支线，不阻塞 action-only 主线。Stage2 V2 已完成，不再列为
   待执行项；外部 PT-exp2-100k 权重同步前不得启动 `exp4_4_1/exp4_7_1`。policy-specific hard mining 继续暂停。
