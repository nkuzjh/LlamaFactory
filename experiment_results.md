# BrickNet-MM 实验结果账本

本文档是 LEGO 项目的唯一结果事实源。阅读顺序：[BrickNet-MM Agentic LEGO Planner README](../BrickNet/BrickNet-MM%20Agentic%20LEGO%20Planner/README.md) → [Constructor Plan.md](../BrickNet/BrickNet-MM%20Agentic%20LEGO%20Planner/Constructor%20Plan.md)。
命令只见 [record.md](record.md)，项目历史只见 [Progress Log](../BrickNet/BrickNet-MM%20Agentic%20LEGO%20Planner/Progress%20Log.md)；本账本不重复命令或历史流水。

本文只记录可引用的配置、结果、状态和会影响有效性的极短异常。validated/frozen 结果可用于结论，diagnostic 只用于定位，invalidated 不进入模型或系统排名；机器证据统一记为“由 artifact gate 验证”。

## 统计口径

- 后续统一对比实验均在 512 条 BrickNet-MM-VAL 上调参、选择搜索预算并报告指标。已有 non-thinking 实验使用 max_new_tokens=4096、top_k=20、top_p=0.95、temperature=1.0；阶段 2 Thinking-Hard 使用相同 sampling，但为完整 trace 使用 max_new_tokens=16384。
- 该集合会被反复观察，因此结果属于 repeated-use VAL，不是未触碰的 locked test 或无偏泛化估计。
- Train loss 是整个训练过程的平均 next-token cross-entropy loss，不是最后一步 loss。
- GRPO 的 Reward、五个分量和权重统一见[统一指标词典](#统一指标词典)；训练表结果为 1,000 个训练更新步骤的日志均值。
- exp0、exp1、exp1_1 直接使用 BrickNet-MM-VAL 训练，只用于验证训练/推理链路和过拟合能力，不能代表泛化性能。
- 图文指标仅在成功解析、转换为 LDR 并完成八视图渲染的样本上计算；Adj. 指标将未成功渲染的样本按 0 分计入固定的 512 条分母。

## 统一指标词典

本节统一解释后文 exp4_2 Agentic Stage 5–7 结果、GRPO 在线训练结果和 Condition Generation 结果的同名指标。除非表格另有说明，验证集比例均以固定的 512 条样本为分母；百分数只是把 [0, 1] 比例乘以 100 后展示。

- Parsable 与 Connectivity 计算相同，前者只显示比例，后者同时显示成功条数和比例。
- Parse 与 Parse Prefix 计算相同，是“失败前完成了多少”的连续分数；它们不等于只看完整成功与否的 Parsable/Connectivity。
- Length 与 Length Score 计算相同；VQA 与 VQAScore、SigLIP2 与 SigLIP 2 也只是表头写法不同。
- Reward 与 Dense Reward 使用同一加权公式，但 Reward 是在线训练期间模型临时生成结果的训练日志均值，Dense Reward 是固定验证集最终预测的评测均值，因此不能直接把两者当成同一批数据上的前后变化。

### 生成文本、结构安全与图文指标

| 英文名 | 中文对应与计算 | 含义、用途和限制 |
| --- | --- | --- |
| Parsable / Connectivity (Num, %) | 可完整读取率。预测从头到尾都能解析记为成功；Connectivity 展示成功数及其占 512 条的比例，Parsable 只展示比例。 | 表示文本格式和连接树完整，是最低可用性要求；不保证零件清单正确、无碰撞或形状正确。 |
| Clean (Num, %) / Clean | 无碰撞完整率。完整可解析且碰撞列表为空的样本数除以 512；带 Num 的表同时展示成功数。 | 表示基础结构安全；仍不要求与目标三维形状一致。 |
| Collision | 首次失败前平均放置数。单条样本取总动作数、首次解析失败位置和首次碰撞位置中的最小值，无失败时取总动作数，再对 512 条求平均。 | 越高表示通常能连续正确放置得更久。它不是碰撞次数，且受目标长度影响；不要与归一化的 Collision Prefix 混淆。 |
| BLEU-4 | 四阶局部文字重合分。生成路径与参考路径按字符比较连续一至四个字符片段，使用 NLTK 的第 3 种平滑方法，逐样本计分、乘 100 后取平均。 | 反映局部文本模仿程度；文字相似不代表结构合法或形状正确。 |
| ROUGE-1 / ROUGE-2 / ROUGE-L | 顺序文字重合分。使用 jieba 分词后，分别比较单个词、连续两个词和最长保持顺序的公共部分，取兼顾准确与覆盖的分数，逐样本乘 100 后求平均。 | 用于观察生成顺序和参考文本的重合；同一结构可能有不同合法搭建顺序，因此只作辅助诊断。 |
| PE | 图文外观相似分。使用 PE-Core-bigG-14-448 把文字描述和八个渲染视角转成归一化数值特征并计算余弦相似度；每条样本取八视角最高分，再对成功渲染样本求平均。 | 衡量可见外观与描述是否接近；只覆盖可成功解析和渲染的子集。 |
| SigLIP2 / SigLIP 2 | 第二种图文语义相似分。使用 google/siglip2-giant-opt-patch16-384，把每个视角的图文匹配输出转换到 0 至 1；每条样本取八视角最高分，再对成功渲染样本求平均。 | 提供另一种图文语义证据；同样受成功渲染子集的选择影响。 |
| VQA / VQAScore | 问答式图文相符分。向 Qwen/Qwen2-VL-7B-Instruct 询问渲染结构是否符合描述，以第一个回答单位为 Yes 的概率计分；八视角取最高值后求平均。 | 只说明图像与描述是否看起来相符，不证明连接、碰撞或精确姿态正确。 |
| PE Adj. / SigLIP 2 Adj. / VQA Adj. | 覆盖率修正图文分。对应图文均值乘以成功渲染样本数，再除以 512，等价于把未成功渲染样本记为 0 分。 | 防止少量可渲染样本的高分掩盖低覆盖率；必须与未修正图文均值一起解释。 |

### 目标对齐与共享奖励分量

设预测和目标零件数分别为 P/T，按“零件种类与颜色”统计的重合零件数为 O。以下逐样本分数均在 [0, 1] 内，表中展示 512 条均值或成功比例。

| 英文名 | 中文对应与计算 | 含义、用途和限制 |
| --- | --- | --- |
| Parse / Parse Prefix | 可读取进度分。完整解析得 1；否则为已经成功解析的预测零件数除以目标零件数，上限为 1。 | 区分“一开始就失败”和“接近完成才失败”；它是连续完成度，不是完整可读取率。 |
| Inventory F1 | 零件清单综合分，计算为 2O/(P+T)；零件种类、颜色和重复数量都参与比较。 | 同时惩罚多放、少放、用错零件和用错颜色；不检查连接与姿态。 |
| Length / Length Score | 长度分，计算为 min(P,T)/max(P,T)；预测数和目标数相同得 1，两者都为空时也定义为 1。 | 只衡量零件总数接近程度，不能说明使用了正确零件。 |
| Collision Prefix | 无碰撞进度分。没有碰撞得 1；否则为首次碰撞位置除以目标零件数，上限为 1。 | 是按目标长度归一化的安全进度。空结果也可能因“未检测到碰撞”得到 1，必须与可读取进度等其他分量共同看。 |
| Pose Match | 姿态匹配分。先消除预测与目标整体起点和方向的差异，再按相同零件种类和颜色配对；位置误差不超过 0.5、旋转误差不超过 5 度的目标零件数除以目标零件数。 | 衡量是否真正还原目标三维形状，只在有参考答案的评测或训练中使用，不能作为部署时检查程序偷看的信息。 |
| Dense Reward | 连续综合分：0.20×Parse Prefix + 0.20×Inventory F1 + 0.10×Length Score + 0.20×Collision Prefix + 0.30×Pose Match。 | 最终成功较少时用于比较整体进步和定位失败原因；不能代替 Strict Success。 |
| Reward | 在线训练总奖励，使用与 Dense Reward 相同的五项和权重；表中是 1,000 个训练更新步骤的日志均值。 | 统计的是训练时每题生成八个答案的表现，不是固定 VAL512 的最终质量。训练还会在每题内部比较八个答案的相对高低。 |
| Strict Success | 最终任务成功率。必须同时满足完整解析、预测与目标零件数相同、零件种类/颜色/数量完全一致、无碰撞且 Pose Match=1。 | 第一主指标，回答“是否完整还原了目标结构”；不能用文本相似分或仅合法的输出率替代。 |

### Stage 6–7 外层纠错机制与计算成本

| 英文名 | 中文对应与计算 | 含义、用途和限制 |
| --- | --- | --- |
| Controller hard-valid success | 外层程序合法完成数。外层程序返回成功，并经复核满足完整解析、零件清单和长度完全正确、无碰撞。 | 回答“是否得到一个合法、安全、用料正确的完整结构”，但不要求姿态与目标一致，因此只是安全上限，不能代替 Strict Success。 |
| Recovery rate | 恢复率：至少出现过一次拒绝、但最终仍合法完成的样本数，除以至少出现过一次拒绝的样本数。 | 衡量重试、分支尝试和回退救回了多少原本遇到错误的样本。 |
| Mean expansions | 平均展开数：每条样本实际取得的备选动作总数，再对 512 条求平均。 | 近似表示外层程序尝试了多少条路；越大通常表示计算成本越高。 |
| Mean generated tokens | 平均生成量：每条样本所有模型调用产生的基本文字单位总数，再对 512 条求平均。 | 比较不同重试和搜索方案消耗的生成预算。 |
| Mean latency | 平均墙钟用时：从单条样本开始到结束的实际经过时间，再对 512 条求平均。 | 反映用户等待时间；它包含模型生成和外层检查，不等于纯显卡计算时间，机器同时有其他任务时也会受影响。 |

### 成对差值与不确定范围

A → B 的差值统一计算为 B 减 A。两种方法必须使用相同的 512 条样本和相同顺序，先逐样本相减，再用固定随机起点 42 对这 512 个差值进行 10,000 次有放回重复抽取。

| 英文名 | 中文对应与计算 | 判读方式 |
| --- | --- | --- |
| Final strict delta | 系统最终层的 Strict Success 差值，以百分点展示。例如 +1 pp 表示成功率绝对提高 1 个百分点。 | 正数有利于后一个方法，负数有利于前一个方法。 |
| Final dense delta | 系统最终层的 Dense Reward 均值差，直接使用 [0, 1] 分数单位，不是百分点。 | 正数表示后一个方法的连续综合分更高。 |
| 95% CI | 95% 不确定范围：10,000 次重复抽取所得平均差值的第 2.5% 和第 97.5% 位置。 | 整段大于 0 才称稳定提升；整段小于 0 才称稳定下降；跨过 0 表示当前样本不足以证明稳定差异。 |
| 判定 | 根据差值方向、不确定范围是否跨 0，以及实验是否通过协议检查写出的文字结论。 | 协议失效的结果即使区间不跨 0，也只能保留为事故诊断，不能成为项目结论。 |

### 在线训练专用字段

| 英文名 | 中文对应与计算 | 含义、用途和限制 |
| --- | --- | --- |
| Zero-std | 零差异答案组比例。每道训练题生成八个答案；若八个答案的总奖励完全相同，该题组记为零差异，再对训练步骤求平均。 | 零差异组无法告诉模型哪个答案更好，比例过高说明奖励缺乏区分能力；越低通常越有利，但不能单独证明训练有效。 |
| Loss | 在线训练优化目标的日志均值，由题内相对奖励、概率变化限制以及与参考模型的偏离约束共同形成。 | 不是监督训练中的逐字预测误差，数值可接近 0 或短暂为负；不能用其绝对大小跨实验判断生成质量。 |
| 状态 | 实验是否完成、完成多少训练步骤，以及结果是否通过检查或仍被暂停。 | 不是质量分数；只有状态允许进入比较的结果才能支持结论。 |

### 指标文件来源与决策顺序

- BLEU-4 和 ROUGE-* 来自逐样本预测与参考文本。
- Parsable/Connectivity、Clean 和 Collision 来自 BrickNet 解析与真实网格碰撞结果。
- 三种图文分数来自成功渲染的八视角图像，修正值再把未渲染样本计为 0。
- Parse Prefix 到 Strict Success 来自逐样本预测与目标结构的对齐结果。
- 外层程序成功率、恢复率、展开数、生成量和用时来自逐样本执行记录。
- 在线训练的 Reward、五个分量、Zero-std 和 Loss 来自训练状态日志。

项目决策先看 Strict Success 及其 95% 不确定范围，再看 Dense Reward，随后确认合法完整输出覆盖率，最后比较展开数、生成量和用时。结构安全、图文和文字重合指标用于解释原因，不替代第一主指标。

## 实验配置与状态

已有 non-thinking BrickNet-MM 实验均使用 Qwen/Qwen3.5-0.8B、qwen3_5_nothink、cutoff_len=4096、LoRA target all、LoRA rank/alpha 64/128。PT/SFT 使用 batch size 2、gradient accumulation 8、learning rate 5e-5 和 cosine scheduler。

阶段 2 Thinking-Hard 是明确例外：qwen3_5_nothink、enable_thinking=false、cutoff_len=16384、packing=false、train_on_prompt=false，推理 max_new_tokens=16384。显式 <think> 是 assistant 监督文本，不叠加 Qwen 原生 thinking 模板。无思考对照统一命名为 NonThinking-Control。PT-exp2 下游使用独立 PT-exp2-text8m/mm-e1/e2/e3 命名；exp4_4/exp4_7 为同初始化 10k paired 实验。数据、配置和 gate 见 [PT-exp2 专题](../BrickNet/BrickNet-MM%20Agentic%20LEGO%20Planner/PT-exp2%20Pretraining%20and%20Downstream%20Plan.md) 与 [Stage 2 专题](../BrickNet/BrickNet-MM%20Agentic%20LEGO%20Planner/Stage%202%20Reasoning%20Data%20and%20Experiments.md)。

| Exp | 框架 | Train output | 初始化 | 数据 | 样本数 | Epoch | 主要 ablation | Train loss | 状态 |
| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |
| PT-exp0 | LlamaFactory | train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64 | Qwen3.5-0.8B | BrickNet-MM-PT | 135,051 | 3 | MM-PT：图像+inventory → path | 0.1507 | 完成并评测 |
| PT-exp1 | LlamaFactory | train_PT_exp1_qwen35_08b_bricknet_text270k_mmpt135k_ep1_bs2_ga8_lora64 | Qwen3.5-0.8B | BrickNet text PT + BrickNet-MM-PT | 405,153 | 1 | 固定曝光预算的 text+MM mixed PT | 0.0683 | 完成；final 作为 Stage 2 共同初始化 |
| PT-exp2-text8m | LlamaFactory | train_PT_exp2_text8m_qwen35_08b_path7698261_steps250k_bs4_gbs32_lora64_len6401_nopack | Qwen3.5-0.8B | first-round + cross-pool exact-dedup text path PT | 7,698,261 | 250k steps（约 1.03919 epoch） | 实际双卡 BS16、GA1、global batch 32；path+EOS、6,401-token non-packed full-sequence PT；目录中的 bs4 为遗留命名 | 0.3343 | 250,000/250,000 完成；final adapter/train/eval artifact 完整 |
| PT-exp2-mm-e1/e2/e3-v2 | LlamaFactory | train_PT_exp2_mm_{e1,e2,e3}_v2_nometa_qwen35_08b_text8m_mm135k_replay1to1_ep1_bs2_gbs16_lora64_len6400 | PT-exp2-text8m → v2 e1 → v2 e2 adapter | all MM-PT + 三组不重叠 1:1 text replay 的无 meta 训练投影 | 150,668 / 150,637 / 150,718 | 各 1 epoch | 物理 CUDA 1 单卡 BS2、GA8、global batch 16；三次顺序 multimodal consolidation；每轮后独立 VAL512 推理 | - | 三轮训练、512/512 prediction/scored/alignment 与完整评测完成；固定规则 e1 选中并冻结 |
| PT-exp2-mm-rowbal-cont3 | LlamaFactory | train_PT_exp2_mm_rowbal_cont3_qwen35_08b_text8m250k_mm135051_text135051_ep3_bs2_gbs16_lora64_len6400 | PT-exp2-text8m 250k final adapter | 固定 1:1 行数平衡 MM/text（无 meta） | 270,102（135,051 + 135,051） | 3（连续；同一 optimizer/scheduler） | LR 1e-5、cosine/warmup 3%、CUDA1 单卡 BS2/GA8/global16、random、epoch/full-state saves；steps 16,882/33,764/50,646；训练后按 ep1/2/ep3 做 VAL512 | 0.2540496625 | training complete / valid；ep1/ep2/ep3 prediction/evaluation complete/valid；点估计推荐 checkpoint-33764 |
| exp2 | LlamaFactory | train_exp2_qwen35_08b_sft1w_ep3_bs2_ga8_lora64 | Qwen3.5-0.8B | BrickNet-MM-SFT | 10,000 | 3 | 无 PT，小规模 SFT | 0.2418 | 完成并评测 |
| exp2_1 | LlamaFactory | train_exp2_1_qwen35_08b_sft5w_ep3_bs2_ga8_lora64 | Qwen3.5-0.8B | BrickNet-MM-SFT | 50,000 | 3 | 无 PT，扩大 SFT 数据量 | 0.2031 | 完成并评测 |
| exp2_2 | LlamaFactory | train_exp2_2_qwen35_08b_sft_ep3_bs2_ga8_lora64 | Qwen3.5-0.8B | BrickNet-MM-SFT | 334,355 | 3 | 无 PT，全量 SFT | - | 中断于 20,660/62,694；可恢复 checkpoint-20000 |
| exp3 | LlamaFactory | train_exp3_qwen35_08b_pt_sft1w_ep3_bs2_ga8_lora64 | PT-exp0 adapter | BrickNet-MM-SFT | 10,000 | 3 | PT 初始化后新建 SFT adapter | 0.1701 | 完成并评测 |
| exp3_0_1 | LlamaFactory | train_exp3_0_1_qwen35_08b_pt_sft1w_ep10_bs2_ga8_lora64 | PT-exp0 adapter | BrickNet-MM-SFT | 10,000 | 10 | exp3 的 epoch ablation | 0.1090 | 完成并评测 |
| GRPO-exp0 | ms-swift | ../ms-swift/output/bricknet_grpo/exp0_qwen35_08b_exp3_rl_n2000_g8 | PT-exp0 merged + exp3 adapter | BrickNet-MM-RL | 2,000 | 1 | GRPO，五项结构/几何 reward，G=8 | - | 完成并评测 |
| exp3_1 | LlamaFactory | train_exp3_1_qwen35_08b_pt_sft5w_ep3_bs2_ga8_lora64 | PT-exp0 adapter | BrickNet-MM-SFT | 50,000 | 3 | PT + 50k SFT | 0.1673 | 完成并评测 |
| exp3_2 | LlamaFactory | train_exp3_2_qwen35_08b_pt_sft_ep3_bs2_ga8_lora64 | PT-exp0 adapter | BrickNet-MM-SFT | 334,355 | 3 | PT + 全量 SFT | - | 未开始 |
| exp4 | LlamaFactory | train_exp4_qwen35_08b_mixedpt_stage2_nonthinking_control_val511_ep3_bs1_ga16_lora64_len16384 | mixed PT-exp1 final | Stage2 NonThinking-Control VAL511 | 511 | 3 | 无思考 overfit 链路检查 | 0.1738 | 训练完成；VAL512 512/512 推理完成 |
| exp4_1 | LlamaFactory | train_exp4_1_qwen35_08b_mixedpt_stage2_thinking_hard_val511_ep3_bs1_ga16_lora64_len16384 | mixed PT-exp1 final | Stage2 Thinking-Hard VAL511 | 511 | 3 | Thinking-Hard overfit 链路检查 | 0.0860 | 训练完成；VAL512 512/512 推理完成 |
| exp4_2 | LlamaFactory | train_exp4_2_qwen35_08b_mixedpt_stage2_nonthinking_control_10k_ep3_bs1_ga16_lora64_len16384 | mixed PT-exp1 final | Stage2 NonThinking-Control 10k | 10,000 | 3 | 无思考正式 paired 对照 | 0.1727 | 训练、512 推理和全指标完成 |
| exp4_3 | LlamaFactory | train_exp4_3_qwen35_08b_mixedpt_stage2_thinking_hard_10k_ep3_bs1_ga16_lora64_len16384 | mixed PT-exp1 final | Stage2 Thinking-Hard 10k | 10,000 | 3 | Thinking-Hard 正式 paired 实验 | 0.0434 | 训练、512 推理、strict extraction 和全指标完成 |
| exp4_3_1 | LlamaFactory | train_exp4_3_1_qwen35_08b_mixedpt_stage2_thinking_hard_v2_lean_state_10k_ep3_bs1_ga16_lora64_len16384 | mixed PT-exp1 final | Stage2 V2 Thinking-Hard Lean-State 10k | 10,000 | 3 | 移除 GT-next-action 泄漏的短 state-before 诊断 | 0.1083 | 训练、512 推理、strict extraction 和全指标完成；hold |
| exp4_4 | LlamaFactory | train_exp4_4_qwen35_08b_PT_exp2_v2_stage2_nonthinking_control_10k_ep3_bs1_ga16_lora64_len16384 | final PT-exp2-v2 e1 | Stage2 NonThinking-Control 10k | 10,000 | 3 | CUDA1 单卡、BS1/GA16/global batch 16；与 exp4_7 同初始化/预算；无 VAL511 | 0.1477434707 | 训练 1,875/1,875、512 推理和全指标完成；validated/frozen |
| exp4_7 | LlamaFactory | train_exp4_7_qwen35_08b_PT_exp2_v2_stage2_thinking_hard_v2_lean_state_10k_ep3_bs1_ga16_lora64_len16384 | final PT-exp2-v2 e1 | Stage2 V2 Lean-State 10k | 10,000 | 3 | CUDA1 单卡、BS1/GA16/global batch 16；只替换 exp4_4 的监督数据 | 0.10565751036008199 | 训练 1,875/1,875、512 推理和全指标完成；trace warning；validated/frozen |
| exp4_5 | LlamaFactory | train_exp4_5_qwen35_08b_PT_exp2_stage2_nonthinking_control_50k_ep3_bs1_gbs16_lora64_len16384 | final PT-exp2 | Stage2 NonThinking-Control 50k | 50,000 | 3 | 单/双卡自适应、global batch 16；exp4_4 收益 gate 后扩容 | - | dormant；未物化 50k |
| exp4_6 | LlamaFactory | train_exp4_6_qwen35_08b_PT_exp2_stage2_nonthinking_control_all66456_ep3_bs1_gbs16_lora64_len16384 | final PT-exp2 | Stage2 NonThinking-Control all | 66,456 | 3 | 单/双卡自适应、global batch 16；exp4_5 收益 gate 后扩容 | - | dormant |
| exp4_4_1 | LlamaFactory | train_exp4_4_1_qwen35_08b_PT_exp2_100k_stage2_nonthinking_control_10k_ep3_bs1_gbs16_lora64_len16384 | external PT-exp2 checkpoint-100000 | Stage2 NonThinking-Control 10k | 10,000 | 3 | 外部 100k-step PT 权重的独立下游 | 0.1601 | 训练 1,875/1,875 steps、512 推理和全指标完成；validated/hold |
| exp4_7_1 | LlamaFactory | train_exp4_7_1_qwen35_08b_PT_exp2_100k_stage2_thinking_hard_v2_lean_state_10k_ep3_bs1_gbs16_lora64_len16384 | external PT-exp2 checkpoint-100000 | Stage2 V2 Lean-State 10k | 10,000 | 3 | 外部 100k-step PT 权重 + Lean-State | 0.1085 | 训练 1,875/1,875 steps、512 推理、strict extraction 和全指标完成；validated/hold |
| exp4_4_3 | LlamaFactory | train_exp4_4_3_qwen35_08b_PT_exp2_mm_rowbal_cont3_ep3_stage2_nonthinking_control_10k_ep3_bs1_ga16_lora64_len16384 | rowbal-cont3 ep3 checkpoint-50646 | Stage2 NonThinking-Control 10k | 10,000 | 3 | rowbal-cont3 ep3 endpoint-sensitivity；CUDA0 单卡、BS1/GA16/global batch 16 | 0.14706837952931723 | 训练 1,875/1,875、512 推理和全指标完成；complete / valid |
| exp4_7_3 | LlamaFactory | train_exp4_7_3_qwen35_08b_PT_exp2_mm_rowbal_cont3_ep3_stage2_thinking_hard_v2_lean_state_10k_ep3_bs1_ga16_lora64_len16384 | rowbal-cont3 ep3 checkpoint-50646 | Stage2 V2 Lean-State 10k | 10,000 | 3 | rowbal-cont3 ep3 endpoint-sensitivity；CUDA0 单卡、BS1/GA16/global batch 16 | 0.10613786784807841 | 训练 1,875/1,875、512 推理和全指标完成；complete / valid，trace warning |
| Stage8 R1-S | LlamaFactory | train_stage8_r1_s_act_success_10k_ep3_bs1_ga16_lora64_len16384 | PT-exp1 + exp4_2 adapters | GT success-only Act trajectory | 10,000 sources | 3 | 逐 placement observation 协议 cold start；accepted-action boundary window | - | 历史 64-source protocol/window 测试通过；正式 10k 未物化/训练 |
| Stage8 R1-C | LlamaFactory | train_stage8_r1_c_act_correction_10k_token_matched_lora64_len16384 | PT-exp1 + exp4_2 adapters，新 LoRA | supervised tokens 80% success + 20% real rejection/correction | 10,000 sources | token-matched | 与 R1-S supervised action-token budget 匹配 | - | blocked；等待 R1-S checkpoint、paired gate 和 policy rejection |
| Stage8 R1-B | LlamaFactory | train_stage8_r1_b_act_rollback_10k_token_matched_lora64_len16384 | PT-exp1 + exp4_2 adapters，新 LoRA | supervised tokens 70% success + 20% correction + 10% rollback | 10,000 sources | token-matched | 只接受真实 successful rollback branch | - | blocked；等待至少 1,000 rollback transitions / 100 sources |

### 状态与异常

| 对象 | 状态 | 是否进入当前比较 | 最短有效性说明 |
| --- | --- | --- | --- |
| Stage6–7 B1/V1/V2/A0 | validated/frozen; adopted | 是 | 四组正式结果均完成，由 artifact gate 验证 |
| Stage6–7 compact A1 | implementation complete; pending | 否 | 只等待 A1 单独重跑；不重跑已采用的四组 |
| Stage6–7 A1 v1 partial | invalidated | 否 | 完整失败子树反复回传导致第 2 行 prefill OOM；partial 不是正式 artifact |
| exp4_3_1 Lean-State | validated/hold; trace warning | 是 | path 评测有效，但 strict trace-format 仅 38/512，reasoning 保持 hold |
| exp4_4_1 / exp4_7_1 external 100k | validated/hold | 是 | 独立 PT 初始化支线；Lean-State trace-format 31/512；不替换 exp4_2 主线 |
| PT-exp2-mm v1 mixed-meta e1 | invalidated | 否 | Arrow 的异构 meta 在训练前失败，未形成训练结果 |
| PT-exp2-mm v2 e1/e2/e3 | validated/frozen | 是（e1 选中） | 三轮结果完整；按 strict → dense → clean → parsable 选 e1 |
| PT-exp2-mm-rowbal-cont3 | complete / valid | 是（事实比较） | 连续 3 epoch、三个 checkpoint 和三轮评测完成；点估计推荐 checkpoint-33764 |
| Stage8 R1-S | protocol validated / artifact blocked | 仅协议证据 | token gate 未通过，不是正式训练结果 |

### PT-exp2 初始化与选择结果

| Run | Train steps | Epoch | Train loss | Eval loss | Eval perplexity | Input tokens | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PT-exp2-text8m | 250,000/250,000 | 1.0391942503460516 | 0.3343376237487793 | 0.29681164026260376 | 1.345561825904564 | 13,681,772,704 | validated/frozen；作为 v2 e1 初始化 |

| Run | Train global/max | Outputs | Strict Success | Dense Reward | Clean | Parsable | 选择 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| mm-e1 | 9417/9417 | 512/512 | 6/512 | 0.5857273423 | 0.220703125 | 0.8203125 | selected |
| mm-e2 | 9415/9415 | 512/512 | 4/512 | 0.5860430454 | 0.208984375 | 0.794921875 | - |
| mm-e3 | 9420/9420 | 512/512 | 4/512 | 0.5941138849 | 0.220703125 | 0.806640625 | - |

### PT-exp2-mm-rowbal-cont3 checkpoint 对比

以下表格收纳三个 checkpoint 的核心 prediction/evaluation 数值；训练和推理流水不在结果账本展开。

| Checkpoint | Prediction / evaluation | Rendered rows | Skipped | Render failures | BLEU-4 | ROUGE-1 | ROUGE-2 | ROUGE-L | Parsable | Clean | Collision mean actions | Parse Prefix | Inventory F1 | Length Score | Collision Prefix | Pose Match | Dense Reward | Strict Success | Exact path | PE | SigLIP2 | VQA |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| checkpoint-16882（ep1） | 512/512 | 404 | - | 0 | 79.8114 | 91.6656 | 62.759 | 53.2726 | 404/512 | 105/512 | 8.421875 | 0.9102900774232086 | 0.8155414811184989 | 0.809655785706696 | 0.5570950990310708 | 0.12373974525207188 | 0.5746728336608469 | 2/512 (0.00390625) | - | 0.27744618028697399 | 0.76755156375394007 | 0.74573100058564745 |
| checkpoint-33764（ep2） | 512/512 | 402 | - | 0 | 83.8182900390625 | 93.14293828125 | 64.2137451171875 | 53.681630664062496 | 402/512 | 113/512 | 8.033203125 | 0.9183373919040974 | 0.8566068334354298 | 0.8518979466795903 | 0.5456691592547537 | 0.13199846714201502 | 0.5889120117294198 | 8/512 (0.015625) | 3 | 0.27715973355876866 | 0.7880970399771163 | 0.7456277761960504 |
| checkpoint-50646（ep3） | 512/512 | 405 | 107 | 0 | 84.08433886718751 | 93.0461208984375 | 64.09938632812501 | 53.8440525390625 | 405/512 | 105/512 | - | 0.9130015178424298 | 0.8555066953271355 | 0.8465148582783681 | 0.5459210124616339 | 0.1257369235575623 | 0.5852584080213454 | 5/512 (0.009765625) | 0 | 0.27697331934799385 | 0.77552385447937766 | 0.73738091385658877 |

## exp4_2 Agentic Stage 5–7 结果

Stage5 full-pool replay 的正式结果如下；Stage6–7 的层级和成本结果见后表。

| Stage | Protocol / result | 状态 |
| --- | --- | --- |
| Stage5 | 66,456/66,456 reference；1,751,435 GT actions；failures=0 | validated/frozen，由 artifact gate 验证 |

Stage6–7 的 final_system 是 controller 最终可以安全交付的 hard-valid path，失败 episode 记为空串。诊断层按模式分为 full_path_raw（B1）、greedy_observed_candidate0_prefix（V1/A0）和 search_observed_candidate0_prefix（V2/A1）；系统选择以 pose-aware task strict success 为第一指标、dense reward 为第二指标，不能用 controller hard-valid success 代替 task strict success。B1/V1/V2/A0 已采用；A1 两次旧实现均为 partial，compact A1 仍 pending。

| Mode | Layer | Parsable | Clean | PE | SigLIP2 | VQA | BLEU-4 | ROUGE-L | Inventory F1 | Pose Match | Dense Reward | Strict Success |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 post-hoc（validated/adopted） | final_system | 18.95% | 18.95% | 0.305568 | 0.917509 | 0.819251 | 17.7410 | 12.7263 | 0.189453 | 0.084633 | 0.320116 | 15/512 (2.93%) |
| B1 post-hoc（validated/adopted） | full_path_raw | 72.85% | 19.14% | 0.279856 | 0.801712 | 0.760395 | 90.4902 | 55.7241 | 0.900121 | 0.142103 | 0.584527 | 15/512 (2.93%) |
| V1 silent retry（validated/adopted） | final_system | 28.32% | 28.32% | 0.296717 | 0.876359 | 0.794801 | 26.3474 | 17.7328 | 0.283203 | 0.091264 | 0.368981 | 13/512 (2.54%) |
| V1 silent retry（validated/adopted） | greedy_observed_candidate0_prefix | 92.97% | 18.95% | 0.254801 | 0.620582 | 0.560921 | 32.6413 | 38.4381 | 0.555496 | 0.137295 | 0.483099 | 13/512 (2.54%) |
| V2 silent DFS（validated/adopted） | final_system | 67.38% | 67.38% | 0.281494 | 0.814256 | 0.775299 | 62.3077 | 38.4094 | 0.673828 | 0.131197 | 0.576273 | 17/512 (3.32%) |
| V2 silent DFS（validated/adopted） | search_observed_candidate0_prefix | 99.22% | 97.66% | 0.183736 | 0.204999 | 0.189219 | 2.5720 | 11.9388 | 0.183169 | 0.087427 | 0.470342 | 1/512 (0.20%) |
| A0 explicit feedback（validated/adopted） | final_system | 0.00% | 0.00% | - | - | - | 0.0000 | 0.0000 | 0.000000 | 0.000000 | 0.200000 | 0/512 |
| A0 explicit feedback（validated/adopted） | greedy_observed_candidate0_prefix | 0.78% | 0.59% | 0.190674 | 0.381594 | 0.343724 | 3.2756 | 17.5184 | 0.134770 | 0.075274 | 0.273344 | 0/512 |
| A1 feedback search（invalidated） | legacy final | 5.27% | 5.27% | 0.307337 | 0.892875 | 0.774451 | 4.8555 | 3.2588 | 0.052734 | 0.019809 | 0.232310 | 3/512 (0.59%) |
| A1 feedback search（invalidated） | legacy raw | 0.20% | 0.20% | 0.288818 | 0.777344 | 0.464743 | 0.1923 | 0.1172 | 0.001953 | 0.001674 | 0.201479 | 0/512 |

前八行为当前采用的四组冻结结果；末两行只保留更早 A1 异常实现的历史诊断。Parsable/Clean 检查原始 prediction，task strict 在统一换行规范化后计算，两者输入边界不同，不能互相倒填。

| Mode | Controller hard-valid success | Recovery rate | Mean expansions | Mean generated tokens | Mean latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1（validated/adopted） | 97/512 | 0.00% | 1.00 | 984.66 | 20.364 s |
| V1（validated/adopted） | 145/512 | 14.05% | 20.44 | 465.41 | 18.021 s |
| V2（validated/adopted） | 345/512 | 62.47% | 476.77 | 14,786.60 | 153.498 s |
| A0（validated/adopted） | 0/512 | 0.00% | 5.85 | 123.30 | 4.995 s |
| A1（invalidated） | 27/512 | 5.27% | 295.81 | 8,647.52 | 69.890 s |

所有 controller 成功输出经 evaluator 复核均为 hard-valid；当前只报告 wall-clock latency，不能宣称 GPU-time 优势。

下表来自更早的 invalidated cohort，只保留历史 provenance，不是当前统计结论；当前 mixed-contract paired bootstrap 等 compact A1 后再行统计。箭头方向 A → B 始终表示 B-A。

| Comparison | Final strict delta | 95% CI | Final dense delta | 95% CI | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| B1 → V1 | -0.391 pp | [-1.562, +0.586] pp | +0.048864 | [+0.030827, +0.067106] | strict 未见稳定差异；V1 dense 显著提高 |
| V1 → A0 | -2.539 pp | [-3.906, -1.172] pp | -0.167711 | [-0.191202, -0.144270] | 未训练显式反馈显著退化 |
| V2 → A1 | -2.539 pp | [-4.102, -1.367] pp | -0.354754 | [-0.378715, -0.330678] | feedback search 显著劣于 silent DFS |
| B0/exp4_2 → V2 final | 0.000 pp | [-0.977, +1.172] pp | +0.005471 | [-0.015181, +0.025735] | strict/dense 均未证明稳定提升 |

Stage6–7 的 gate 结果由 artifact gate 验证。

## GRPO 在线训练结果

本表的 Parse/Inventory F1/Length/Collision Prefix/Pose Match 分别对应统一词典中的 Parse Prefix/Inventory F1/Length Score/Collision Prefix/Pose Match；Reward 使用与 Dense Reward 相同的加权公式，但统计的是在线训练生成结果。Zero-std、Loss 和状态见[在线训练专用字段](#在线训练专用字段)。

| Exp | Reward | Parse | Inventory F1 | Length | Collision Prefix | Pose Match | Zero-std | Loss | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| GRPO-exp0 | 0.451437 | 0.707373 | 0.757767 | 0.702507 | 0.325427 | 0.076910 | 1.20% | 5.732e-7 | 1000/1000，完成 |

GRPO-exp0 的训练前 reference reward 为 1.0；正式结果由 artifact gate 验证。

## Condition Generation 结果

BrickNet-MM 和 BrickNet-MM-RL 均在同一组 512 条 BrickNet-MM-VAL 上评测。除新增的 Exp 列外，Train Data 至 VQAScore 八列沿用 BrickNet condition generation 的展示方式，其余指标追加在最右侧。论文公布的基线统一标记为 BrickNet-paper。

### 官方 BrickNet-Render 结果

以下表格只记录采用官方 BrickNet-Render CLI（bricknet-render ... --views 8）重新渲染后计算的指标。GT 上限使用已全量验证的 512×8 个官方 raw views；其非渲染字段记为 -，五个 BrickNet-paper SFT 模型均已完成全量验证。

| Exp | Train Data | Model | Training / Sampling | Connectivity (Num, %) | Collision | PE | SigLIP 2 | VQAScore | Clean (Num, %) | BLEU-4 | ROUGE-1 | ROUGE-2 | ROUGE-L | PE Adj. | SigLIP 2 Adj. | VQA Adj. | Parse Prefix | Inventory F1 | Length Score | Collision Prefix | Pose Match | Dense Reward | Strict Success |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GT upper bound | BrickNet-MM-VAL | Ground Truth | - | - | - | 0.3151 | 0.8209 | 0.7388 | - | - | - | - | - | 0.3151 | 0.8209 | 0.7388 | - | - | - | - | - | - | - |
| BrickNet-paper | BrickNet | Qwen3-0.6B-SFT | - | 466 (91.02%) | 10.6504 | 0.2777 | 0.5897 | 0.5794 | 151 (29.49%) | - | - | - | - | 0.2528 | 0.5367 | 0.5273 | - | - | - | - | - | - | - |
| BrickNet-paper | BrickNet | Qwen3-1.7B-SFT | - | 469 (91.60%) | 10.9980 | 0.2799 | 0.6306 | 0.5881 | 134 (26.17%) | - | - | - | - | 0.2564 | 0.5776 | 0.5387 | - | - | - | - | - | - | - |
| BrickNet-paper | BrickNet | Qwen3-4B-SFT | - | 481 (93.95%) | 10.2598 | 0.2837 | 0.6355 | 0.6167 | 146 (28.52%) | - | - | - | - | 0.2665 | 0.5970 | 0.5794 | - | - | - | - | - | - | - |
| BrickNet-paper | BrickNet | Qwen3-8B-SFT | - | 483 (94.34%) | 11.2227 | 0.2829 | 0.6407 | 0.6118 | 149 (29.10%) | - | - | - | - | 0.2669 | 0.6045 | 0.5771 | - | - | - | - | - | - | - |
| BrickNet-paper | BrickNet | Qwen3-14B-SFT | - | 479 (93.55%) | 11.3047 | 0.2842 | 0.6427 | 0.6214 | 151 (29.49%) | - | - | - | - | 0.2659 | 0.6013 | 0.5813 | - | - | - | - | - | - | - |

五个模型的三项图文指标均完整通过，无最终 OOM 空值；部分 SigLIP2 使用单视图 micro-batch，部分 VQAScore 使用 2 GiB 上限，Qwen3-1.7B-SFT 首次 VQAScore OOM 后低显存重试通过。

### 指标含义、计算、来源与作用

所有重复字段已经合并到文档前部[统一指标词典](#统一指标词典)。本表中的字段对应关系为：

- Connectivity = Parsable 的成功数与比例版本；Clean (Num, %) = Clean 的成功数与比例版本。
- SigLIP 2 = SigLIP2，VQAScore = VQA。
- Parse Prefix = GRPO 表中的 Parse，Length Score = GRPO 表中的 Length。
- Dense Reward 与 GRPO 表中的 Reward 公式相同，但前者统计固定 VAL512，后者统计在线训练生成结果。
- Collision 是首次失败前的绝对动作数；Collision Prefix 是按目标长度归一化的无碰撞进度，两者不能互换。
- Trace-format Valid（仅显式 trace 实验）要求 think/action 状态机闭合、每个 action 可提取，且 state-before 与已生成 path/inventory 内部一致；它用于判断 reasoning 协议是否被模型遵守，不等同于 path 可解析，也不替代 evaluator 的 Strict Success。

指标来源和项目决策顺序也统一见词典末尾；汇总文件只能从逐样本证据聚合，不能根据表内均值反向补写。

### 历史记录：旧版渲染方法

以下表格原样保留此前采用旧版渲染方法得到的历史结果。原始论文行的三项 Adj. 由表中已四舍五入的原始值和覆盖率推算，末位可能有舍入误差。

本表各实验均无额外 LDR 转换或渲染失败，因此图文指标样本数等于 Connectivity Num。BrickNet 原始结果没有保存可与当前 VAL reference 对齐的逐样本 path，因此 BLEU/ROUGE 和七项对齐指标记为 -，不根据汇总值反推。

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
| exp4_4 | PT-exp2-v2 + NonThinking-Control 10k | Qwen3.5-0.8B-PT-SFT | lr=5e-5, ep=3, len=16384; eval p=.95, k=20, t=1 | 407 (79.49%) | 8.6895 | 0.2796 | 0.7899 | 0.7458 | 122 (23.83%) | 87.1478 | 94.2752 | 65.2193 | 55.1334 | 0.2223 | 0.6279 | 0.5929 | 0.9260 | 0.9050 | 0.8841 | 0.5516 | 0.1378 | 0.6063 | 11 (2.15%) |
| exp4_7 | PT-exp2-v2 + Thinking-Hard V2 Lean-State 10k | Qwen3.5-0.8B-PT-SFT | lr=5e-5, ep=3, len=16384; eval extracted path p=.95, k=20, t=1 | 409 (79.88%) | 8.0430 | 0.2795 | 0.8001 | 0.7507 | 125 (24.41%) | 90.0337 | 94.8888 | 65.4540 | 55.0061 | 0.2233 | 0.6391 | 0.5997 | 0.9100 | 0.8982 | 0.8978 | 0.5478 | 0.1457 | 0.6047 | 13 (2.54%) |
| exp4_4_1 | external PT-exp2 checkpoint-100000 + NonThinking-Control 10k | Qwen3.5-0.8B-PT-SFT | lr=5e-5, ep=3, len=16384; eval p=.95, k=20, t=1 | 399 (77.93%) | 8.2031 | 0.2801 | 0.7939 | 0.7500 | 115 (22.46%) | 85.0800 | 93.6209 | 64.4886 | 53.8982 | 0.2183 | 0.6187 | 0.5844 | 0.9141 | 0.8829 | 0.8553 | 0.5430 | 0.1349 | 0.5940 | 8 (1.56%) |
| exp4_7_1 | external PT-exp2 checkpoint-100000 + Thinking-Hard V2 Lean-State 10k | Qwen3.5-0.8B-PT-SFT | lr=5e-5, ep=3, len=16384; eval extracted path p=.95, k=20, t=1 | 412 (80.47%) | 7.5469 | 0.2792 | 0.7783 | 0.7469 | 128 (25.00%) | 89.7759 | 94.7682 | 65.3057 | 55.1706 | 0.2247 | 0.6263 | 0.6010 | 0.9068 | 0.8912 | 0.8965 | 0.5253 | 0.1390 | 0.5960 | 12 (2.34%) |
| exp4_4_2 | PT-exp2 text8m 250k + NonThinking-Control 10k | Qwen3.5-0.8B-PT-SFT | lr=5e-5, ep=3, len=16384; eval p=.95, k=20, t=1 | 409 (79.88%) | 8.1895 | 0.2817 | 0.8218 | 0.7653 | 117 (22.85%) | 85.0105 | 93.7561 | 64.7255 | 54.2303 | 0.2250 | 0.6565 | 0.6113 | 0.9239 | 0.8837 | 0.8652 | 0.5351 | 0.1448 | 0.5985 | 14 (2.73%) |
| exp4_7_2 | PT-exp2 text8m 250k + Thinking-Hard V2 Lean-State 10k | Qwen3.5-0.8B-PT-SFT | lr=5e-5, ep=3, len=16384; eval extracted path p=.95, k=20, t=1 | 423 (82.62%) | 8.0684 | 0.2774 | 0.7915 | 0.7458 | 130 (25.39%) | 89.0394 | 94.3946 | 65.1474 | 54.8891 | 0.2292 | 0.6539 | 0.6161 | 0.9326 | 0.9001 | 0.9151 | 0.5436 | 0.1397 | 0.6087 | 11 (2.15%) |
| exp4_4_3 | PT-exp2 mm-rowbal-cont3 ep3 + NonThinking-Control 10k | Qwen3.5-0.8B-PT-SFT | lr=5e-5, ep=3, len=16384; eval p=.95, k=20, t=1 | 406 (79.30%) | 8.5078 | 0.2792 | 0.8016 | 0.7564 | 123 (24.02%) | 86.7395 | 94.1259 | 64.7050 | 53.6856 | 0.2214 | 0.6357 | 0.5998 | 0.9182 | 0.8932 | 0.8772 | 0.5668 | 0.1419 | 0.6059 | 10 (1.95%) |
| exp4_7_3 | PT-exp2 mm-rowbal-cont3 ep3 + Thinking-Hard V2 Lean-State 10k | Qwen3.5-0.8B-PT-SFT | lr=5e-5, ep=3, len=16384; eval extracted path p=.95, k=20, t=1 | 415 (81.05%) | 8.2773 | 0.2802 | 0.7932 | 0.7604 | 116 (22.66%) | 89.4535 | 94.6925 | 65.2733 | 54.7981 | 0.2271 | 0.6429 | 0.6163 | 0.9267 | 0.9099 | 0.9091 | 0.5449 | 0.1422 | 0.6099 | 12 (2.34%) |

## 阶段性结论

- MM-PT-only 在匹配的 image+inventory 协议下证明了条件生成能力；它与带 caption 的 SFT 行输入不同，不能当作严格的训练阶段 ablation。
- Direct SFT 随数据扩充受益，MM-PT 对小数据 SFT 仍有初始化和数据效率价值；延长 epoch 的训练损失下降不等于泛化提升。
- GRPO 改善了部分有效前缀和连续奖励，但未改善 strict success，不能据此启动 policy-specific hard mining。
- Thinking-Hard 与 Lean-State 没有稳定主指标优势；低 trace-format 合规性说明 reasoning schema 仍应 hold。
- PT-exp2-v2、external 100k 和 rowbal-cont3 的 Control/Lean-State 比较均为混合点估计，尚无单一方案支配，不能宣称 winner。
- Stage6–7 的 B1/V1/V2/A0 已采用；A1 v1 因完整失败子树造成 prompt OOM 而 invalidated，compact A1 仍是独立待办。
- Stage8 及之后的路线与 gate 以 [Constructor Plan.md](../BrickNet/BrickNet-MM%20Agentic%20LEGO%20Planner/Constructor%20Plan.md) 为准。

## MM-VAL 过拟合与推理参数 ablation

exp1 公共配置为：BrickNet-MM-VAL 512 条、10 epochs、LoRA rank/alpha 16/32、learning rate 5e-5、cosine scheduler、batch size 1、gradient accumulation 8。推理统一使用 max_new_tokens=512，长标签可能被截断。

| Exp | Model / 主要 ablation | Train loss | Sampling | BLEU-4 | ROUGE-L | Parsable | Clean | Collision | 状态 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| exp0 | Qwen3-VL-2B，3 epochs，LoRA 32/64，训练链路 debug | 0.5432 | - | - | - | - | - | - | 训练完成，未正式评测 |
| exp1 | Qwen3-VL-2B，10 epochs | 0.3231 | k20/p0.95/t1.0 | 67.9555 | 51.8508 | 12.89% | 5.86% | 2.5938 | 完成 |
| exp1 | Qwen3.5-0.8B，10 epochs | 0.2980 | k20/p0.95/t1.0 | 69.1700 | 55.0429 | 17.38% | 9.18% | 3.0332 | 完成 |
| exp1 | Qwen3.5-2B，10 epochs | 0.2702 | k20/p0.95/t1.0 | 69.8198 | 56.7406 | 22.66% | 12.89% | 3.3203 | 完成 |
| exp1_1 | Qwen3.5-2B，20 epochs，BS4/GA8，LoRA 32/64，LR 1e-5，constant-with-warmup | 0.3800 | k20/p0.95/t1.0 | 67.9795 | 51.8816 | 11.33% | 5.86% | 2.5859 | 完成 |
| exp1_1 | 同一 adapter，仅修改 sampling | 0.3800 | k50(default)/p0.9/t0.95 | 68.3941 | 52.3675 | 14.26% | 5.08% | 2.6836 | 完成 |

Qwen3.5-2B exp1 是当前 VAL 过拟合实验中表现最好的配置；exp1_1 延长 epoch、增大 LoRA/有效 batch 并降低学习率后没有提升。由于训练集和评测集相同，这部分结果仅用于配置调试。

## Unconditional Generation 结果

本节记录固定 a prompt、stop_after_newlines=199、每模型生成 2,048 条样本的 原始BrickNet官方 无条件 PT 复现结果。
本地 PT-exp0 / PT-exp1 的 MM-PT adapter 训练时接收图像和 inventory 条件，不能使用固定 a prompt 评测；正确协议为 VAL image + empty caption + inventory → path。对应命令只见 [record.md](record.md)。
本地 PT-exp2 的 stage1 是使用BrickNet-PT text-only训练的adapter，使用unconditional generation的固定 a prompt 评测。
本表仅用于相同生成/评分协议下的描述性 system comparison；模型家族/规模、语料和训练预算不同，因此不是单因素 ablation，也不支持显著性比较结论。

| Exp | Model / Adapter | Parsable | Clean | Collision | 状态 |
| --- | --- | ---: | ---: | ---: | --- |
| BrickNet-PT | Qwen3-0.6B + BrickNet-0.6B-PT | 85.89% | 1.81% | 15.1333 | 完成 |
| BrickNet-PT | Qwen3-1.7B + BrickNet-1.7B-PT | 89.40% | 2.10% | 16.2275 | 完成 |
| BrickNet-PT | Qwen3-4B + BrickNet-4B-PT | 92.43% | 1.32% | 16.1934 | 完成 |
| BrickNet-PT | Qwen3-8B + BrickNet-8B-PT | - | - | - | 启动过但无输出 |
| BrickNet-PT | Qwen3-14B + BrickNet-14B-PT | - | - | - | 启动过但无输出 |
| PT-exp0 | Qwen3.5-0.8B + 本地 MM-PT adapter | 0.00% | 0.00% | 0.0000 | 固定 a 输入协议不匹配，结果无效 |
| PT-exp0 | Qwen3.5-0.8B + 本地 MM-PT adapter | 60.55% | 15.23% | 5.2188 | MM-PT matched protocol，512 VAL 完成；完整指标见 Condition Generation 主表 |
| PT-exp1 | Qwen3.5-0.8B + 本地 MM-PT adapter | 68.55% | 16.80% | 5.9551 | MM-PT matched protocol，512 VAL 完成；完整指标见 Condition Generation 主表 |
| PT-exp2-text8m | Qwen3.5-0.8B + 本地 text8m-PT adapter | - | - | - | 任务已实现；等待人工确认 250k final adapter 路径，未启动推理 |


## 待完成实验

待完成路线、依赖关系和 gate 统一见 [Constructor Plan.md](../BrickNet/BrickNet-MM%20Agentic%20LEGO%20Planner/Constructor%20Plan.md)。
