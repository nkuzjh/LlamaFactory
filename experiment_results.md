# 实验进度与结果汇总

更新时间：2026-08-06

本文档统一记录 LlamaFactory 的 PT/SFT 实验和 ms-swift 的 GRPO 实验。具体训练、推理
命令仍分别保留在 [LlamaFactory record](record.md) 和
[ms-swift record](../ms-swift/record.md) 中。

## 统计口径

- 后续统一对比实验均在 512 条 `BrickNet-MM-VAL` 上调参、选择搜索预算并报告指标。已有 non-thinking 实验使用
  `max_new_tokens=4096`、`top_k=20`、`top_p=0.95`、`temperature=1.0`；阶段 2 Thinking-Hard 保持同一
  sampling，但为完整 trace 使用 `max_new_tokens=16384`。
- 该集合会被反复观察，因此结果属于 repeated-use VAL，不是未触碰的 locked test 或无偏泛化估计。
- `Train loss` 是整个训练过程的平均 next-token cross-entropy loss，不是最后一步 loss。
- GRPO 的 `Reward` 是五个 `[0, 1]` 分量的加权和：
  `0.2×ParsePrefix + 0.2×InventoryF1 + 0.1×Length + 0.2×CollisionPrefix
  + 0.3×PoseMatch`。训练日志结果为 1,000 个 optimizer step 的均值。
- `exp0`、`exp1`、`exp1_1` 直接使用 BrickNet-MM-VAL 训练，只用于验证训练/推理链路和
  过拟合能力，不能代表泛化性能。
- 图文指标仅在成功解析、转换为 LDR 并完成八视图渲染的样本上计算；`Adj.` 指标将未成功
  渲染的样本按 0 分计入固定的 512 条分母。

## 实验配置与进度

已有 non-thinking BrickNet-MM 实验均使用 `Qwen/Qwen3.5-0.8B`、`qwen3_5_nothink`、
`cutoff_len=4096`、LoRA target `all`、LoRA rank/alpha `64/128`。PT/SFT 使用 batch
size 2、gradient accumulation 8、learning rate `5e-5` 和 cosine scheduler。

阶段 2 Thinking-Hard 是明确例外：用户于 2026-08-06 冻结 `qwen3_5_nothink`、`enable_thinking=false`、
`cutoff_len=16384`、`packing=false`、`train_on_prompt=false`，推理 `max_new_tokens=16384`。显式 `<think>` 是 assistant
监督文本，不叠加 Qwen 原生 thinking 模板。无思考对照统一命名为 `NonThinking`。该阶段尚未启动。
训练/预测配置、安全 dry-run launcher、extractor 和 10k/50k/all nested manifest 已准备。overfit 协议为
直接用官方 VAL 中 collision-free 的 511 条训练，并在原始完整 512 条上推理；两个 token gate 均已通过。
当前只等待阶段 0 冻结公共初始化和用户 gate。完整准备状态见
[Stage 2 runbook](bricknet-stage2-thinking-hard.md)。

| Exp | 框架 | Train output | 初始化 | 数据 | 样本数 | Epoch | 主要 ablation | Train loss | 状态 |
| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |
| PT-exp0 | LlamaFactory | `train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64` | Qwen3.5-0.8B | BrickNet-MM-PT | 135,051 | 3 | MM-PT：图像+inventory → path | 0.1507 | 完成并评测 |
| PT-exp1 | LlamaFactory | `train_PT_exp1_qwen35_08b_bricknet_text270k_mmpt135k_ep1_bs2_ga8_lora64` | Qwen3.5-0.8B | BrickNet text PT + BrickNet-MM-PT | 405,153 | 1 | 固定曝光预算的 text+MM mixed PT | - | 运行中；2026-08-05 22:33核对时最新`checkpoint-20000`，总25,323 steps |
| exp2 | LlamaFactory | `train_exp2_qwen35_08b_sft1w_ep3_bs2_ga8_lora64` | Qwen3.5-0.8B | BrickNet-MM-SFT | 10,000 | 3 | 无 PT，小规模 SFT | 0.2418 | 完成并评测 |
| exp2_1 | LlamaFactory | `train_exp2_1_qwen35_08b_sft5w_ep3_bs2_ga8_lora64` | Qwen3.5-0.8B | BrickNet-MM-SFT | 50,000 | 3 | 无 PT，扩大 SFT 数据量 | 0.2031 | 完成并评测 |
| exp2_2 | LlamaFactory | `train_exp2_2_qwen35_08b_sft_ep3_bs2_ga8_lora64` | Qwen3.5-0.8B | BrickNet-MM-SFT | 334,355 | 3 | 无 PT，全量 SFT | - | 中断于 20,660/62,694；可恢复 `checkpoint-20000` |
| exp3 | LlamaFactory | `train_exp3_qwen35_08b_pt_sft1w_ep3_bs2_ga8_lora64` | PT-exp0 adapter | BrickNet-MM-SFT | 10,000 | 3 | PT 初始化后新建 SFT adapter | 0.1701 | 完成并评测 |
| exp3_0_1 | LlamaFactory | `train_exp3_0_1_qwen35_08b_pt_sft1w_ep10_bs2_ga8_lora64` | PT-exp0 adapter | BrickNet-MM-SFT | 10,000 | 10 | exp3 的 epoch ablation | 0.1090 | 完成并评测 |
| GRPO-exp0 | ms-swift | `../ms-swift/output/bricknet_grpo/exp0_qwen35_08b_exp3_rl_n2000_g8` | PT-exp0 merged + exp3 adapter | BrickNet-MM-RL | 2,000 | 1 | GRPO，五项结构/几何 reward，G=8 | - | 完成并评测 |
| exp3_1 | LlamaFactory | `train_exp3_1_qwen35_08b_pt_sft5w_ep3_bs2_ga8_lora64` | PT-exp0 adapter | BrickNet-MM-SFT | 50,000 | 3 | PT + 50k SFT | 0.1673 | 完成并评测 |
| exp3_2 | LlamaFactory | `train_exp3_2_qwen35_08b_pt_sft_ep3_bs2_ga8_lora64` | PT-exp0 adapter | BrickNet-MM-SFT | 334,355 | 3 | PT + 全量 SFT | - | 未开始 |

PT-exp0 虽然命名为 PT，但在 LlamaFactory 中使用 `stage=sft` 和 BrickNet-MM-PT，属于
多模态监督预训练式训练，不等同于原始 BrickNet 使用固定 `"a"` prompt 的无条件
text-only PT。

## GRPO 在线训练结果

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
| exp2 | MM-SFT-10k | Qwen3.5-0.8B-SFT | lr=5e-5, ep=3; eval p=.95, k=20, t=1 | 237 (46.29%) | 4.4941 | 0.2814 | 0.7933 | 0.7570 | 68 (13.28%) | 90.2491 | 94.9980 | 65.6216 | 54.8258 | 0.1303 | 0.3672 | 0.3504 | 0.6671 | 0.7216 | 0.6612 | 0.4789 | 0.1233 | 0.4766 | 7 (1.37%) |
| exp2_1 | MM-SFT-50k | Qwen3.5-0.8B-SFT | lr=5e-5, ep=3; eval p=.95, k=20, t=1 | 354 (69.14%) | 5.6426 | 0.2824 | 0.7996 | 0.7522 | 85 (16.60%) | 91.6652 | 95.7064 | 66.2222 | 55.2598 | 0.1952 | 0.5529 | 0.5201 | 0.8413 | 0.8716 | 0.8372 | 0.4487 | 0.1497 | 0.5609 | 17 (3.32%) |
| exp3_1 | MM-PT + SFT-50k | Qwen3.5-0.8B-PT-SFT | lr=5e-5, ep=3; eval p=.95, k=20, t=1 | 367 (71.68%) | 6.0332 | 0.2854 | 0.8190 | 0.7666 | 93 (18.16%) | 92.1417 | 96.0075 | 66.6486 | 56.0046 | 0.2046 | 0.5870 | 0.5495 | 0.8641 | 0.8932 | 0.8609 | 0.4521 | 0.1617 | 0.5765 | 20 (3.91%) |
| exp3 | MM-PT + SFT-10k | Qwen3.5-0.8B-PT-SFT | lr=5e-5, ep=3; eval p=.95, k=20, t=1 | 340 (66.41%) | 5.6895 | 0.2830 | 0.8218 | 0.7662 | 85 (16.60%) | 91.5485 | 95.5846 | 66.0981 | 55.2628 | 0.1880 | 0.5457 | 0.5088 | 0.8289 | 0.8608 | 0.8247 | 0.4702 | 0.1501 | 0.5595 | 17 (3.32%) |
| exp3_0_1 | MM-PT + SFT-10k | Qwen3.5-0.8B-PT-SFT | lr=5e-5, ep=10; eval p=.95, k=20, t=1 | 267 (52.15%) | 5.1797 | 0.2833 | 0.8225 | 0.7657 | 75 (14.65%) | 91.3399 | 95.4301 | 66.2890 | 56.0601 | 0.1477 | 0.4289 | 0.3993 | 0.7335 | 0.7851 | 0.7299 | 0.4827 | 0.1463 | 0.5171 | 18 (3.52%) |
| GRPO-exp0 | MM-PT + SFT-10k + RL-2k | Qwen3.5-0.8B-PT-SFT-RL | GRPO lr=5e-6, ep=1, G=8, t_train=.9, p_train=1; eval p=.95, k=20, t=1 | 324 (63.28%) | 5.8086 | 0.2831 | 0.8182 | 0.7549 | 78 (15.23%) | 91.6202 | 95.6808 | 66.2536 | 55.7319 | 0.1791 | 0.5178 | 0.4777 | 0.8301 | 0.8625 | 0.8262 | 0.4630 | 0.1486 | 0.5583 | 14 (2.73%) |

### 指标含义与计算

- `Connectivity (Num, %)`：完整解析的预测数及其占 512 条预测的比例，即
  `invalid is None`。path 每个动作向同一棵树添加一个零件，因此完整解析也表示生成结构
  保持连接。
- `Collision`：每条预测在首次解析失败或首次几何碰撞前成功执行的动作数，再对 512 条
  取平均。设总动作数、解析失败位置、碰撞位置集合分别为 `A/I/C`，单样本值为
  `min(A, I, min(C))`，不存在的失败项忽略；完整可解析且无碰撞时取 `A`。越高越好。
- `PE`：用 `PE-Core-bigG-14-448` 编码 caption 和八个渲染视角，计算归一化 embedding
  余弦相似度；每个样本取八视图最大值，再对成功渲染样本求均值。
- `SigLIP 2`：用 SigLIP2 giant 计算图文 `sigmoid(logits_per_image)`；每个样本取八视图
  最大值，再对成功渲染样本求均值。
- `VQAScore`：向 `Qwen2-VL-7B-Instruct` 提问 `Is this LEGO set {caption}?`，以首个
  生成 token 为 `Yes` 的概率计分；每个样本取八视图最大值，再求均值。
- `Clean (Num, %)`：完整可解析且整个增量放置过程中无碰撞的预测数和比例，即
  `invalid is None and collisions is empty`。
- `BLEU-4`：生成 path 与参考 path 的字符级 sentence BLEU-4，使用 NLTK smoothing
  method 3；逐样本乘 100 后对 512 条求平均。
- `ROUGE-1/2/L`：生成与参考 path 经 jieba 分词后计算 unigram、bigram 和最长公共
  子序列 F1；逐样本乘 100 后求平均。
- `PE/SigLIP 2/VQA Adj.`：coverage-adjusted mean-max，即对应图文指标乘成功渲染数再
  除以 512。BrickNet 原始行由其表中四舍五入值推算，末位可能有舍入误差。
- `Parse Prefix`：完整解析得 1；否则为成功解析的预测零件数除以 target 零件数，上限 1。
- `Inventory F1`：按 `(part_id, color_code)` 比较预测与 target 多重集合。交集计数为
  `O`、预测和 target 零件数为 `P/T` 时，计算 `2O/(P+T)`。
- `Length Score`：`min(P,T)/max(P,T)`；两者均为空时为 1。
- `Collision Prefix`：无碰撞得 1；否则为首次碰撞动作位置除以 target 零件数，上限 1。
  它是归一化 reward 分量，不等同于上面的绝对动作数 `Collision`。
- `Pose Match`：按共享 `(part_id, color_code)` 锚点做全局刚体对齐后，平移误差不超过
  0.5、旋转误差不超过 5° 的 target 零件比例。
- `Dense Reward`：五项 GRPO reward 分量的加权和，公式见“统计口径”。
- `Strict Success`：完整解析、长度相同、inventory 完全一致、无碰撞且
  `Pose Match=1` 的样本数和比例。

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
- BLEU/ROUGE 较高不代表路径结构合法；当前主要瓶颈仍是完整可解析率和无碰撞率。

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
`VAL image + empty caption + inventory -> path`：使用
`scripts/prepare_bricknet_mm_pt_eval.py` 从 held-out `BrickNet-MM-VAL` 生成
`BrickNet-MM-PT-VAL`，再执行：

```bash
python scripts/prepare_bricknet_mm_pt_eval.py --overwrite
llamafactory-cli train examples/train_lora/qwen35_08b_bricknet_mm_pt_predict.yaml
```

GPU 可用后可先将参数覆盖为单样本、512-token smoke test：

```bash
llamafactory-cli train examples/train_lora/qwen35_08b_bricknet_mm_pt_predict.yaml \
  max_samples=1 max_new_tokens=512 \
  output_dir=saves/Qwen3.5-0.8B-Thinking/lora/eval_PT_exp0_ptval_smoke
```

完整生成结束后，运行结构、八视图和图文指标：

```bash
cd ../BrickNet
/home/jiahao/miniconda3/envs/bricknet/bin/python scripts/evaluate_experiment.py \
  --predictions ../LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20/generated_predictions.jsonl \
  --text-metrics ../LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20/predict_results.json \
  --output-dir outputs_val/qwen35_08b/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20 \
  --prompts-file data/bricknet_datasets/captions_val.jsonl
```

随后使用 ms-swift 的 BrickNet-MM `alignment-worker` 对同一 512 条 prediction/reference
补算 inventory、长度、pose、dense reward 和 strict success。完整产物位于
`../BrickNet/outputs_val/qwen35_08b/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20/metrics.json`
和同目录 `metrics.md`；310 个可解析结构均成功完成 LDR 转换和八视图渲染，无额外失败。

该任务不使用原始 BrickNet text-only PT 的固定 `"a"` prompt，也不使用
`stop_after_newlines=199`；生成应由 image/inventory 条件和模型 EOS 决定终止位置。

## 待完成实验

1. 从 `exp2_2/checkpoint-20000` 恢复并完成全量 Direct SFT，再用统一生成参数评测。
2. 完成 mixed text+MM PT-exp1；冻结 checkpoint/config/data hash，作为后续 non-thinking/Thinking 共同初始化候选。
3. 根据 exp2_2、exp3_1 和 mixed PT-exp1 的趋势决定是否启动全量 SFT。
4. 默认不启用 policy-specific hard 数据；条件性恢复时新增实验编号、记录 mining 配置并限制为 2,000 prompts。
