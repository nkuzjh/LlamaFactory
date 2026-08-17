# BrickNet PT-exp2 Runbook

状态（2026-08-12）：代码、dataset registry、PT/MM/downstream YAML 和 gate-protected launcher 已准备；
MM 使用 e1→e2→e3 三次顺序 1-epoch 训练，三份推理 YAML 分别读取对应 final adapter。三组 MM 数据已通过
完整 processor token gate，训练未开始。用户已冻结 7,698,261 条 first-round exact-dedup
path 为本机规范；31 个既有 shard 已经完整源重扫、逐行对比、hash 和原子提升，未重写 34 GiB 数据。
seed-0 first-round PT-loss VAL1000 已生成；全量 7,698,261 条 parse 为 0 error。确定性 10k collision replay
实查 10,000/10,000、发现 94（0.94%），现作为 provenance-only 记录，`audit.eligible=true`，31-shard 训练
视图已创建。

总设计与证据见
[BrickNet PT-exp2 说明](../BrickNet/BrickNet-MM%20Agentic%20LEGO%20Planner/PT-exp2%20Pretraining%20and%20Downstream%20Plan.md)。

## 固定实验序列

| 名称 | 作用 | 配置/输出 |
| --- | --- | --- |
| `PT-exp2-text8m` | 7,698,261 path、non-packing、250k-step PT | `qwen35_08b_bricknet_pt_exp2_text8m.yaml` / `train_PT_exp2_text8m_qwen35_08b_path7698261_steps250k_bs4_gbs32_lora64_len6401_nopack` |
| `PT-exp2-mm-e1` | text8m 后第一轮 MM consolidation | `qwen35_08b_bricknet_pt_exp2_mm_e1.yaml` / `train_PT_exp2_mm_e1_..._ep1_...` |
| `PT-exp2-mm-e2` | 从 e1 adapter 继续的第二轮 | `qwen35_08b_bricknet_pt_exp2_mm_e2.yaml` / `train_PT_exp2_mm_e2_..._ep1_...` |
| `PT-exp2-mm-e3` | 从 e2 adapter 继续的第三轮 | `qwen35_08b_bricknet_pt_exp2_mm_e3.yaml` / `train_PT_exp2_mm_e3_..._ep1_...` |
| `PT-exp2` | e1/e2/e3 VAL512 全指标选出的只读 alias | `saves/Qwen3.5-0.8B-Thinking/lora/PT-exp2` |
| `exp4_4` | PT-exp2 + NonThinking-Control 10k | `qwen35_08b_bricknet_stage2_exp4_4_nonthinking_control_10k_pt_exp2.yaml` |
| `exp4_5` | 10k gate 后的独立 50k SFT | `qwen35_08b_bricknet_stage2_exp4_5_nonthinking_control_50k_pt_exp2.yaml` |
| `exp4_6` | 50k gate 后的独立 all-66,456 SFT | `qwen35_08b_bricknet_stage2_exp4_6_nonthinking_control_all_pt_exp2.yaml` |

没有 PT-exp2 VAL511 训练或验证配置；所有 prediction YAML 均对完整 VAL512。

## 数据状态

`data/bricknet_pt_exp2/mm/manifest.json` 已通过。历史构造结果为：

- e1/e2/e3 总行数 `150,668 / 150,637 / 150,718`；
- 每轮 MM 行数固定 `135,051`；
- replay 行数 `15,617 / 15,586 / 15,667`，三轮 ID 互不重叠；
- replay/MM assistant target-token ratio `1.0000053 / 1.0000294 / 0.9999832`；
- 数据 SHA-256 `ce3a5185...815c3928 / 4c1df1d6...92d911a9 / acfea4cc...9b769a4`；text replay 显式
  `images=[]`，不依赖 Arrow 对缺失 media column 的隐式填充。
- 三组真实 Qwen processor audit 均为 `errors=0`、`truncated=0`、`training_eligible=true`；raw total
  p50/p95/p99/max 为 e1 `776/2294/2743/4183`、e2 `776/2295/2737/4183`、e3 `776/2295/2738/4183`。

active registry 分别保留 `BrickNet-PT-exp2-mm-e1/e2/e3`。训练按 e1→e2→e3 顺序执行，每轮都含完整
135,051 条 MM，并依次消费 hash 排序后的不重叠 replay slice：`15,617/15,586/15,667` 条，对应 target tokens
`26,285,287/26,285,922/26,284,707` 和 ratio `1.0000053/1.0000294/0.9999832`。e1 从 text8m adapter
以 LR `2e-5` 训练，e2/e3 从前一轮 adapter 以 LR `1e-5` 继续；三次均为独立 1-epoch cosine schedule。

`data/bricknet_pt_exp2/text8m` 已冻结并具备 `audit.eligible=true`。第一次 all-rounds 误扫已保留为
`text8m.superseded-all-rounds-v0/`（36 个 shard、约 30 GiB），不得注册为训练集。官方只读分支 commit `1499f66`
的 `scripts/train.py:first_round_rows()` 明确作者只用每个文件首行 round；按这一口径的完整扫描得到
`6,515,749 + 1,182,789 = 7,698,538` 行，exact-dedup 后为 `7,698,261`。发布的 8,092,423 仅作 provenance，
不再是 blocker，也不从后续轮次补齐。`finalize-existing` 完整重扫 38,485,631 个源行，将每个 unique path 与
31 shard 逐行比较后原子提升；ordered corpus SHA-256=`985b8473...07d0ab6`，shard-set SHA-256=
`aaaa26bf...354fb3f`。旧 count-failure marker 已归档。`text8m_train` 已在全量 parse 通过后创建。

277 条 exact duplicate 全部来自 pool 合并边界：`paths_pt` 首轮 6,515,749 条均为 first occurrence，
`paths_sft` 首轮 1,182,789 条中有 277 条已在 PT 出现。论文说明 PT/SFT 是 overlapping sets，故这不是单个
split 内 dedup 失败。`DATA.md` 的 sampled-path 段描述 each-split dedup，training recipe 又描述 PT+SFT 合并去重；
实际 `train.py:pt_paths()` 只拼接两个 `first_round_rows()`，不执行后者。PT-exp2 因此保留官方代码的首轮选择，
并按文字 recipe 删除 277 个跨池后出现副本。

实际审计全量 parse 通过，但 10k replay 失败 94。manifest 保存的前 20 个明细中，18 个能定位回 first-round
`paths_pt`，2 个定位回 first-round `paths_sft`，其中 4 个还是 complete-component walk。官方 DATA 对 path 的
collision-free 描述与当前 release checker + 21,084 inset mesh 的结果冲突。论文说明 PT path 在采样时执行
collision detection，官方 sampler 调用 `sample_collision_free_tree()`；官方训练代码则直接读取发布 path，未在
训练加载阶段重新 parse/collision 筛除。因此用户已批准只记录本机 revision 差异并放行，不删除这些 path；
前 20 个 path hash/collision index 与准确发现总数位于 `text8m/manifest.json:audit`。

10k 是本地工程审计预算，不是论文/官方 recipe 的数值。审计扫描全量 corpus，以规范 path text 的 64-bit
BLAKE2b hash 选择 score 最小的 10,000 条，再调用
`parse_sample() → score.check_tree() → collision.check_placements()` 重放；不是文件前 10k，也不调用
`sample_collision_free_tree()`。94 表示 94 条 path 至少有一个碰撞 action，不是 collision action 总数。
`sample_collision_free_tree()` 属于 Graph→新 path 的采样期 candidate-rejection；本地审计属于发布
path→当前 checker/mesh 的事后 replay。缺少原生成时 generator/catalog/checker/mesh revision 绑定，不能把 94
唯一归因为 sampler、数据、catalog、量化/解码、mesh 或 checker 中的某一个。

text PT 配置为 Qwen3.5-0.8B、path+EOS full loss、`packing=false`、`cutoff_len=6401`、LoRA 64/128（仅
q/k/v/o/gate/up/down）、250k steps、每卡 micro=4、单/双卡 GA=`8/4`、LR `5e-5`、warmup 12,500、
minimum-LR ratio 0.01。
1 epoch=`240,571` steps；250k=`1.03919425` nominal epoch，第二轮 9,429 steps，名义曝光 8,000,000。
独立 `BrickNet-PT-exp2-text-val1000` 只评估 PT loss，不属于 VAL511 实验。

## 安全启动

PT-exp2 训练支持单机单卡或双卡自适应。`--gpus 0` 使用单进程；`--gpus 0 1` 会设置两卡可见，并由 launcher
显式注入 `FORCE_TORCHRUN=1`、`NPROC_PER_NODE=2`、`NNODES=1`。单卡时 text8m/MM/downstream GA 为
`8/8/16`，双卡时为 `4/4/8`；每卡 BS 固定为 `4/2/1`，global batch 始终为 32/16/16，不改变 steps、
epochs 或 LR schedule。输出目录以 `gbs32/gbs16` 命名，使两种模式共用同一 adapter 链。所有训练 YAML 显式设置
`ddp_find_unused_parameters=false`。launcher 默认只报告检查和命令：

text8m 后续恢复配置为
`.llamafactory_cache/tokenized_dataset/PT-exp2-text7698261-qwen35-08b-len6401-nopack-with-length`，并声明
`length_column_name: length`。2026-08-13 检查发现该目录被普通 tokenization 流程重新生成，当前只有
`input_ids/attention_mask`，因此 launcher 会用 `TOKENIZED_LENGTH_CACHE_MISSING_OR_INVALID` 阻断，而不会回退为
逐条扫描 7,698,261 条 `input_ids`。应先迁移到一个新的目标目录并更新 YAML，不能把目录名中的
`with-length` 当作缓存有效性的依据。

通用 PT/SFT 预构建工具为 `scripts/build_tokenized_cache_with_length.py`。新实验 YAML 同时配置
`tokenized_path`、`train_sampling_strategy: group_by_length`、`length_column_name: length` 后执行：

```bash
conda run -n llamafactory --no-capture-output python \
  scripts/build_tokenized_cache_with_length.py --config examples/train_lora/<experiment>.yaml
```

若目标 cache 不存在，脚本复用 LlamaFactory 原生 PT/SFT tokenizer/template/processor 流程首次构建；若迁移
已有 cache，则追加 `--source-cache <old-cache> --output-cache <new-cache>`。输出经临时目录构建、
行数/schema/抽样 length 校验后才原子提升，已有无效目标不会被覆盖。

YAML 同时控制 launcher 的缓存策略：存在 `length_column_name` 即为 `with_length` 模式；未配置该字段即为
`standard` 模式。执行 `--action train --execute` 时，with-length 目标不存在则 launcher 先以单进程调用上述脚本，
构建完成并复检后才启动单卡或 DDP 训练；有效目标直接复用；目标已存在但缺少该列则阻断并要求显式迁移。
dry-run 只报告 `build_required` 和预构建命令，不修改缓存。自动构建默认使用 4 个进程，可用
`--cache-num-proc` 和 `--cache-batch-size` 调整。

`datasets==4.0.0` 的 `dataset["length"]` 返回懒加载 `Column`，Transformers 原生 `LengthGroupedSampler` 对它进行
随机标量读取会使 7,698,261 行的 epoch 索引排序长时间停在 `0/250000`。PT/SFT trainer 现仅在
`group_by_length` 且 cache 确实含配置的 length 列时，从 Arrow 一次性转换为连续 NumPy 数组后构造 sampler；
当前 length 数组约 `29.37 MiB`，转换 `0.061s`，全量 epoch-0 分组索引测试约 `4.384s`。没有 length 的普通
cache、非 grouped 策略和 `disable_shuffling` 均继续使用原有路径，不受该兼容影响。修改前已经启动的 Python
进程不会热加载此兼容，必须重启训练进程后才生效。

下列正式序列仍以本机双卡为默认。若只使用一张卡，将任一训练命令中的 `--gpus 0 1` 改为 `--gpus 0`；
launcher 会自动选择单卡 GA。例如：

```bash
python scripts/launch_bricknet_pt_exp2.py --gpus 0 --action train --run text8m
```

```bash
python scripts/launch_bricknet_pt_exp2.py --gpus 0 1 --action train --run text8m
python scripts/launch_bricknet_pt_exp2.py --gpus 0 1 --action train --run mm-e1
python scripts/launch_bricknet_pt_exp2.py --gpus 0 --action predict --run mm-e1
python scripts/launch_bricknet_pt_exp2.py --gpus 0 --action evaluate --run mm-e1
python scripts/launch_bricknet_pt_exp2.py --gpus 0 1 --action train --run mm-e2
python scripts/launch_bricknet_pt_exp2.py --gpus 0 --action predict --run mm-e2
python scripts/launch_bricknet_pt_exp2.py --gpus 0 --action evaluate --run mm-e2
python scripts/launch_bricknet_pt_exp2.py --gpus 0 1 --action train --run mm-e3
python scripts/launch_bricknet_pt_exp2.py --gpus 0 --action predict --run mm-e3
python scripts/launch_bricknet_pt_exp2.py --gpus 0 --action evaluate --run mm-e3
python scripts/launch_bricknet_pt_exp2.py --action select-final --run mm-e3
python scripts/launch_bricknet_pt_exp2.py --gpus 0 1 --action train --run exp4_4
```

所有训练/推理/评测实际执行都需 `--execute`。最终 alias 与 scale gate 还需 `--approve`。launcher 检查：

- corpus manifest、exact count、parse audit eligible 和 shard-only view；
- YAML 含 `length_column_name` 时，cache 不存在可在 `--execute` 中自动预构建；已有 cache 必须在
  `tokenized_path/train/dataset_info.json` 声明对应长度列；
- MM manifest、逐轮 dataset 文件、逐轮 processor audit 和前置 adapter；
- 显式选择一张或两张训练 GPU，并报告所选卡上的 compute process；GPU 占用 blocker 当前按临时决策关闭，执行前需人工确认所选卡空闲；
- 输出目录不存在，防止覆盖或误续训；
- `PT-exp2` alias、10k 数据和 50k/all 人工收益 gate。

50k 只在 `exp4_4-approved.json` 后通过 `--action materialize --run exp4_5 --execute` 物化；all 直接读取不可变
66,456 源。任何 gate 未满足均不得绕过 launcher 直接运行 YAML。

## 当前阻塞

1. 当前 GPU 可能存在其他用户任务；launcher 只报告、不阻断，正式执行前必须人工确认所选 GPU 均空闲。
2. e1 等待 text8m final adapter，e2/e3 依次等待前一轮 final adapter；`exp4_4` 等待 e1/e2/e3 全指标和
   `PT-exp2` alias。

旧 `REVIEW_TEXT8M_COUNT_7698261_VS_8092423` 已由用户决策关闭，不得再次列为 blocker。
