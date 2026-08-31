# BrickNet PT-exp2 Runbook

状态（2026-08-29 05:47:32 +08:00）：`PT-exp2-text8m` 已完成 250,000/250,000 steps，final adapter、trainer state 和
train/eval results 已验证，训练进程已退出。MM 使用 e1→e2→e3 三次顺序 1-epoch 训练，三份推理 YAML
分别读取对应 final adapter；v2 e1/e2/e3 的训练、512/512 prediction/scored/alignment 和 base+alignment
全指标评测均已完成，global/max 分别为 `9417/9417`、`9415/9415`、`9420/9420`。首次 e1 启动发现冻结 v1 JSONL 的 MM/replay `meta` struct 不同，
在 Hugging Face Arrow materialization 阶段失败，未进入 tokenization 或 optimizer；v1 数据、配置和日志已按
`invalidated` provenance 原样保留。当前活动入口改为独立 `v2-no-meta` 训练投影：逐行保持 `id/messages/images`
完全一致，只从训练视图移除未注册给模型的 `meta`。三轮 v2 全池 processor audit 均为零错误、零截断且
`training_eligible=true`，完整 LlamaFactory loader smoke 通过，v2 e1 launcher 已无协议 blocker。用户已冻结
7,698,261 条 first-round exact-dedup
path 为本机规范；31 个既有 shard 已经完整源重扫、逐行对比、hash 和原子提升，未重写 34 GiB 数据。
seed-0 first-round PT-loss VAL1000 已生成；全量 7,698,261 条 parse 为 0 error。确定性 10k collision replay
实查 10,000/10,000、发现 94（0.94%），现作为 provenance-only 记录，`audit.eligible=true`，31-shard 训练
视图已创建。原 2026-08-21 14:06 九阶段 batch 在 e3 PE 阶段因并发显存 OOM 以退出码 1 结束；e3 后续补评于
14:37 完成，14:48 完成选择，旧失败日志只保留为历史故障 provenance，不代表当前 artifact 状态。

`PT-exp2-v2` 下游中，`exp4_4` 已完成 1,875/1,875-step 训练、512/512 推理和数值评测。原串行入口随后
因通用 Stage2 evaluator 没有生成新下游 launcher 要求的 `alignment_manifest.json` 而 fail-closed 停止；
2026-08-25 evaluator/launcher 契约已修复。既有数值 artifact 未重跑，而是在标准 VAL512 reference、逐行
alignment/scored 对应、输入输出 hash 和聚合指标复算全部通过后安全回填 manifest。该文件 SHA-256=
`c8b064186ddc3fc7c4cc8aeba88712c0f5d9f10018722a8a9e7a2ab70f184ba8`，
`freeze.mode=verified_numeric_backfill`；`exp4_4` 下游 gate 现为 `already_complete/output_complete`。
`exp4_7` 已完成训练、512/512 推理和 Stage-2 评测：`global_step=max_steps=1875`、train loss=`0.10565751036008199`、
runtime=`10756.4909s`、derived tokens/s=`8377.19`、samples/s=`2.789`、steps/s=`0.174`。最终 wrapper
`/data/jiahao/task/LlamaFactory/tmp_bash/run_exp4_4_exp4_7_pt_exp2_v2_cuda1.sh` 于 `2026-08-26 03:34:07 +08:00`
记录 complete、exit status=`0`；PID=`2099550` 已退出、物理 CUDA1 空闲。两个只读 launcher 最终
`ready=true`、`already_complete=true`、无 blockers，manifest/artifact hash 契约通过，回归 `10 passed`。
训练期间仅一次非致命 allocator warning；Transformers docstring `[ERROR]` 仅为 warning，不影响 exit 0。
adapter_config SHA-256=`30898fb95cb02bb51edbd28c48f38f203825e8956bc1226e111eabceefc51ef6`，
adapter_model.safetensors SHA-256=`c8ad92be7a6091fc5f2198fc6efe8391d6e32e321d79156e16df2a294d1d69f6`。

独立 `PT-exp2-mm-rowbal-cont3` 三 epoch 训练及 ep1/ep2/ep3 prediction/evaluation 均已完成且有效；固定规则
`strict → dense → clean → parsable` 的点估计排序推荐 `checkpoint-33764`。不创建 alias 或下游绑定；完整数值与 hash
见[统一实验结果账本](experiment_results.md)。

独立于上述主链的外部 100k-step PT checkpoint 已同步，并完成 `exp4_4_1` Control 与
`exp4_7_1` Lean-State 两个 10k 下游的训练、512/512 推理和统一评测。两组均保持 hold；
该 checkpoint 与 text8m→MM-e1/e2/e3 主链相互独立，不能写成主链 final。数值和 hash 见
[统一实验结果账本](experiment_results.md)，直接命令见 [record](record.md)。

总设计与证据见
[BrickNet PT-exp2 说明](../BrickNet/BrickNet-MM%20Agentic%20LEGO%20Planner/PT-exp2%20Pretraining%20and%20Downstream%20Plan.md)。

## 固定实验序列

| 名称 | 作用 | 配置/输出 |
| --- | --- | --- |
| `PT-exp2-text8m` | 7,698,261 path、non-packing、250k-step PT | `qwen35_08b_bricknet_pt_exp2_text8m.yaml` / `train_PT_exp2_text8m_qwen35_08b_path7698261_steps250k_bs4_gbs32_lora64_len6401_nopack` |
| `PT-exp2-mm-e1 v2` | text8m 后第一轮 MM consolidation | `qwen35_08b_bricknet_pt_exp2_mm_e1_v2.yaml` / `train_PT_exp2_mm_e1_v2_nometa_..._ep1_...` |
| `PT-exp2-mm-e2 v2` | 从 v2 e1 adapter 继续的第二轮 | `qwen35_08b_bricknet_pt_exp2_mm_e2_v2.yaml` / `train_PT_exp2_mm_e2_v2_nometa_..._ep1_...` |
| `PT-exp2-mm-e3 v2` | 从 v2 e2 adapter 继续的第三轮 | `qwen35_08b_bricknet_pt_exp2_mm_e3_v2.yaml` / `train_PT_exp2_mm_e3_v2_nometa_..._ep1_...` |
| `PT-exp2-v2` | v2 e1/e2/e3 VAL512 全指标选出的只读 alias | `saves/Qwen3.5-0.8B-Thinking/lora/PT-exp2-v2` |
| `PT-exp2-mm-rowbal-cont3` | text8m 250k 后的 1:1 行数平衡 MM/text、连续三 epoch 独立系统对比 | `qwen35_08b_bricknet_pt_exp2_mm_rowbal_cont3.yaml` / `train_PT_exp2_mm_rowbal_cont3_qwen35_08b_text8m250k_mm135051_text135051_ep3_bs2_gbs16_lora64_len6400` |
| `exp4_4` | PT-exp2-v2 + NonThinking-Control 10k | `qwen35_08b_bricknet_stage2_exp4_4_nonthinking_control_10k_pt_exp2_v2.yaml` |
| `exp4_7` | PT-exp2-v2 + Stage2 V2 Lean-State 10k | `qwen35_08b_bricknet_stage2_exp4_7_thinking_hard_v2_lean_state_10k_pt_exp2_v2.yaml` |
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
- 历史三组真实 Qwen processor audit 均为 `errors=0`、`truncated=0`、`training_eligible=true`；raw total
  p50/p95/p99/max 为 e1 `776/2294/2743/4183`、e2 `776/2295/2737/4183`、e3 `776/2295/2738/4183`。
- 2026-08-20 复查曾发现活动报告路径
  `../BrickNet/outputs_preprocess/BrickNet-MM-PT-exp2/reports/token_audit_mm6400/{e1,e2,e3}` 均不存在；launcher
  当时正确 fail-closed。现已对当前数据重跑全池真实 processor audit，report SHA-256 为
  `738ec582...6e132c / 1acb1b10...886fb1 / d87c2a4e...11f65a`，长度分布与历史记录一致。

冻结 v1 全局 registry 仍保留 `BrickNet-PT-exp2-mm-e1/e2/e3`。v1 原设计按 e1→e2→e3 顺序执行，每轮都含完整
135,051 条 MM，并依次消费 hash 排序后的不重叠 replay slice：`15,617/15,586/15,667` 条，对应 target tokens
`26,285,287/26,285,922/26,284,707` 和 ratio `1.0000053/1.0000294/0.9999832`。e1 从 text8m adapter
以 LR `2e-5` 训练，e2/e3 从前一轮 adapter 以 LR `1e-5` 继续；三次均为独立 1-epoch cosine schedule。

上述 `data/bricknet_pt_exp2/mm` 与全局 registry 是冻结 v1 provenance，不再作为活动训练入口。v1 的首次 e1
执行在第 135,052 行进入 text replay 后因嵌套 `meta` schema 改变而触发 `DatasetGenerationError`；目标 adapter
目录和 tokenized cache 均未生成。修复没有原地改写 v1，也没有修改 LlamaFactory 通用 loader，而是新增：

- `data/bricknet_pt_exp2_mm_v2/manifest.v2.json`，SHA-256=
  `fb610b7857a048b8dafc30922f718bb782cbdfed0193fd78ced466edc250103c`；
- 独立 dataset registry `BrickNet-PT-exp2-mm-e1/e2/e3-v2`，不修改全局 `data/dataset_info.json`；
- v2 e1/e2/e3 文件 SHA-256=
  `6c8b082e...e3d88e / bde202d8...6f1a52 / c9559719...bc1c02`；
- ordered ID SHA-256=
  `38a538c0...47fa8 / 967834ea...b487 / 2386db7c...ba8c`，与 v1 audit 完全一致；
- `loader_validation_report.json` SHA-256=
  `8f84cc204b6e958a3ab854e1a945ac725a39509f0e3bdd039ff173cfff488634`，三份完整 JSONL 均通过真实
  LlamaFactory `get_dataset`，每份再以真实 Qwen processor tokenization 4 条；未加载模型或启动 optimizer；
- v2 全池 audit report SHA-256=
  `9adeee87...da61e / 01697ec5...1738 / b5f1824a...c6266`，三轮仍为 0 error、0 truncation，长度分布与
  v1 一致。一次错误媒体根 audit 原样保存在 `token_audit_mm6400_v2.invalid-media-root`，不被 launcher 消费。

v2 builder、loader validator、launcher、缓存、adapter、预测、评测和批处理日志均使用独立命名空间。v2 e1
只读既有 text8m adapter，并校验其权重 SHA-256=`a5ec2be5...685bd`；v2 e2/e3 只接受前一轮 v2 adapter。

### v2 收尾状态（2026-08-25）

- e1/e2/e3 训练 `global_step/max_steps` 为 `9417/9417`、`9415/9415`、`9420/9420`；每组
  `prediction/scored/alignment` 均为 `512/512`，base+alignment 完整性 gate 已通过。
- metrics SHA-256：e1=`2a6a45af06221b5b6d2ba7d4583a28ca340e98d174dba9bcfadfbd4d449e2e23`，
  e2=`da0f28c922b15cda597336b191390eb04606c5a17a69cb808390844ee4a8ec4a`，
  e3=`59e16916c0038a3bbe119aa32cfedfdd6cbea7c46e13f6e1dbb9673396bd95a1`。
- 固定排序 `strict_success_rate → dense_reward_mean → clean_rate → parsable_rate` 的 rank tuple 为：
  e1=`6/512, 0.5857273423, 0.220703125, 0.8203125`；e2=`4/512, 0.5860430454, 0.208984375, 0.794921875`；
  e3=`4/512, 0.5941138849, 0.220703125, 0.806640625`。因此 e1 胜出。
- adapter SHA-256：e1=`808129ac82542c79273a78e3ffeae5d493d3c6ff38ae5c768d112e1915dda97a`，
  e2=`cd50f4c22b9c6f50da5ff2f5c135c111cd56906154d0488f3149010e118f4a1d`，
  e3=`9ae578c8601ea8139015017eed299d8ce8257fadff15ed8f93c8d175e92672c4`。
- `data/bricknet_pt_exp2/gates/PT-exp2-v2-selection.json` 已为 `ready=true`、`executed=true`、
  `selected=recommended=mm-e1`；`saves/Qwen3.5-0.8B-Thinking/lora/PT-exp2-v2` 是只读 alias，精确指向 e1。
- 原九阶段 batch 的 e3 PE OOM 退出码 1 仅是历史失败日志；14:37 的 e3 补评和 14:48 的 selection record
  才是当前 artifact 状态。

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
q/k/v/o/gate/up/down）、250k steps。实际 final `training_args.bin` 为每卡 micro=16、单/双卡 GA=`2/1`、
global batch 32、LR `5e-5`、warmup 12,500、minimum-LR ratio 0.01；输出目录中的 `bs4` 是遗留命名。
1 epoch=`240,571` steps；250k=`1.03919425` nominal epoch，第二轮 9,429 steps，名义曝光 8,000,000。
独立 `BrickNet-PT-exp2-text-val1000` 只评估 PT loss，不属于 VAL511 实验。

最终训练结果：`global_step=250000`、epoch=`1.0391942503460516`、train loss=`0.3343376237487793`、
eval loss=`0.29681164026260376`、eval perplexity=`1.345561825904564`、input tokens=
`13,681,772,704`、runtime=`514,805.1127s`。final adapter model SHA-256=
`a5ec2be5d96a8beb54a4b53e0b1816626fae3cd2cd5da717917804204f5685bd`。

## 安全启动

PT-exp2 训练支持单机单卡或双卡自适应。`--gpus 0` 使用单进程；`--gpus 0 1` 会设置两卡可见，并由 launcher
显式注入 `FORCE_TORCHRUN=1`、`NPROC_PER_NODE=2`、`NNODES=1`。单卡时 text8m/MM/downstream GA 为
`2/8/16`，双卡时为 `1/4/8`；每卡 BS 固定为 `16/2/1`，global batch 始终为 32/16/16，不改变 steps、
epochs 或 LR schedule。输出目录以 `gbs32/gbs16` 命名，使两种模式共用同一 adapter 链。所有训练 YAML 显式设置
`ddp_find_unused_parameters=false`。launcher 默认只报告检查和命令：

text8m 使用
`.llamafactory_cache/tokenized_dataset/PT-exp2-text7698261-qwen35-08b-len6401-nopack-with-length`，并声明
`length_column_name: length`。2026-08-20 final 复查显示该 cache 为 7,698,261 行，列为
`input_ids/attention_mask/length`，launcher 报 `eligible=true/build_required=false`；训练已实际完成。

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
python scripts/launch_bricknet_pt_exp2_mm_v2.py --gpus 1 --action train --run mm-e1
python scripts/launch_bricknet_pt_exp2_mm_v2.py --gpus 1 --action predict --run mm-e1
python scripts/launch_bricknet_pt_exp2_mm_v2.py --gpus 1 --action evaluate --run mm-e1
python scripts/launch_bricknet_pt_exp2_mm_v2.py --gpus 1 --action train --run mm-e2
python scripts/launch_bricknet_pt_exp2_mm_v2.py --gpus 1 --action predict --run mm-e2
python scripts/launch_bricknet_pt_exp2_mm_v2.py --gpus 1 --action evaluate --run mm-e2
python scripts/launch_bricknet_pt_exp2_mm_v2.py --gpus 1 --action train --run mm-e3
python scripts/launch_bricknet_pt_exp2_mm_v2.py --gpus 1 --action predict --run mm-e3
python scripts/launch_bricknet_pt_exp2_mm_v2.py --gpus 1 --action evaluate --run mm-e3
python scripts/launch_bricknet_pt_exp2_mm_v2.py --action select-final --run mm-e3
python scripts/launch_bricknet_pt_exp2_v2_downstream.py \
  --run exp4_4 --action train --gpus 1 --pt-exp2-v2-approved
python scripts/launch_bricknet_pt_exp2_v2_downstream.py \
  --run exp4_7 --action train --gpus 1 --pt-exp2-v2-approved
```

九阶段 MM v2 串行批处理入口为 `tmp_bash/run_pt_exp2_mm_e1_e2_e3_v2.sh`，本次历史运行使用独立锁文件并按用户批准
将训练、推理、评测全部固定到物理 CUDA 1；单卡 MM 训练为 BS2/GA8/global batch 16。GPU 空闲等待调用已注释，
不以现有 compute process 阻断。每步仍检查 launcher gate 和至少 40 GiB 可用空间；训练/推理不完整目录会
fail-closed，同输入的评测 `status=running` manifest 可由 evaluator 原生续跑。评测 stage 只有在 base evaluator 和
alignment worker 都通过 512-row 完整性 gate 后才算完成。脚本不执行 `select-final`：

```bash
cd /data/jiahao/task/LlamaFactory
# 历史复现入口；当前 e1/e2/e3 已完成，不要据此推断当前 artifact 状态。
nohup bash tmp_bash/run_pt_exp2_mm_e1_e2_e3_v2.sh >/dev/null 2>&1 &
tail -f tmp_bash/pt_exp2_mm_e1_e2_e3_v2.nohup.log
```

每轮 `evaluate` 现在是一个不可拆分的两段流程：先由 BrickNet evaluator 生成结构、碰撞、图像和文本指标及
`scored.jsonl`，再以标准
`BrickNet/outputs_preprocess/BrickNet-MM/sharegpt/BrickNet-MM_VAL.jsonl` 运行 alignment worker，补写
`task_alignment`、Dense Reward、Strict Success 和 `condition_generation`。launcher 在 alignment 前逐行确认：
prediction、标准 VAL512 和 scored 均为 512 行；prediction 的 `label` 与同索引标准 assistant reference 一致；
scored 的 path 与 prediction 一致且 `collisions` 为列表。验证后才原子写入 `alignment_input.jsonl`。

完整性 gate 要求 `metrics.json` 同时有 512-row `structure/task_alignment/condition_generation`，并校验 Dense
Reward、Strict Success、固定 reward weights、pose tolerance、`alignment.jsonl`、`metrics.md` 和所有输入/输出
SHA-256。`alignment_manifest.json` 绑定 prediction、scored、标准 VAL512、base evaluation manifest 和 alignment
evaluator 代码；任一输入或实现发生变化都会使旧结果失效。`select-final` 只有在 e1/e2/e3 三轮均通过该 gate 后
才会按 `strict_success_rate → dense_reward_mean → clean_rate → parsable_rate` 排序。

历史 batch shell 不会热加载 shell 函数修改；本次 e3 PE 阶段曾因并发显存 OOM 以退出码 1 结束，随后使用
独立 launcher 完成 e3 补评和 selection。当前三轮的 prediction/scored/alignment 均已通过 512-row、provenance
和 hash gate；若未来复现，shell 自身仍会用同一 gate 判定是否可跳过评测。

所有训练/推理/评测实际执行都需 `--execute`。最终 alias 与 scale gate 还需 `--approve`。launcher 检查：

- corpus manifest、exact count、parse audit eligible 和 shard-only view；
- YAML 含 `length_column_name` 时，cache 不存在可在 `--execute` 中自动预构建；已有 cache 必须在
  `tokenized_path/train/dataset_info.json` 声明对应长度列；
- v1 父 hash、v2 manifest/registry/dataset hash、Arrow materialization、真实 LlamaFactory loader report、
  逐轮 v2 processor audit 和前置 adapter；
- 显式选择一张或两张训练 GPU，并报告所选卡上的 compute process；GPU 占用 blocker 当前按临时决策关闭，执行前需人工确认所选卡空闲；
- 输出目录不存在，防止覆盖或误续训；
- 评测 base+alignment 均为 512 行，且标准 reference、scored、Dense Reward、Strict Success、权重、pose tolerance
  和 provenance hash 全部一致；
- `select-final` 前 e1/e2/e3 均具备通过完整性 gate 的 `task_alignment`；
- `PT-exp2-v2` alias；下游仍保持隔离。当前 `exp4_4/exp4_7` 是已明确批准的独立 v2 配置，未来新增配置仍需另行批准。

50k 只在 `exp4_4-approved.json` 后通过 `--action materialize --run exp4_5 --execute` 物化；all 直接读取不可变
66,456 源。任何 gate 未满足均不得绕过 launcher 直接运行 YAML。

## 当前状态

1. v1 mixed-meta 首次 e1 启动已 `invalidated` 并保留；v2 Arrow、真实 loader、三轮零截断 gate 和三轮
   prediction/scored/alignment 均通过，selection record 已执行并选择 e1。
2. `PT-exp2-v2` alias 已精确指向 e1，可作为新的、显式绑定的下游 SFT 初始化；旧 `PT-exp2` YAML/alias
   和配置未被静默改绑。
3. 用户已独立批准并完成 `exp4_4/exp4_7` 准备：四份新 YAML、隔离 launcher、统一 evaluator 注册和
   CUDA1 串行 batch 均已就绪；两套 10k 与两套 VAL processor audit 均为 0 error、0 truncation。`exp4_4`
   已完成并通过完整下游 gate：train loss=`0.1477434707`、parsable=`407/512`、clean=`122/512`、
   Dense Reward=`0.6062629319`、Strict Success=`11/512`。缺失的 alignment manifest 已在不重跑数值 artifact
   的前提下严格校验并 safe-backfill；manifest hash 与 freeze mode 见本文顶部。
4. 六阶段入口于 `2026-08-25T15:54:51+08:00` 以独立 `setsid -f` 会话恢复，三个 `exp4_4` stage 均安全 no-op；
   `exp4_7` 已完成训练、512/512 推理和 Stage-2 评测。wrapper 于 `2026-08-26 03:34:07 +08:00` complete、
   exit status=`0`，PID=`2099550` 已退出、CUDA1 空闲；训练为 `global_step=max_steps=1875`、train loss=
   `0.10565751036008199`，prediction runtime=`30150.9915s`、512 rows，evaluation scored/alignment_input/alignment
   均为 512 rows，renders/PE/SigLIP2/VQA 均为 409、无 render failure。两个只读 launcher 最终
   `ready=true/already_complete=true`、无 blockers，回归 `10 passed`。`trace_format_valid=34/512`，不能声称可靠
   Lean-State 状态追踪；这是模型输出质量信号，不是 evaluator/launcher 失败。直接修复、验证和恢复命令见 `record.md`。
5. 原 e3 PE OOM 日志只作历史故障 provenance，不覆盖 14:37 补评和 14:48 selection 的当前状态。
6. 独立 `PT-exp2-mm-rowbal-cont3` 的 ep1/ep2/ep3 prediction/evaluation 均已有效完成：每轮均为 `512/512`、
   alignment manifest `status=complete`；按 `strict → dense → clean → parsable` 的点估计固定规则推荐
   `checkpoint-33764`，不创建 alias。完整数值与 hash 见统一结果账本。

旧 `REVIEW_TEXT8M_COUNT_7698261_VS_8092423` 已由用户决策关闭，不得再次列为 blocker。

## PT-exp2-mm-rowbal-cont3（独立系统对比，训练与 ep1/ep2/ep3 prediction/evaluation 完成，推荐 checkpoint-33764）

该实验从冻结的 `PT-exp2-text8m` 250k final adapter 直接开始，使用一个固定 JSONL 并在同一个
LlamaFactory Trainer 进程中连续训练 3 个 epoch。每个 epoch 重用完全相同的
`135,051` 条 MM 行和 `135,051` 条纯 text 行；纯 text 行显式使用 `images=[]`。它同时改变了
行数平衡和 optimizer/scheduler 的连续性，因此是独立 system comparison，不是对 v2 e1/e2/e3 的
单因素 ablation，也不是 compute-matched 结论；不创建 alias、不绑定下游、不作正式 winner 或显著性结论，
只报告固定规则下的点估计首选。

- 数据目录：`data/bricknet_pt_exp2_mm_rowbal_cont3/`；总行数 `270,102`，MM/text=`135,051/135,051`，
  顶层 schema 固定为 `id,images,messages`，三轮使用同一个文件。
- 数据 JSONL SHA-256=`ac55229c93c548008051f138b6587676f020d7f54c0426208c49163b8d285d84`；ordered-ID
  SHA-256=`2bc8bb9381ac397135fed382fe18abd753692948137d54d65b709a99bf4c838b`；manifest SHA-256=
  `b2d77a977ded33ba27bdd0fa3985c2a46dee1eb176500741521d06e8b73bc100`。
- 目标 token audit 为 MM=`26,285,148`、text=`227,155,208`，text/MM=`8.641960395277211`；这是行数
  平衡而非 token 数平衡。loader 和全池 processor audit 已通过：`270,102/270,102`、0 errors、0 truncation，
  media root 正确。
- 训练配置固定为 LR=`1e-5`、cosine、warmup=`3%`、BS2/GA8/global batch 16、`cutoff_len=6400`、
  `packing=false`、`train_on_prompt=false`、冻结 vision/projector、`random` sampler；保存完整状态且
  `save_strategy=epoch`。预期 epoch checkpoint 为 `checkpoint-16882`、`checkpoint-33764`、
  `checkpoint-50646`。
- 训练期间不做 prediction/evaluation。三 epoch 全部结束后，才按 ep1→ep2→ep3 使用对应 checkpoint
  依次执行同一 PT VAL512（输入/输出各 4096、sample、temperature 1、top-k 20、top-p .95、seed 42）
  推理和评测。

截至 2026-08-28 04:30 的历史状态为 `preparation complete / running / training_started=true`。标准 tokenized cache
已于 `2026-08-26 16:04 +08:00` 完成；新 launcher `prepare-cache` dryrun 为 `ready=true, blockers=[]`，实际
Arrow train split 为 `270102` 行，包含 `input_ids/attention_mask/labels/images/videos/audios/length`，
`length mismatch=0`。训练 dryrun（`--gpus 1`）为 `executed=false`，内部 gate 全部通过；在 handoff 启动前唯一
blocker 是当时的 CUDA1 计算进程 PID=`2773643`；该历史 dryrun 时训练输出目录不存在。
2026-08-26 16:32 +08:00 首次 nohup handoff PID=`2905400` 在 16:38 前异常消失，没有 EXIT 日志、锁已释放，
未触碰 CUDA1，也未启动 rowbal；旧 PID 文件已移至
`LlamaFactory/tmp_bash/run_pt_exp2_mm_rowbal_cont3_after_text250k_cuda1.pid.stale-2905400-20260826T163848`。
2026-08-26 16:39 +08:00 曾用受工具会话托管方式恢复 handoff：当时 handoff PID=`2909872`，统一 exec session=`70789`，
锁为 held，绑定上游 wrapper PID=`2773476` 及其日志字节偏移 `453661`。当时仅等待上游两组完整
train/predict/evaluate，不占用 CUDA1，rowbal `training_started=false`；两组成功、CUDA1 空闲且 evaluate dryrun
严格 complete 后才会继续 rowbal train 及 ep1/ep2/ep3 prediction/evaluate，任一 gate 失败即 fail-closed。当时训练、推理、
评测均未启动，直接命令见 `record.md`；禁止重复启动 handoff，尤其不要使用 nohup 重启。

2026-08-27 07:27:45 +08:00 上游 Text250k wrapper 已完成并以 `exit status=0` 退出，PID=`2773476` 的 pidfile 已清理。
handoff PID=`2909872` 于 07:28:01 严格验证两组 evaluate dryrun（`ready=true`、`already_complete=true`、
`checks.output_complete=true`、`blockers=[]`）后开始 `rowbal_train`；07:28:30 实际训练启动。当前参数为
`270102` examples、3 epochs、`16882` update steps/epoch、总 `50646` steps、device BS2、GA8、global16、CUDA1。
07:28:52 观测到一次 `CUDACachingAllocator allocation failed` warning，但训练继续并至少推进到 step 90；该 warning
非 fatal，不表示实验失败。2026-08-27 17:53:57 +08:00，epoch1 的 `checkpoint-16882` 已完整落盘并复核有效：
`/data/jiahao/task/LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp2_mm_rowbal_cont3_qwen35_08b_text8m250k_mm135051_text135051_ep3_bs2_gbs16_lora64_len6400/checkpoint-16882`。
六个必需文件（adapter、config、trainer state、optimizer、scheduler、RNG state）均齐全，`trainer_state` 为
`global_step=16882`、`epoch=1.0`、`max_steps=50646`；保存后训练继续。
2026-08-28 04:23:04 +08:00，epoch2 的 `checkpoint-33764` 已完整落盘并由主代理复核有效：
`/data/jiahao/task/LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp2_mm_rowbal_cont3_qwen35_08b_text8m250k_mm135051_text135051_ep3_bs2_gbs16_lora64_len6400/checkpoint-33764`。
该目录的六个必需文件均齐全且稳定可读；`trainer_state` 为 `global_step=33764`、`epoch=2.0`、
`max_steps=50646`。保存后训练继续，04:30 已到 step=`33956`；该段为训练中的历史快照。

2026-08-28 14:55:45 +08:00，连续三 epoch 训练以 `stage_complete` 成功完成：
`trainer_state.global_step=50646`、`epoch=3`、`max_steps=50646`，`train_loss=0.2540496625`、
`runtime=113205.8521s`、`train_steps_per_second=0.447`。`checkpoint-50646` 和根输出目录的 adapter
均已完整落盘并通过训练完成校验，训练状态为 `valid`。

原 handoff 随即进入 ep1，但因可用磁盘空间低于 `40 GiB` 的安全门退出（exit `1`），没有启动 prediction；
这是当时的历史状态。为解除该磁盘门，已删除明确可重建且没有活动引用的旧 cache：
`/data/jiahao/task/LlamaFactory/.llamafactory_cache/tokenized_dataset/PT-exp2-text7698261-qwen35-08b-len6401-nopack`
（`64,769,798,144` bytes）；正式使用的独立 `...nopack-with-length` cache 未删除，原始数据仍保留，该 cache 可按
既有 prepare-cache 命令重建。可用空间约由 `30.5 GiB` 恢复至 `90.7 GiB`。

删除后 ep1 dry-run 已返回 `ready=true`；随后两次恢复被物理 CUDA1 上 mingyang 的短/长任务安全门阻止，
这些均为当时的历史状态，不能终止或干扰他人进程。

### ep1 prediction/evaluation（有效完成，2026-08-28 23:58:23 +08:00）

已验证的 `checkpoint-16882` 对完整 PT VAL512 生成 `512/512` prediction，runtime=`11238.3337s`；BLEU-4=
`79.8114`、ROUGE-1/2/L=`91.6656/62.759/53.2726`，prediction SHA-256=
`ae573f7970b7cee7888ab600a232a2dee3f4b853b7caf64d091f5c51b69b2172`。base evaluation fully parsable=`404/512`、
collision-free=`105/512`、mean actions before first failure=`8.421875`（约 `8.4`）、complete render=`404/404`、
render failure=`0`。

alignment `samples=512`：parse prefix=`0.9102900774232086`、inventory F1=`0.8155414811184989`、length=
`0.809655785706696`、collision prefix=`0.5570950990310708`、pose=`0.12373974525207188`、dense reward=
`0.5746728336608469`、strict=`2/512=0.00390625`；PE=`0.27744618028697399`、SigLIP2=`0.76755156375394007`、
VQA=`0.74573100058564745`（图像指标样本均为 `404`）。evaluator exit=`0`；`metrics.json` SHA-256=
`1a5814a6820e70eaf0ccf667e316765cc4626d88195d5dc3d20c614fd6508bfc`、`alignment_manifest.json` SHA-256=
`8243d937d775a44b9f6b73a11c12401b60c6a64eb57891eba2eb1caba288da27`，status=`complete`。

ep2 prediction/evaluation 已有效完成，ep3 prediction/evaluation 也已按顺序完成并通过验证。详细结果见下方；按固定点
估计规则推荐 `checkpoint-33764`，不创建 alias。

### ep2 prediction/evaluation（有效完成，2026-08-29 02:51:07 +08:00）

已验证的 `checkpoint-33764` 对完整 PT VAL512 生成 prediction=`512/512`、exit=`0`，完成于 `2026-08-29 02:40:54 +08:00`，
runtime=`9611.6262s`；BLEU-4=`83.8182900390625`、ROUGE-1/2/L=`93.14293828125/64.2137451171875/53.681630664062496`，
prediction SHA-256=`c156da9d485a2f398478c82742eace44b2adb4d499c7228c09c580dc2f46f8b1`。
evaluation exit=`0` at `2026-08-29 02:51:07 +08:00`；fully parsable=`402/512`、clean/collision-free=`113/512`、
collision/mean actions before first failure=`8.033203125`、complete renders=`402/402`、render failures=`0`。

alignment `samples=512`：parse prefix=`0.9183373919040974`、inventory F1=`0.8566068334354298`、length=
`0.8518979466795903`、collision prefix=`0.5456691592547537`、pose=`0.13199846714201502`、dense reward=
`0.5889120117294198`、strict=`8/512=0.015625`、exact path=`3`；PE=`0.27715973355876866`、SigLIP2=
`0.7880970399771163`、VQA=`0.7456277761960504`（图像指标样本均为 `402`）。`metrics.json` SHA-256=
`950a8356ca0285cb53494f9b4a8846b90e92c73b7f12dfe4bcf187f7e0c4a17d`；`alignment_manifest.json` SHA-256=
`ec1f3aa21cad2e24523396d0b6c724890325ee9856cf74f35e4070882c1be2d5`，status=`complete`。
完成后的 evaluate dry-run 仅报告 `EVALUATION_ALREADY_COMPLETE`，这是预期 blocker，不是失败。

相对 ep1，ep2 的 BLEU/ROUGE、clean、parse prefix、inventory F1、length、pose、dense、strict、SigLIP2 均为更高的点估计
（strict `8/512` 对 `2/512`），但 fully parsable 为 `402/512` 对 `404/512`，collision prefix、mean actions、PE 和
VQA 不更高；这是事实比较，不是显著性结论。ep3 结果如下；三轮固定排序的最终推荐为 `checkpoint-33764`，不创建 alias。

### ep3 prediction/evaluation（有效完成，2026-08-29 05:47:32 +08:00）

已验证的 `checkpoint-50646` 对完整 PT VAL512 生成 prediction=`512/512`，runtime=`9785.7039s`；BLEU-4=
`84.08433886718751`、ROUGE-1/2/L=`93.0461208984375/64.09938632812501/53.8440525390625`，prediction SHA-256=
`031a7ad92cd0bee16e685e0a02270e8e4b63f3189e50abe416f9cd783991324a`，`predict_results.json` SHA-256=
`adaa00298f46a2a3bc7555d6facf2cfc09e9bbcd0c750fbba46fa1d640721d4d`。
evaluation 于 `2026-08-29 05:47:32 +08:00` 以 exit=`0` 完成；fully parsable=`405/512`、clean/collision-free=`105/512`、
skipped=`107`、complete renders=`405/405`、render failures=`0`。

alignment `samples=512`：parse prefix=`0.9130015178424298`、inventory F1=`0.8555066953271355`、length=`0.8465148582783681`、
collision prefix=`0.5459210124616339`、pose=`0.1257369235575623`、dense reward=`0.5852584080213454`、
strict=`5/512=0.009765625`、exact path=`0`；PE=`0.27697331934799385`、SigLIP2=`0.77552385447937766`、
VQA=`0.73738091385658877`（图像指标样本均为 `405`）。`metrics.json` SHA-256=
`0da1cde9c94709dbee596ddfd93a0dd0349535b01c22668d8be2d43c3328ec6e`；`alignment.jsonl` SHA-256=
`815fcd28e52a637b88947890ccc4ceaafbb9926b3649d1fb640aff448bba4805`；`alignment_manifest.json` SHA-256=
`414e2340bcbc67c740ca28d99a5d5994c4d7fcaadc4e2e6cbf86bdc91ec67d58`，status=`complete`。
VAL/对齐 ordered-ID SHA-256=`908489739b3fff8489877da01b034081f80c5db8c21f357164c7ac38e02a6e51`。

ep1/ep2/ep3 的固定选择 tuple（strict→dense→clean→parsable）为
`(2/512, 0.5746728336608469, 105, 404)`、`(8/512, 0.5889120117294198, 113, 402)`、
`(5/512, 0.5852584080213454, 105, 405)`；因此推荐 `checkpoint-33764`。这是点估计排序，不宣称显著性。

## Text250k-only downstream（complete / rowbal 三 epoch 与 ep1/ep2/ep3 完成，推荐 checkpoint-33764，2026-08-29）

新增 `exp4_4_2` NonThinking-Control 与 `exp4_7_2` Thinking-Hard V2 Lean-State 两组独立 train/predict
配置、evaluator-only 注册、fail-closed launcher 和 CUDA1 六阶段 wrapper。两组直接读取冻结的 text8m
250k adapter，使用 single-seed `42`、单卡 BS1/GA16；不读取 PT-exp2 alias，也不读取任何 MM e1/e2/e3
adapter。wrapper 已按 `exp4_4_2`→`exp4_7_2` 顺序完成，rowbal handoff 已通过上游 gate 并完成训练；rowbal ep1/ep2/ep3
prediction/evaluation 已有效完成，按固定点估计规则推荐 `checkpoint-33764`；入口和逐阶段命令见 `record.md`，wrapper
只读检查为：

```bash
cd /data/jiahao/task/LlamaFactory
bash tmp_bash/run_exp4_4_2_exp4_7_2_pt_exp2_text250k_cuda1.sh --preflight-only
```

当前状态更新（2026-08-27 07:27 +08:00）：两组均完成 train、512/512 predict 与有效 Stage-2 evaluate，
alignment manifest 均为 `complete/post_evaluation_freeze`。`exp4_4_2` 为 parsable=`409`、clean=`117`、
Dense=`0.5984927319348795`、Strict=`14/512`；`exp4_7_2` 为 parsable=`423`、clean=`130`、Dense=
`0.608682876136042`、Strict=`11/512`。Text250k wrapper exit status=`0`；尚未运行统计分析或 paired 比较，
等待用户后续指令。

## rowbal-cont3 ep3 downstream（historical attempt interrupted / queued / waiting for CUDA0，2026-08-31）

已按用户批准的实验矩阵准备两个新的独立 SFT 分支：

- `exp4_4_3`：`PT-exp2-mm-rowbal-cont3` ep3 `checkpoint-50646` + Stage2
  NonThinking-Control 10k。
- `exp4_7_3`：同一 ep3 checkpoint + Stage2 Thinking-Hard V2 Lean-State 10k。

两组复用 `_2` 的 Stage2 SFT 与 VAL512 协议：单卡 CUDA0、BS1/GA16/global16、LoRA
64/128、LR=`5e-5`、3 epochs、`cutoff_len=16384`、seed=`42`，并使用各自独立的
config/cache/train/prediction/evaluation namespace。预测 adapter 顺序仅为
`checkpoint-50646,<new SFT adapter>`；不再单独加载 text250k adapter，不使用 PT alias，
两个 SFT adapter 也不串接。

该设计是用户指定的 ep3 endpoint sensitivity comparison。rowbal 自身的固定点估计选择
仍推荐 ep2 `checkpoint-33764`；因此不将 `checkpoint-50646` 或 `_3` 写成“rowbal best”。
新分支只用于比较不同 PT 路线在相同两种 SFT 方案下的完整 PT→SFT 系统效果；不是
compute-matched 的纯 PT 因果消融。

已准备四份 YAML、专用 fail-closed launcher、evaluator-only 注册、CUDA0 六阶段 wrapper
和静态/只读测试。launcher 直接冻结 ep3 checkpoint 权重和完整状态文件、rowbal
run/data manifest、Stage2 数据与 token audit、四份 YAML 文件 hash/字段与 adapter 顺序、
VAL512 以及 alignment manifest；
训练执行额外要求 `--rowbal-ep3-approved`。按 2026-08-30 的最新执行指令，wrapper 固定
`exp4_4_3 train→predict→evaluate→exp4_7_3 train→predict→evaluate`，并有 CUDA0、
`>=40 GiB` 磁盘、lock/PID/log 和 fail-stop gate。它不会杀死已有 GPU 进程。

历史启动时状态为 `running / training_started=true`。2026-08-30 05:10:50 +08:00 的正式 preflight
exit=`0`；启动前 focused regression 为 `53 passed`。05:11:03 +08:00 曾启动
`LlamaFactory/tmp_bash/run_exp4_4_3_exp4_7_3_pt_exp2_mm_rowbal_cont3_cuda0.sh`，wrapper PID=`1588945`，
exec session 由主代理持有；当前首阶段 `exp4_4_3 train` 正在初始 token-cache 构建，尚无 step、训练指标或
ETA。启动时物理 CUDA0 为空闲，可用磁盘约 `74 GiB`。当前 prediction/evaluation 尚未启动，也没有新的
metrics；两组均完成 512-row prediction、Stage2 evaluation 和 alignment manifest 冻结前，不登记结果、
winner、alias 或 paired 统计结论。

准备阶段 focused regression 为 `52 passed`，启动前复核为 `53 passed`；Python/shell 语法、Ruff、
diff-check 以及既有只读 launcher dry-run 均已通过。此前 CUDA1 占用仅是准备阶段历史 blocker，本轮不影响
CUDA0；wrapper 不会杀死或修改已有 GPU 进程，任一门失败时停止，不继续下一阶段。

此前准备阶段的 `run_exp4_4_3_exp4_7_3_pt_exp2_mm_rowbal_cont3_cuda1.sh` 仅作为历史入口保留，
不用于本轮；本轮应使用 `run_exp4_4_3_exp4_7_3_pt_exp2_mm_rowbal_cont3_cuda0.sh`。

### 当前状态更正（2026-08-31）

上述 `2026-08-30 05:11:03` CUDA0 启动是一次早停的历史尝试：初始 tokenizer/cache 处理约为
`2500/10000`，训练仍为 `0/1875`，并于 `05:15:19` 结束。日志末尾 `exit status=0` 仅表示外层
wrapper 返回，不能证明训练完成；没有形成可用的 train adapter、完整 token-cache、prediction 或
evaluation 产物，`exp4_7_3` 也未启动。

截至 `2026-08-31`，CUDA0 由其他用户 `lingyu` 的 `PID=1908540`（`python cog5b.py`）占用；不得杀死、
暂停或干预该进程。当前状态为 `queued / waiting for CUDA0`。CUDA0 空闲后，从 `exp4_4_3 train` 重新
运行完整的 fail-stop 串行链；不恢复早停中间状态，也不把旧 PID=`1588945` 当作运行中实例。

2026-08-31 14:10 +08:00 已实际启动独立 `nohup+setsid` CUDA0 waiter：PID=`2197540`、PPID=`1`、
PGID=SID=`2197540`，日志为 `tmp_bash/wait_for_exp4_4_3_exp4_7_3_pt_exp2_mm_rowbal_cont3_cuda0.log`。
该 waiter 每 60 秒只读检查，不杀死或干预任何进程；只有 CUDA0 空闲、可用磁盘至少 `40 GiB` 且
preflight 通过后，才会 `exec` 原完整串行 wrapper。
