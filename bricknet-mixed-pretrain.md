# BrickNet 与 BrickNet-MM 混合预训练

## 现有 BrickNet-MM PT 实验

已有实验输出为：

`saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64`

训练配置和结果如下：

| 项目 | 配置或结果 |
| --- | --- |
| Base model | `Qwen/Qwen3.5-0.8B` |
| Dataset | `BrickNet-MM-PT` |
| 数据量 | 135,051 |
| Stage | `sft` |
| Template | `qwen3_5_nothink`，`enable_thinking=false` |
| 输入 | 8 视角拼图、空 caption、完整 parts inventory |
| 目标 | 一条完整 BrickNet path text |
| Loss | assistant path token 上的 next-token CE |
| LoRA | rank/alpha `64/128`，target `all` |
| 视觉模块 | vision tower 和 multimodal projector 均冻结 |
| 训练 | batch size 2，GA 8，LR `5e-5`，cosine，3 epochs |
| 结果 | 25,323 steps，420,349,552 input tokens，train loss `0.1507` |
| 运行时间 | 296,545.8 秒，约 82.4 小时 |

数据构造报告显示：

- 原始 `pt.npz` 有 253,623 个 graph。
- `paths_pt.jsonl` 有 32,572,697 条 path 记录。
- 构造时只保留 `complete_npz=true` 的 path，并设置 `max_paths_per_graph=1`。
- 缺失完整 8 视角渲染的 graph 会被过滤，最终得到 135,051 条训练记录。

因此，现有 BrickNet-MM PT 的准确粒度是：

> **graph 级均衡采样，path 级语言建模监督。**

每个 graph 最多选择一条完整 path，graph 的渲染和 inventory 是条件输入；模型并不直接预测 graph
节点或边，而是在 assistant response 上自回归预测序列化后的 path token。

## 与 BrickNet 论文 PT/SFT 的区别

BrickNet 论文将 PT 和 SFT 分工如下：

| 阶段 | 数据与输入 | 目标 |
| --- | --- | --- |
| BrickNet PT | `paths_pt` 与 `paths_sft` path pool 合并、按 path text 去重；无 caption | 无条件生成 BrickNet path，学习 part、connector、path grammar 和局部几何 |
| BrickNet SFT | caption 与 path 配对 | caption-conditioned path generation |
| 当前 BrickNet-MM PT | PT split 的 image、空 caption、inventory | image/inventory-conditioned path generation |
| 当前 BrickNet-MM SFT | SFT split 的 image、caption、inventory |完整条件下的 path generation |

论文 PT 使用约 8,092,423 条去重 path，覆盖的 path 远多于 graph 数；当前 BrickNet-MM PT 为每个可用 graph
最多保留一条完整 path。前者强调 path 多样性和连接语法，后者强调最终图像、parts inventory 与 path 的对应关系。
论文的 representation、dataset 和 training recipe 见
[BrickNet: Graph-Backed Generative Brick Assembly](https://arxiv.org/html/2604.22984)。

## 混合数据设计

推荐的第一组实验采用固定训练预算：

| 数据分支 | 样本数 | 比例 | 作用 |
| --- | ---: | ---: | --- |
| `BrickNet-PT-Text-270k` | 270,102 | 66.7% | 原始 BrickNet PT+SFT path pool 的去重随机子集，学习 path grammar 和 connector 组合 |
| `BrickNet-MM-PT` | 135,051 | 33.3% | PT graph 的图像和 inventory 条件，保持多模态 grounding |
| 合计 | 405,153 | 100% | 训练 1 epoch |

旧实验的样本曝光量为 `135,051 × 3 = 405,153`。新方案训练 1 epoch 后具有相同样本曝光量；在 batch size 2、
GA 8 和单卡训练下仍约为 25,323 optimizer steps。这样可以在基本固定计算预算的前提下判断 path diversity
是否改善后续 SFT。

原始文本 path 从 BrickNet 论文使用的 `paths_pt.jsonl` 和 `paths_sft.jsonl` 合并池中抽取，不使用
`paths_val.jsonl`。抽样脚本使用带 seed 的 BLAKE2b bottom-k：

- 对规范化后的 path text 做确定性 hash 抽样，不依赖源文件顺序。
- 跨 PT/SFT pool 按 path text 去重。
- 不要求 `complete_npz=true`，保留论文 PT 中最长 100 parts 的有效 collision-free build prefix。
- 输出无图像的 ShareGPT 样本：`unconditional instruction -> path`。

不推荐直接把完整的 8,092,423 条 BrickNet path 与 135,051 条 BrickNet-MM 样本 concat，因为多模态样本只占约
1.6%，会使图像条件在训练中被严重稀释。

## 为什么使用 `stage=sft`

LlamaFactory 的 `stage=pt` 使用纯文本 pretrain processor，对整段文本计算 loss，不能与带 `images` 的
ShareGPT 记录共用同一数据协议。混合实验统一使用 `stage=sft`：

- 文本分支：固定无条件指令作为 user prompt，BrickNet path 作为 assistant response。
- 多模态分支：image、空 caption、inventory 作为 user prompt，完整 path 作为 assistant response。
- `train_on_prompt=false`，两类数据都只在 path response token 上计算 next-token CE。

这里的 `PT` 表示任务阶段和数据用途，并不要求 LlamaFactory 参数必须写成 `stage=pt`。

## 数据准备

在 LlamaFactory 根目录执行：

```bash
conda run -n llamafactory --no-capture-output \
  python scripts/prepare_bricknet_text_pt.py
```

默认输入：

```text
../BrickNet/data/bricknet_datasets/paths_pt.jsonl
../BrickNet/data/bricknet_datasets/paths_sft.jsonl
```

默认输出：

```text
data/BrickNet-PT_text_270102_seed42.jsonl
data/BrickNet-PT_text_270102_seed42.jsonl.report.json
```

两个源文件约 175 GB，脚本需要顺序扫描一次。可先执行格式 smoke test：

```bash
conda run -n llamafactory --no-capture-output \
  python scripts/prepare_bricknet_text_pt.py \
  --num-samples 32 \
  --max-input-rows 1000 \
  --output /tmp/BrickNet-PT_text_smoke.jsonl
```

当前已经生成完整数据：

| 项目 | 值 |
| --- | ---: |
| 输出大小 | 1.4 GB |
| 样本数 | 270,102 |
| `paths_pt` | 228,587（84.63%） |
| `paths_sft` | 41,515（15.37%） |
| Parts | min 1 / mean 72.7455 / max 100 |
| `complete_npz=true` | 69,969（25.90%） |
| 重复 ID | 0 |
| SHA-256 | `f9b5e410abe1ec4e1aae436d51de12538433b1acb2a891a67eb97c6bd577dd48` |

## 预训练命令

数据准备完成后执行：

```bash
conda run -n llamafactory --no-capture-output \
  llamafactory-cli train examples/train_lora/qwen35_08b_bricknet_mixed_pt.yaml
```

该配置从原始 `Qwen/Qwen3.5-0.8B` 开始创建新的混合 PT LoRA，不加载旧的 BrickNet-MM PT adapter。后续
BrickNet-MM SFT 应将 `adapter_name_or_path` 指向新的混合 PT 输出，并启用 `create_new_adapter`，以保持
PT 与 SFT adapter 分离。
