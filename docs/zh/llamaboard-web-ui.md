# LlamaBoard Web UI 后训练使用指南

本文基于当前项目中的 WebUI 实现整理，主要参考：

- `src/llamafactory/webui/components/top.py`
- `src/llamafactory/webui/components/train.py`
- `src/llamafactory/webui/components/eval.py`
- `src/llamafactory/webui/components/infer.py`
- `src/llamafactory/webui/components/export.py`
- `src/llamafactory/webui/runner.py`
- `src/llamafactory/webui/locales.py`

WebUI 的核心作用是把页面控件转换成 `llamafactory-cli train`、评估、推理和导出的参数。训练开始后，WebUI 会在输出目录下保存：

- `training_args.yaml`：真正传给训练进程的参数。
- `llamaboard_config.yaml`：WebUI 页面配置，用于恢复界面状态。
- `webui_subprocess.log`：训练子进程 stdout/stderr 日志。

## 启动 WebUI

在项目根目录启动：

```bash
llamafactory-cli webui
```

默认监听 `0.0.0.0:7860`。如果在服务器上运行，可通过浏览器访问对应机器的 IP 和端口。

建议使用项目支持的 Python 版本。当前仓库的 `pyproject.toml` 标注支持 Python 3.11、3.12、3.13；过新的 Python 版本可能导致 `datasets`、`dill`、`deepspeed` 等依赖在加载数据或初始化分布式训练时失败。

## 后训练基本流程

1. 准备模型

   在顶部区域选择 `Model name`，WebUI 会自动填充 `Model path` 和 `Template`。如果模型不在内置列表，选择 `Custom`，手动填写本地路径或 Hugging Face / ModelScope 模型 ID。

2. 准备数据

   数据默认放在 `data/` 下，并在 `data/dataset_info.json` 中注册。WebUI 的 `Dataset` 下拉框来自该文件。不同训练阶段会筛选不同类型的数据集。

3. 选择训练阶段

   常用后训练阶段包括：

   | WebUI 名称 | 实际参数 `stage` | 用途 | 数据要求 |
   | --- | --- | --- | --- |
   | `Supervised Fine-Tuning` | `sft` | 监督微调，学习指令到回答 | 普通指令数据或 ShareGPT 对话数据 |
   | `Reward Modeling` | `rm` | 训练奖励模型 | 偏好数据，包含 `chosen` / `rejected` |
   | `PPO` | `ppo` | 基于奖励模型做强化学习 | prompt 数据，并且必须选择奖励模型 |
   | `DPO` | `dpo` | 直接偏好优化 | 偏好数据，包含 `chosen` / `rejected` |
   | `KTO` | `kto` | Kahneman-Tversky Optimization | 带好坏标签的数据，例如 `kto_tag` |
   | `Pre-Training` | `pt` | 继续预训练 | 纯文本或预训练格式数据，WebUI 会默认启用 packing |

4. 选择微调方式

   顶部 `Finetuning method` 决定训练哪些参数：

   | 方法 | 实际参数 `finetuning_type` | 作用 | 典型场景 |
   | --- | --- | --- | --- |
   | `full` | `full` | 全参数微调 | 显存充足、需要最大可塑性 |
   | `freeze` | `freeze` | 只训练部分层或模块 | 低成本微调、保留大部分原模型能力 |
   | `lora` | `lora` | 训练 LoRA adapter | 最常用，显存占用低，便于合并和复用 |
   | `oft` | `oft` | 训练 OFT adapter | PEFT 方法之一，WebUI 顶部可选，但训练页主要暴露 LoRA 参数 |

5. 设置训练参数

   填写数据、batch、学习率、精度、输出目录等。点击 `Preview command` 可查看 WebUI 生成的 CLI 命令。

6. 开始训练

   点击 `Start` 后，WebUI 会调用：

   ```bash
   llamafactory-cli train <output_dir>/training_args.yaml
   ```

   如果启用 DeepSpeed，WebUI 会设置 `FORCE_TORCHRUN=1`，通过分布式启动。

7. 评估、聊天测试、导出

   训练完成后，在顶部 `Checkpoint path` 选择 adapter 或 checkpoint，然后使用 `Chat` 测试效果，或使用 `Export` 合并/导出模型。

## 顶部公共参数

顶部参数会被 Train、Evaluate & Predict、Chat、Export 共用。

| 参数 | 对应字段 | 含义和作用 |
| --- | --- | --- |
| `Language` | UI 状态 | 界面语言，不影响训练。 |
| `Model name` | UI 模型别名 | 从内置模型列表选择模型。它是 WebUI 显示名，不一定等于 Hugging Face repo 名。例如 `Qwen3.5-0.8B-Thinking` 映射到 `Qwen/Qwen3.5-0.8B`。 |
| `Model path` | `model_name_or_path` | 真实模型路径，可以是本地目录，也可以是 Hugging Face / ModelScope 模型 ID。训练、推理、导出都以它为准。 |
| `Hub name` | 环境选择 | 选择模型下载源：`huggingface`、`modelscope`、`openmind`。切换后会影响内置模型路径的解析。 |
| `Finetuning method` | `finetuning_type` | 微调方法：`full`、`freeze`、`lora`、`oft`。会影响 checkpoint 加载、量化可用性和训练参数。 |
| `Checkpoint path` | `adapter_name_or_path` 或 `model_name_or_path` | 选择已有训练结果。LoRA/OFT 会作为 adapter 加载；full/freeze 通常作为模型 checkpoint 加载。训练时可用于继续基于已有 adapter 训练，推理/导出时用于加载训练结果。 |
| `Quantization bit` | `quantization_bit` | 在线量化加载模型，用于 QLoRA。WebUI 只允许 PEFT 方法使用量化；`full` / `freeze` 下会被禁用。 |
| `Quantization method` | `quantization_method` | 量化算法。`bnb` 支持 4/8 bit，`hqq` 支持更多 bit，`eetq` 支持 8 bit。 |
| `Template` | `template` | 对话模板，决定数据如何拼接成模型输入，包括 special tokens、system/user/assistant 格式、多模态 token、tool 格式、thinking 处理等。必须和模型类型匹配。 |
| `RoPE scaling` | `rope_scaling` | RoPE 插值方式，用于扩展上下文长度。可选 `none`、`linear`、`dynamic`、`yarn`、`llama3`。模型不支持时不要乱开。 |
| `Booster` | `flash_attn` / `use_unsloth` / `enable_liger_kernel` | 加速方式。`flashattn2` 设置 `flash_attn=fa2`；`unsloth` 设置 `use_unsloth=True`；`liger_kernel` 设置 `enable_liger_kernel=True`；`auto` 使用默认策略。 |

### Template 与 enable_thinking

`template` 是数据格式参数，决定样本如何被编码。`enable_thinking` 是 reasoning 模板的开关，只在 `template` 对应 `ReasoningTemplate` 时生效。

以 Qwen3.5 为例：

| Template | 类型 | 行为 |
| --- | --- | --- |
| `qwen3_5` | reasoning template | 会处理 `<think>...</think>`，并根据 `enable_thinking` 决定是否训练思考内容。 |
| `qwen3_5_nothink` | 普通 template | 不做 reasoning 特殊处理，`enable_thinking` 基本不起作用。 |

常见选择：

- 想训练模型输出思考内容：`template=qwen3_5`，`enable_thinking=true`。
- 想训练模型直接回答：优先使用 `template=qwen3_5_nothink`，并设置 `enable_thinking=false`。
- 使用 `qwen3_5` 且 `enable_thinking=false`：仍使用 reasoning 模板，但会移除样本中的 thinking 内容，并抑制模型学习思考过程。

## Train 页参数

### 数据与阶段

| 参数 | 对应字段 | 含义和作用 |
| --- | --- | --- |
| `Training stage` | `stage` | 训练阶段。WebUI 显示英文名称，内部转换为 `sft`、`rm`、`ppo`、`dpo`、`kto`、`pt`。 |
| `Dataset dir` | `dataset_dir` | 数据目录，默认 `data`。该目录下需要有 `dataset_info.json`。 |
| `Dataset` | `dataset` | 训练数据集名称，可多选。名称来自 `dataset_info.json`。多选后会用英文逗号拼接传入训练参数。 |
| `Preview dataset` | UI 功能 | 预览数据样本，不写入训练参数。 |
| `Preview count` / `Page index` | UI 功能 | 控制数据预览数量和页码，不写入训练参数。 |

数据集筛选规则：

- `rm`、`dpo` 会显示 `dataset_info.json` 中 `ranking: true` 的数据集。
- `sft`、`pt`、`ppo`、`kto` 会显示非 `ranking` 数据集。
- `pt` 阶段切换时，WebUI 会自动打开 `packing`。

### 基础训练超参数

| 参数 | 对应字段 | 含义和作用 |
| --- | --- | --- |
| `Learning rate` | `learning_rate` | AdamW 初始学习率。LoRA 常见范围 `1e-4` 到 `5e-5`；全参训练通常更小。 |
| `Epochs` | `num_train_epochs` | 训练轮数。每轮遍历一次训练集。 |
| `Max grad norm` | `max_grad_norm` | 梯度裁剪阈值，防止梯度爆炸。`1.0` 是常用默认值。 |
| `Max samples` | `max_samples` | 每个数据集最多使用多少样本。调试时可设小，正式训练设为大于数据集大小。 |
| `Compute type` | `fp16` / `bf16` / `pure_bf16` | 训练精度。`bf16` 适合 Ampere/Hopper 等支持 bf16 的 GPU；`fp16` 兼容性更广；`fp32` 最稳但显存和速度开销大；`pure_bf16` 使用纯 bf16。 |
| `Cutoff length` | `cutoff_len` | tokenized 后最大序列长度。超过会截断。多模态样本中图像 token 也会占上下文预算。 |
| `Batch size` | `per_device_train_batch_size` | 每张 GPU 上的 micro batch size。显存不够时优先降低它。 |
| `Gradient accumulation` | `gradient_accumulation_steps` | 梯度累积步数。有效全局 batch size 约等于 `batch_size * gradient_accumulation_steps * GPU 数量`。 |
| `Val size` | `val_size` | 从训练集切分验证集的比例或数量。大于 0 且非 PPO 时，WebUI 会设置 `eval_strategy=steps`、`eval_steps=save_steps`。 |
| `LR scheduler` | `lr_scheduler_type` | 学习率调度器，例如 `cosine`、`linear`、`constant` 等，来自 Transformers 的 `SchedulerType`。 |

### 其它参数设置

| 参数 | 对应字段 | 含义和作用 |
| --- | --- | --- |
| `Logging steps` | `logging_steps` | 每隔多少 update step 输出一次日志。 |
| `Save steps` | `save_steps` | 每隔多少 update step 保存一次 checkpoint。验证集开启时也作为 `eval_steps`。 |
| `Warmup steps` | `warmup_steps` | 学习率预热步数。 |
| `NEFTune alpha` | `neftune_noise_alpha` | 在 embedding 上加入噪声的强度，用于 NEFTune。`0` 表示关闭。 |
| `Extra args` | 直接合并到训练参数 | JSON 格式的额外参数。例如 `{"optim": "adamw_torch"}`。这里可以传任何 CLI 支持但 WebUI 没暴露的参数。 |
| `Packing` | `packing` | 把多个短样本打包到一个长序列，提高训练效率。适合预训练或大量短样本；对需要严格轮次边界的任务要谨慎。 |
| `Neat packing` | `neat_packing` | 无污染 packing，避免 packed 样本之间产生交叉注意力。开启后会自动启用 `packing`。 |
| `Train on prompt` | `train_on_prompt` | SFT 中是否对 prompt 部分也计算 loss。默认只训练 assistant response。 |
| `Mask history` | `mask_history` | 多轮 SFT 中只训练最后一轮 assistant，历史轮次不计算 loss。不能和 `train_on_prompt` 同时开启。 |
| `Resize vocab` | `resize_vocab` | 调整 tokenizer 词表和 embedding 大小。添加新 token 或 special token 时需要。 |
| `Use LLaMA Pro` | `use_llama_pro` | 只训练块扩展后的参数。属于特殊训练方法，普通 LoRA/SFT 不需要。 |
| `Enable thinking` | `enable_thinking` | 是否启用 reasoning 模型的思考模式。只在 reasoning template 下生效。 |
| `Report to` | `report_to` | 外部实验记录后端，如 `none`、`wandb`、`mlflow`、`tensorboard`、`trackio`、`all`。 |

Trackio Settings：

| 参数 | 对应字段 | 含义和注意事项 |
| --- | --- | --- |
| `Project Name` | `project` | Trackio/W&B 等实验项目名。当前 WebUI 控件存在，但 `runner.py` 没有把它写入训练参数；需要在 `Extra args` 中补充。 |
| `Trackio Space ID` | `trackio_space_id` | Trackio 的 Hugging Face Space ID。当前需要通过 `Extra args` 传入。 |
| `Private Repository` | `hub_private_repo` | 是否创建私有 HF 仓库。当前需要通过 `Extra args` 传入。 |

如果使用 Trackio，建议：

```json
{
  "project": "my-project",
  "trackio_space_id": "org/space",
  "hub_private_repo": true
}
```

### Freeze 参数

仅当顶部 `Finetuning method=freeze` 时写入训练参数。

| 参数 | 对应字段 | 含义和作用 |
| --- | --- | --- |
| `Trainable layers` | `freeze_trainable_layers` | 可训练层数。正数表示最后 N 层可训练，负数表示最前 N 层可训练。 |
| `Trainable modules` | `freeze_trainable_modules` | 隐藏层中可训练模块名，英文逗号分隔。`all` 表示所有可用模块。 |
| `Extra modules` | `freeze_extra_modules` | 隐藏层以外额外可训练模块名，例如 embedding、lm_head 等，英文逗号分隔。 |

### LoRA 参数

仅当顶部 `Finetuning method=lora` 时写入训练参数。

| 参数 | 对应字段 | 含义和作用 |
| --- | --- | --- |
| `LoRA rank` | `lora_rank` | LoRA 低秩矩阵的秩。越大容量越强，显存和参数量越高。 |
| `LoRA alpha` | `lora_alpha` | LoRA 缩放系数。常见设置为 `2 * rank`。 |
| `LoRA dropout` | `lora_dropout` | LoRA 分支 dropout，防止过拟合。 |
| `LoRA+ LR ratio` | `loraplus_lr_ratio` | LoRA+ 中 B 矩阵相对 A 矩阵的学习率倍数。`0` 表示不启用。 |
| `Create new adapter` | `create_new_adapter` | 在已有 adapter 基础上创建一个随机初始化的新 adapter。用于继续训练但不直接覆盖原 adapter。 |
| `Use rslora` | `use_rslora` | 使用 rank-stabilized LoRA 缩放。 |
| `Use DoRA` | `use_dora` | 使用 DoRA，即权重分解 LoRA。 |
| `Use PiSSA` | `pissa_init` / `pissa_convert` | 使用 PiSSA 初始化并在保存时转换为普通 LoRA。 |
| `LoRA target` | `lora_target` | 应用 LoRA 的模块名，英文逗号分隔。留空时 WebUI 传 `all`，表示尽量作用到线性层。 |
| `Additional target` | `additional_target` | 除 LoRA 层外额外训练并保存的模块，例如 `embed_tokens,lm_head`。 |

`Create new adapter` 的具体行为取决于顶部是否选择了已有 `Checkpoint path`：

| 场景 | 未勾选 `Create new adapter` | 勾选 `Create new adapter` |
| --- | --- | --- |
| 未选择已有 adapter | 创建一个新的 LoRA adapter。 | 创建一个新的 LoRA adapter，效果基本相同。 |
| 选择 1 个已有 adapter | 加载该 adapter 并继续训练，会修改/续训它。 | 先将该 adapter 合并进基础模型，再创建一个随机初始化的新 adapter 进行训练。 |
| 选择多个已有 adapter | 前面的 adapter 会先合并进模型，最后一个 adapter 会被加载并继续训练。 | 所有已选 adapter 都会先合并进模型，然后创建一个随机初始化的新 adapter 进行训练。 |

推荐用法：

- 想继续训练已有 LoRA：不要勾选 `Create new adapter`。
- 想保留已有 LoRA 不动，并在它的效果基础上再训练一个新 LoRA：勾选 `Create new adapter`。

注意事项：

- 勾选后训练得到的新 adapter 是在“已有 adapter 已合并进模型”的状态上训练出来的。后续推理或导出时，通常需要同时考虑旧 adapter 和新 adapter 的组合关系。
- 如果启用了量化并且选择了已有 adapter，不能再勾选 `Create new adapter`，否则参数校验会报错：`Cannot create new adapter upon a quantized model.`。

### RLHF 参数

根据训练阶段生效。

| 参数 | 生效阶段 | 对应字段 | 含义和作用 |
| --- | --- | --- | --- |
| `Beta` | `dpo`、`kto` | `pref_beta` | 偏好损失中的 beta 超参数，控制偏好约束强度。 |
| `Ftx gamma` | `dpo`、`kto` | `pref_ftx` | 在偏好优化中混入 SFT loss 的权重。 |
| `Loss type` | `dpo`、`kto` | `pref_loss` | 偏好损失类型：`sigmoid`、`hinge`、`ipo`、`kto_pair`、`orpo`、`simpo`。 |
| `Reward model` | `ppo` | `reward_model` | PPO 使用的奖励模型 checkpoint。PPO 阶段必填。 |
| `Normalize score` | `ppo` | `ppo_score_norm` | 对奖励分数做归一化。 |
| `Whiten rewards` | `ppo` | `ppo_whiten_rewards` | 对奖励做白化，用于优势估计。 |

PPO 训练时，WebUI 还会自动设置：

```yaml
reward_model_type: lora  # finetuning_type=lora 时
top_k: 0
top_p: 0.9
```

### 多模态参数

仅当 `Model name` 属于项目内置的多模态模型列表时写入训练参数。手动填写 `Custom` 模型时，即使模型本身支持视觉，也不会自动触发这些 WebUI 条件逻辑。

| 参数 | 对应字段 | 含义和作用 |
| --- | --- | --- |
| `Freeze vision tower` | `freeze_vision_tower` | 冻结视觉编码器。多模态 LoRA 中通常保持开启。 |
| `Freeze multi-modal projector` | `freeze_multi_modal_projector` | 冻结视觉/音频到语言模型的投影层。想适配新视觉任务时可关闭。 |
| `Freeze language model` | `freeze_language_model` | 冻结语言模型主体，只训练多模态相关模块。 |
| `Image max pixels` | `image_max_pixels` | 输入图像最大像素数。可写 `768*768`，WebUI 会转换为整数。 |
| `Image min pixels` | `image_min_pixels` | 输入图像最小像素数。 |
| `Video max pixels` | `video_max_pixels` | 输入视频帧最大像素数。 |
| `Video min pixels` | `video_min_pixels` | 输入视频帧最小像素数。 |

多模态数据需要在 `dataset_info.json` 中配置 `images`、`videos` 或 `audios` 列，并在样本文本中放入对应占位符，如 `<image>`。

### GaLore 参数

开启 `Use GaLore` 后生效。

| 参数 | 对应字段 | 含义和作用 |
| --- | --- | --- |
| `Use GaLore` | `use_galore` | 使用 GaLore 低秩梯度投影优化器。 |
| `GaLore rank` | `galore_rank` | 梯度投影秩。 |
| `Update interval` | `galore_update_interval` | 更新投影矩阵的步数间隔。 |
| `GaLore scale` | `galore_scale` | GaLore 缩放系数。 |
| `GaLore target` | `galore_target` | 应用 GaLore 的模块名，英文逗号分隔，`all` 表示所有线性层。 |

### APOLLO 参数

开启 `Use APOLLO` 后生效。

| 参数 | 对应字段 | 含义和作用 |
| --- | --- | --- |
| `Use APOLLO` | `use_apollo` | 使用 APOLLO 优化器。 |
| `APOLLO rank` | `apollo_rank` | APOLLO 低秩投影秩。 |
| `Update interval` | `apollo_update_interval` | 更新投影的步数间隔。 |
| `APOLLO scale` | `apollo_scale` | APOLLO 缩放系数。 |
| `APOLLO target` | `apollo_target` | 应用 APOLLO 的模块名，英文逗号分隔，`all` 表示所有线性层。 |

### BAdam 参数

开启 `Use BAdam` 后生效。

| 参数 | 对应字段 | 含义和作用 |
| --- | --- | --- |
| `Use BAdam` | `use_badam` | 使用 BAdam 优化器。 |
| `BAdam mode` | `badam_mode` | `layer` 表示按层切换训练块；`ratio` 表示按比例更新参数。 |
| `Switch strategy` | `badam_switch_mode` | Layer-wise BAdam 的块切换策略：`ascending`、`descending`、`random`、`fixed`。 |
| `Switch interval` | `badam_switch_interval` | 每隔多少步切换训练块。 |
| `Block update ratio` | `badam_update_ratio` | Ratio-wise BAdam 每次更新的参数比例。 |

### SwanLab 参数

开启 `Use SwanLab` 后生效，用于实验跟踪和可视化。

| 参数 | 对应字段 | 含义和作用 |
| --- | --- | --- |
| `Use SwanLab` | `use_swanlab` | 启用 SwanLab。 |
| `SwanLab project` | `swanlab_project` | 项目名，默认 `llamafactory`。 |
| `SwanLab run name` | `swanlab_run_name` | 实验名，可留空。 |
| `SwanLab workspace` | `swanlab_workspace` | 工作区名，可留空。 |
| `SwanLab API key` | `swanlab_api_key` | API 密钥。环境已登录时可留空。 |
| `SwanLab mode` | `swanlab_mode` | `cloud` 云端模式或 `local` 本地模式。 |

### 训练控制和输出

| 参数 | 对应字段 | 含义和作用 |
| --- | --- | --- |
| `Preview command` | UI 功能 | 生成并展示等价 CLI 命令，不启动训练。 |
| `Save training args` | UI 功能 | 将当前 WebUI 配置保存到 `llamaboard_config/*.yaml`。 |
| `Load training args` | UI 功能 | 从 `llamaboard_config/*.yaml` 载入 WebUI 配置。 |
| `Start` | UI 功能 | 启动训练子进程。 |
| `Stop` | UI 功能 | 递归中断训练子进程。 |
| `Output dir` | `output_dir` | 输出目录。若只填目录名，实际路径为 `saves/<Model name>/<Finetuning method>/<Output dir>`。 |
| `Config path` | UI 功能 | 保存/载入 WebUI 配置文件的路径，位于 `llamaboard_config/`。 |
| `Device count` | UI 显示 | 当前可用训练设备数量，不写入训练参数。 |
| `DeepSpeed stage` | `deepspeed` | 选择 `2` 或 `3` 时，WebUI 会生成并使用 `llamaboard_cache/ds_z*_config.json`。 |
| `Use offload` | `deepspeed` | 搭配 DeepSpeed 使用 CPU offload，降低显存但通常变慢。 |
| `Progress bar` | UI 显示 | 从训练日志中读取进度。 |
| `Output box` | UI 显示 | 展示训练日志或报错。 |
| `Loss viewer` | UI 显示 | 根据 `trainer_log.jsonl` 绘制 loss 曲线。 |

## Evaluate & Predict 页参数

该页用于评估或生成预测结果。它会构造 `do_eval` 或 `do_predict` 参数。

| 参数 | 对应字段 | 含义和作用 |
| --- | --- | --- |
| `Dataset dir` | `dataset_dir` | 评估数据目录。 |
| `Dataset` | `eval_dataset` | 评估数据集名称，可多选。 |
| `Cutoff length` | `cutoff_len` | 评估输入最大 token 长度。 |
| `Max samples` | `max_samples` | 评估最多使用多少样本。 |
| `Batch size` | `per_device_eval_batch_size` | 每张 GPU 的评估 batch size。 |
| `Save predictions` | `do_predict` / `do_eval` | 开启时保存生成预测，使用 `do_predict=True`；关闭时只评估，使用 `do_eval=True`。 |
| `Max new tokens` | `max_new_tokens` | 评估生成时最多生成的新 token 数。 |
| `Top-p` | `top_p` | nucleus sampling 参数。 |
| `Temperature` | `temperature` | 采样温度。越高越随机，越低越保守。 |
| `Output dir` | `output_dir` | 评估结果保存目录。 |
| `Preview command` | UI 功能 | 预览评估命令。 |
| `Start` | UI 功能 | 启动评估或预测。 |
| `Stop` | UI 功能 | 中断评估子进程。 |

评估页也会使用顶部的模型、checkpoint、量化、template、RoPE、booster 等公共参数。

## Chat 页参数

该页用于加载模型并交互测试。它不会启动训练。

### 模型加载参数

| 参数 | 对应字段 | 含义和作用 |
| --- | --- | --- |
| `Infer backend` | `infer_backend` | 推理后端：`huggingface`、`vllm`、`sglang`。 |
| `Infer dtype` | `infer_dtype` | 推理精度：`auto`、`float16`、`bfloat16`、`float32`。 |
| `Extra args` | 合并到推理参数 | JSON 格式的额外推理参数。默认示例 `{"vllm_enforce_eager": true}`。 |
| `Load model` | UI 功能 | 按顶部模型和 checkpoint 加载模型。 |
| `Unload model` | UI 功能 | 卸载模型并清理显存。 |
| `Info box` | UI 显示 | 显示模型加载状态或错误。 |

### 对话参数

| 参数 | 对应字段 | 含义和作用 |
| --- | --- | --- |
| `Role` | message role | 新输入消息的角色，通常为 `user`；`observation` 用于工具调用结果。 |
| `System` | system prompt | 系统提示词。 |
| `Tools` | tools schema | 工具定义，JSON list 格式，每个 tool 需要有 `name`。 |
| `Image` | images | 多模态输入图像，仅多模态模型区域显示。 |
| `Video` | videos | 多模态输入视频。 |
| `Audio` | audios | 多模态输入音频。 |
| `Query` | user content | 用户输入文本。 |
| `Submit` | UI 功能 | 发送消息并流式生成回答。 |
| `Max new tokens` | `max_new_tokens` | 单次最多生成的新 token 数。 |
| `Top-p` | `top_p` | nucleus sampling 参数。 |
| `Temperature` | `temperature` | 采样温度。 |
| `Skip special tokens` | `skip_special_tokens` | 解码时是否移除特殊 token。 |
| `Escape HTML` | UI 展示 | 是否转义 HTML，防止回答被浏览器当作 HTML 渲染。 |
| `Enable thinking` | template 临时属性 | 临时设置当前 template 的 `enable_thinking`，仅对 reasoning template 有意义。 |
| `Clear history` | UI 功能 | 清空当前对话历史。 |

如果模型输出包含 `<think>...</think>`，WebUI 会把思考内容折叠显示。

## Export 页参数

该页用于导出训练后的模型或 adapter。对于 LoRA/OFT，通常先在顶部选择 checkpoint，再导出合并后的模型。

| 参数 | 对应字段 | 含义和作用 |
| --- | --- | --- |
| `Max shard size (GB)` | `export_size` | 导出模型单个分片最大大小，单位 GB。 |
| `Export quantization bit` | `export_quantization_bit` | 导出时做 GPTQ 量化，可选 `8`、`4`、`3`、`2`。`none` 表示不量化。 |
| `Export quantization dataset` | `export_quantization_dataset` | GPTQ 量化校准数据集路径或名称。量化导出时必填。 |
| `Export device` | `export_device` | 导出设备：`cpu` 或 `auto`。`auto` 可使用可用加速设备。 |
| `Export legacy format` | `export_legacy_format` | 使用旧 `.bin` 格式；默认使用 `.safetensors`。 |
| `Export dir` | `export_dir` | 导出模型保存目录。 |
| `HF Hub ID` | `export_hub_model_id` | 如果要上传 Hugging Face Hub，填写目标 repo ID。 |
| `Extra args` | 合并到导出参数 | JSON 格式额外导出参数。 |
| `Start export` | UI 功能 | 开始导出。 |
| `Info box` | UI 显示 | 导出状态或错误。 |

导出限制：

- 如果选择 GPTQ 量化导出，必须填写量化校准数据集。
- 如果不是量化导出，通常需要在顶部选择一个 checkpoint。
- 当前实现不支持同时对多个 LoRA adapter 做 GPTQ 量化导出。

## 常用配置示例

### 单卡 LoRA SFT

适合大多数指令微调。

```yaml
stage: sft
finetuning_type: lora
learning_rate: 5e-5
num_train_epochs: 3
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
lora_rank: 8
lora_alpha: 16
lora_target: all
quantization_bit: none
ds_stage: none
```

显存不足时优先调整：

1. 降低 `Batch size`。
2. 开启 `Quantization bit=4` 做 QLoRA。
3. 降低 `Cutoff length`。
4. 再考虑 DeepSpeed 或 offload。

### QLoRA

```yaml
finetuning_type: lora
quantization_bit: 4
quantization_method: bnb
lora_target: all
```

QLoRA 只适用于 PEFT 方法。WebUI 会在 NPU 以外默认设置 `double_quantization=True`。

### 多模态 LoRA SFT

```yaml
stage: sft
finetuning_type: lora
template: qwen3_vl 或模型对应 template
freeze_vision_tower: true
freeze_multi_modal_projector: true
freeze_language_model: false
image_max_pixels: 768*768
image_min_pixels: 32*32
```

注意：

- 文本模型不能训练带 `images` 的多模态数据。
- 多模态数据集需要在 `dataset_info.json` 中声明 `images` / `videos` / `audios` 列。
- 样本文本中需要有 `<image>`、`<video>`、`<audio>` 等占位符。

### DPO

```yaml
stage: dpo
dataset: <ranking_dataset>
pref_beta: 0.1
pref_loss: sigmoid
pref_ftx: 0
```

DPO 数据通常需要 `chosen` 和 `rejected`。WebUI 会在 DPO 阶段只列出 `ranking: true` 的数据集。

### PPO

```yaml
stage: ppo
reward_model: <reward_model_checkpoint>
ppo_score_norm: false
ppo_whiten_rewards: false
```

PPO 必须选择奖励模型。奖励模型通常来自 `Reward Modeling` 阶段训练结果。

## 排错建议

1. 先看 `webui_subprocess.log`

   训练失败时，优先查看输出目录下的 `webui_subprocess.log`。WebUI 最终显示的错误可能被截断。

2. 用 `Preview command` 复现

   点击 `Preview command` 后复制命令，在终端运行，可以更直接地定位环境变量、路径、权限和依赖问题。

3. 确认真实训练参数

   WebUI 控件不等于最终参数。以输出目录下的 `training_args.yaml` 为准。

4. 检查数据集注册

   如果下拉框看不到数据集，检查：

   - `dataset_dir` 是否正确。
   - `dataset_info.json` 是否是合法 JSON。
   - 当前训练阶段是否会筛选掉该数据集。
   - `ranking` 字段是否符合阶段要求。

5. 检查模型和 template

   Chat/Instruct/Thinking/VL 模型需要对应 template。`default` 只适合基座模型或简单文本续写场景。template 不匹配会导致训练格式错误，表现为模型不收敛、输出异常或 special token 混乱。

6. 检查多模态模型和数据是否匹配

   含 `images`、`videos`、`audios` 的数据必须使用支持对应模态的模型。文本模型不能直接训练图像数据。

7. 谨慎启用 DeepSpeed

   DeepSpeed 依赖和 Python/PyTorch/CUDA 版本强相关。小模型 LoRA 通常不需要 DeepSpeed；如果导入 DeepSpeed 就失败，先关闭 `DeepSpeed stage` 验证普通训练链路。

8. 区分 WebUI 显示名和真实模型路径

   `Model name` 是 LlamaFactory 的别名，`Model path` 才是真正加载的模型。例如某些 `-Thinking` 名称只是 WebUI 分类，实际路径可能是不带该后缀的 Hugging Face repo。
