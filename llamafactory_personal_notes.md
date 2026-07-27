# LlamaFactory 个人使用心得


## 预处理 tokenized dataset

### tokenized_path 的行为

`--tokenized_path` 指向的是预处理后、已经 tokenize 的 Hugging Face dataset 保存目录。

- 如果该目录不存在，或不是有效的 tokenized dataset，LlamaFactory 会读取原始 dataset，按当前参数进行预处理和 tokenize，然后执行 `save_to_disk(tokenized_path)`。
- 如果该目录已经存在且有效，下次训练会直接 `load_from_disk(tokenized_path)`，不会重新处理原始 dataset。
- 直接读取 `tokenized_path` 时，会忽略其它数据参数。因此如果修改了 `template`、`cutoff_len`、`max_samples`、`packing`、`enable_thinking`、`image_max_pixels` 等会影响预处理结果的参数，应该换一个新的 `tokenized_path`，或删除旧目录后重新生成。

相关代码：

- `src/llamafactory/data/loader.py::get_dataset`
- `has_tokenized_data(...)`
- `load_from_disk(...)`
- `dataset_dict.save_to_disk(...)`

### 不设置 tokenized_path 时

如果不设置 `--tokenized_path`，两次完全相同的训练不会走显式的 `load_from_disk(tokenized_path)` 快速路径。

不过 `dataset.map(...)` 内部仍可能使用 Hugging Face Datasets 自己的 map cache：

```python
load_from_cache_file=(not data_args.overwrite_cache)
```

也就是说，不设置 `tokenized_path` 时可能复用部分缓存，但可控性和确定性不如显式设置 `tokenized_path`。

### 只预处理 dataset，不进行 SFT 训练

可以继续使用 `llamafactory-cli train` 入口，但关闭训练、评估和预测：

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
  --stage sft \
  --do_train False \
  --do_eval False \
  --do_predict False \
  --model_name_or_path Qwen/Qwen3.5-0.8B \
  --preprocessing_num_workers 16 \
  --template qwen3_5_nothink \
  --dataset_dir data \
  --dataset BrickNet-MM-SFT \
  --cutoff_len 4096 \
  --max_samples 10000 \
  --packing False \
  --enable_thinking False \
  --report_to none \
  --output_dir saves/preprocess_only \
  --trust_remote_code True \
  --image_max_pixels 589824 \
  --image_min_pixels 1024 \
  --video_max_pixels 65536 \
  --video_min_pixels 256 \
  --tokenized_path .llamafactory_cache/tokenized_dataset/BrickNet-MM-SFT-10000_qwen35-08b_nothink_len4096_img589824
```

只预处理时不需要保留 LoRA、adapter、学习率、epoch、batch size、optimizer、save_steps 等训练参数。

注意：当前 `run_sft()` 的顺序是先 `get_dataset(...)`，再 `load_model(...)`。所以 `--do_train False` 可以避免进入 `trainer.train(...)`，但数据处理完成后流程仍可能继续加载模型并初始化 trainer。

### 使用 tokenized dataset 进行 SFT

第一次使用某个新的 `--tokenized_path` 时：

```text
原始 dataset -> 预处理/tokenize -> save_to_disk(tokenized_path) -> SFT 训练
```

之后继续使用同一个 `--tokenized_path` 时：

```text
load_from_disk(tokenized_path) -> SFT 训练
```

因此推荐把关键预处理参数写进目录名，例如：

```text
BrickNet-MM-SFT-10000_qwen35-08b_nothink_len4096_img589824
```

这样可以避免不同预处理参数生成的数据混用。

## SFT启动方式

### CLI 和 WebUI 启动 SFT 的调用关系

直接命令行训练：

```text
llamafactory-cli
-> src/llamafactory/cli.py::main()
-> src/llamafactory/launcher.py::launch()
-> command == "train"
-> src/llamafactory/train/tuner.py::run_exp()
-> src/llamafactory/hparams/parser.py::get_train_args()
-> src/llamafactory/train/tuner.py::_training_function()
-> finetuning_args.stage == "sft"
-> src/llamafactory/train/sft/workflow.py::run_sft()
-> src/llamafactory/train/sft/trainer.py::CustomSeq2SeqTrainer
-> trainer.train(...)
```

WebUI 点击开始训练：

```text
python src/webui.py
或 llamafactory-cli webui
-> src/llamafactory/webui/interface.py::create_ui()
-> src/llamafactory/webui/components/train.py::create_train_tab()
-> start_btn.click(engine.runner.run_train, ...)
-> src/llamafactory/webui/runner.py::Runner.run_train()
-> Runner._launch()
-> Runner._parse_train_args()
-> save_cmd(args)，保存 training_args.yaml
-> Popen(["llamafactory-cli", "train", training_args.yaml])
```

然后 WebUI 拉起的新子进程会进入和 CLI 一样的训练链路：

```text
llamafactory-cli train training_args.yaml
-> cli.py::main()
-> launcher.py::launch()
-> train/tuner.py::run_exp()
-> parser.py::read_args()
-> parser.py::get_train_args()
-> tuner.py::_training_function()
-> sft/workflow.py::run_sft()
-> CustomSeq2SeqTrainer(...)
-> trainer.train(...)
```

结论：CLI 是直接进入训练入口；WebUI 是先把网页表单参数转成 YAML，再通过 `Popen` 启动同一个 `llamafactory-cli train` 训练入口。真正的 SFT 主逻辑都是 `src/llamafactory/train/sft/workflow.py::run_sft()`。

### SFT 主流程

进入 `run_sft()` 后，CLI 和 WebUI 没有本质区别：

```text
run_sft()
-> load_tokenizer()
-> get_template_and_fix_tokenizer()
-> get_dataset()
   -> 如果 tokenized_path 存在：load_from_disk()
   -> 否则：读取原始 dataset，map 预处理/tokenize，可选 save_to_disk()
-> load_model()
-> SFTDataCollatorWith4DAttentionMask(...)
-> CustomSeq2SeqTrainer(...)
-> trainer.train(...)
-> trainer.save_model()
-> trainer.save_metrics()
-> trainer.save_state()
-> plot_loss()
```
