
# notes

- 统一实验进度与结果：[experiment_results.md](experiment_results.md)。
- 唯一总推进计划：`/home/jiahao/task/BrickNet/BrickNet-MM Agentic LEGO Planner/Constructor Plan.md`。
- 执行约定：以下命令均从 `/home/jiahao/task/LlamaFactory` 目录启动；LlamaFactory 训练/预测命令显式通过
  `conda run -n llamafactory --no-capture-output` 使用项目环境，不依赖当前 shell 的 Python/Transformers 版本。
- 已完成实验的 train/predict 行是可复现实验命令；安全 launcher 在对应输出目录已存在时会按设计拒绝覆盖。
  评测命令可直接执行，其中 Stage-2 统一 evaluator 会复用完整结果，显式传入 `--force` 才会重算。
- 可直接复制的 shell 命令块中，命令前的 `# ...` 是用途说明；注释不会被 shell 执行。
- 2026-08-05 决策：66,456 条基础处理池不再四分；统一使用 512 VAL；policy-specific full hard mining 暂停，
  条件性恢复时最多处理 2,000 prompts。

    Qwen3-VL-2B-Instruct
    Qwen3.5-0.8B
    Qwen3.5-2B
    # 下载 Qwen3.5-2B-Base 权重。
    hf download Qwen/Qwen3.5-2B-Base



    历史 tokenized cache：.llamafactory_cache/tokenized_dataset/BrickNet-MM-SFT_qwen35-08b_nothink_len4096_img589824

    # 下载 Qwen3-VL-4B-Instruct 权重。
    hf download Qwen/Qwen3-VL-4B-Instruct
    # 下载 Qwen3-VL-8B-Instruct 权重。
    hf download Qwen/Qwen3-VL-8B-Instruct
    # 下载 Qwen3-VL-32B-Instruct 权重。
    hf download Qwen/Qwen3-VL-32B-Instruct
    # 下载 Qwen3.5-4B-Base 权重。
    hf download Qwen/Qwen3.5-4B-Base
    # 下载 Qwen3.5-4B 权重。
    hf download Qwen/Qwen3.5-4B
    # 下载 Qwen3.5-9B 权重。
    hf download Qwen/Qwen3.5-9B
    # 下载 Qwen3.5-27B 权重。
    hf download Qwen/Qwen3.5-27B
    # 下载 Stage3 教师 Qwen3.6-27B 权重。
    hf download Qwen/Qwen3.6-27B


# PT experiments
## exp0
### Qwen3.5-0.8B
**train**
训练输出目录：`saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64`
**eval**
生成 PT-exp0 验证集预测：
1. `conda run -n llamafactory --no-capture-output llamafactory-cli train examples/train_lora/qwen35_08b_bricknet_mm_pt_predict.yaml`
运行 PT-exp0 的 BrickNet 文本、结构和图文评测：
2. `cd ../BrickNet && /home/jiahao/miniconda3/envs/bricknet/bin/python scripts/evaluate_experiment.py --predictions ../LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20/generated_predictions.jsonl --text-metrics ../LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20/predict_results.json --output-dir outputs_val/qwen35_08b/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20 --prompts-file data/bricknet_datasets/captions_val.jsonl`
把预测与标签转换为 alignment worker 输入：
3. `jq -c '{response: .predict, label: .label}' saves/Qwen3.5-0.8B-Thinking/lora/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20/generated_predictions.jsonl > ../BrickNet/outputs_val/qwen35_08b/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20/alignment_input.jsonl`
运行 PT-exp0 的 pose/alignment 指标评测：
4. `cd ../ms-swift && /home/jiahao/miniconda3/envs/bricknet/bin/python examples/train/grpo/plugin/bricknet/evaluate_experiment.py alignment-worker --results ../BrickNet/outputs_val/qwen35_08b/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20/alignment_input.jsonl --dataset ../BrickNet/outputs_preprocess/BrickNet-MM/sharegpt/BrickNet-MM_VAL.jsonl --scored ../BrickNet/outputs_val/qwen35_08b/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20/scored.jsonl --metrics-json ../BrickNet/outputs_val/qwen35_08b/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20/metrics.json --metrics-md ../BrickNet/outputs_val/qwen35_08b/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20/metrics.md --output ../BrickNet/outputs_val/qwen35_08b/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20/alignment.jsonl --bricknet-root ../BrickNet`

结果：BLEU-4 `91.2174`，ROUGE-L `55.4362`，Parsable `310/512 (60.55%)`，Clean
`78/512 (15.23%)`，Collision `5.2188`，PE `0.2823`，SigLIP2 `0.8007`，VQA
`0.7604`，Inventory F1 `0.8253`，Pose Match `0.1418`，Dense Reward `0.5355`，
Strict Success `14/512 (2.73%)`。完整结果见
`../BrickNet/outputs_val/qwen35_08b/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20/metrics.json`。

## exp1
### Qwen3.5-0.8B
**train**
# 训练 mixed PT-exp1 adapter。
  conda run -n llamafactory --no-capture-output llamafactory-cli train examples/train_lora/qwen35_08b_bricknet_mixed_pt.yaml
**eval**
1) LlamaFactory 生成预测
# 使用 mixed PT-exp1 adapter 生成验证集预测。
conda run -n llamafactory --no-capture-output llamafactory-cli train examples/train_lora/qwen35_08b_bricknet_mm_pt_exp1_predict.yaml
2) BrickNet 文本+渲染评测
# 运行 PT-exp1 的 BrickNet 文本、结构和图文评测。
cd ../BrickNet && /home/jiahao/miniconda3/envs/bricknet/bin/python scripts/evaluate_experiment.py \
  --predictions ../LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/eval_PT_exp1_ptval_in4096_out4096_p95_t1_k20/generated_predictions.jsonl \
  --text-metrics ../LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/eval_PT_exp1_ptval_in4096_out4096_p95_t1_k20/predict_results.json \
  --output-dir outputs_val/qwen35_08b/eval_PT_exp1_ptval_in4096_out4096_p95_t1_k20 \
  --prompts-file data/bricknet_datasets/captions_val.jsonl
3) 生成 alignment 输入
# 把 PT-exp1 预测与标签转换为 alignment worker 输入。
cd ../LlamaFactory && jq -c '{response: .predict, label: .label}' \
  saves/Qwen3.5-0.8B-Thinking/lora/eval_PT_exp1_ptval_in4096_out4096_p95_t1_k20/generated_predictions.jsonl \
  > ../BrickNet/outputs_val/qwen35_08b/eval_PT_exp1_ptval_in4096_out4096_p95_t1_k20/alignment_input.jsonl
4) ms-swift 对齐评测
# 运行 PT-exp1 的 pose/alignment 指标评测。
cd ../ms-swift && /home/jiahao/miniconda3/envs/bricknet/bin/python \
  examples/train/grpo/plugin/bricknet/evaluate_experiment.py alignment-worker \
  --results ../BrickNet/outputs_val/qwen35_08b/eval_PT_exp1_ptval_in4096_out4096_p95_t1_k20/alignment_input.jsonl \
  --dataset ../BrickNet/outputs_preprocess/BrickNet-MM/sharegpt/BrickNet-MM_VAL.jsonl \
  --scored ../BrickNet/outputs_val/qwen35_08b/eval_PT_exp1_ptval_in4096_out4096_p95_t1_k20/scored.jsonl \
  --metrics-json ../BrickNet/outputs_val/qwen35_08b/eval_PT_exp1_ptval_in4096_out4096_p95_t1_k20/metrics.json \
  --metrics-md ../BrickNet/outputs_val/qwen35_08b/eval_PT_exp1_ptval_in4096_out4096_p95_t1_k20/metrics.md \
  --output ../BrickNet/outputs_val/qwen35_08b/eval_PT_exp1_ptval_in4096_out4096_p95_t1_k20/alignment.jsonl \
  --bricknet-root ../BrickNet



# SFT experiments
## exp0
### Qwen3-VL-2B-Instruct
- Debug LlamaFactory训练
- BrickNet-MM-VAL过拟合训练

## exp1
- BrickNet-MM-VAL过拟合训练
- 跑通LlamaFactory训练和推理

### Qwen3-VL-2B-Instruct
**train**
训练输出目录：`saves/Qwen3-VL-2B-Instruct/lora/train_exp1_qwen3vl_2b_val_ep10_bs1_ga8_lora16`
**eval**
使用 Qwen3-VL-2B-Instruct exp1 adapter 生成 VAL 预测：
1. conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3-VL-2B-Instruct --preprocessing_num_workers 16 --finetuning_type lora --quantization_method bnb --template qwen3_vl_nothink --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 --per_device_eval_batch_size 1 --predict_with_generate True --report_to none --max_new_tokens 512 --top_p 0.95 --temperature 1 --output_dir saves/Qwen3-VL-2B-Instruct/lora/eval_exp1_in4096_out512_p095_k20_t1 --trust_remote_code True --ddp_timeout 180000000 --do_predict True --adapter_name_or_path saves/Qwen3-VL-2B-Instruct/lora/train_exp1_qwen3vl_2b_val_ep10_bs1_ga8_lora16 --top_k 20
补充结果目录：`saves/Qwen3-VL-2B-Instruct/lora/eval_exp1_in4097_out512_p09_t095`。

### Qwen3.5-0.8B
**train**
训练输出目录：`saves/Qwen3.5-0.8B-Thinking/lora/train_exp1_qwen35_08b_val_ep10_bs1_ga8_lora16`
**eval**
使用 Qwen3.5-0.8B exp1 adapter 生成 VAL 预测：
conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --quantization_method bnb --template qwen3_5_nothink --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 --per_device_eval_batch_size 4 --predict_with_generate True --report_to none --max_new_tokens 512 --top_p 0.95 --temperature 1 --output_dir saves/Qwen3.5-0.8B-Thinking/lora/eval_exp1_in4096_out512_p095_k20_t1 --trust_remote_code True --ddp_timeout 180000000 --do_predict True --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_exp1_qwen35_08b_val_ep10_bs1_ga8_lora16 --top_k 20

### Qwen3.5-2B
**train**
训练输出目录：`saves/Qwen3.5-2B-Thinking/lora/train_exp1_qwen35_2b_val_ep10_bs1_ga8_lora16`
**eval**
使用 Qwen3.5-2B exp1 adapter 生成 VAL 预测：
conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3.5-2B --preprocessing_num_workers 16 --finetuning_type lora --quantization_method bnb --template qwen3_5_nothink --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 --per_device_eval_batch_size 1 --predict_with_generate True --report_to none --max_new_tokens 512 --top_p 0.95 --temperature 1 --output_dir saves/Qwen3.5-2B-Thinking/lora/eval_exp1_in4096_out512_p095_k20_t1 --trust_remote_code True --ddp_timeout 180000000 --do_predict True --top_k 20 --adapter_name_or_path saves/Qwen3.5-2B-Thinking/lora/train_exp1_qwen35_2b_val_ep10_bs1_ga8_lora16

## exp1_1
### Qwen3.5-2B
**train**
训练输出目录：`saves/Qwen3.5-2B-Thinking/lora/train_exp1_1_qwen35_2b_val_ep20_bs4_ga8_lora32_lr1e5_schdlconstanwarm`
**eval**
以 temperature=1、top-p=0.95 生成 exp1_1 VAL 预测：
1. conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3.5-2B --preprocessing_num_workers 16 --finetuning_type lora --quantization_method bnb --template qwen3_5_nothink --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 --per_device_eval_batch_size 1 --predict_with_generate True --report_to none --max_new_tokens 512 --top_p 0.95 --temperature 1 --output_dir saves/Qwen3.5-2B-Thinking/lora/eval_exp1_1_in4096_out512_p095_k20_t1 --trust_remote_code True --ddp_timeout 180000000 --do_predict True --top_k 20 --adapter_name_or_path saves/Qwen3.5-2B-Thinking/lora/train_exp1_1_qwen35_2b_val_ep20_bs4_ga8_lora32_lr1e5_schdlconstanwarm

以 temperature=0.95、top-p=0.9 生成 exp1_1 采样消融预测：
2. conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3.5-2B --preprocessing_num_workers 16 --finetuning_type lora --quantization_method bnb --template qwen3_5_nothink --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 --per_device_eval_batch_size 1 --predict_with_generate True --report_to none --max_new_tokens 512 --top_p 0.9 --temperature 0.95 --output_dir saves/Qwen3.5-2B-Thinking/lora/eval_exp1_1_in4097_out512_p09_t095 --trust_remote_code True --ddp_timeout 180000000 --do_predict True --adapter_name_or_path saves/Qwen3.5-2B-Thinking/lora/train_exp1_1_qwen35_2b_val_ep20_bs4_ga8_lora32_lr1e5_schdlconstanwarm

## exp2
### Qwen3.5-0.8B
- bricknet-mm sft 1w
**train**
训练 exp2 的 BrickNet-MM-SFT 10k adapter：
1. conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --do_train True --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --template qwen3_5_nothink --flash_attn auto --dataset_dir data --dataset BrickNet-MM-SFT --cutoff_len 4096 --learning_rate 5e-05 --num_train_epochs 3.0 --max_samples 10000 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --lr_scheduler_type cosine --max_grad_norm 1.0 --logging_steps 10 --save_steps 500 --warmup_steps 0 --packing False --enable_thinking False --report_to none --output_dir saves/Qwen3.5-0.8B-Thinking/lora/train_exp2_qwen35_08b_sft1w_ep3_bs2_ga8_lora64 --bf16 True --plot_loss True --trust_remote_code True --ddp_timeout 180000000 --include_num_input_tokens_seen True --optim adamw_torch --lora_rank 64 --lora_alpha 128 --lora_dropout 0 --lora_target all --freeze_vision_tower True --freeze_multi_modal_projector True --image_max_pixels 589824 --image_min_pixels 1024 --video_max_pixels 65536 --video_min_pixels 256 --tokenized_path .llamafactory_cache/tokenized_dataset/BrickNet-MM-SFT-10000_qwen35-08b_nothink_len4096_img589824

**eval**
使用 exp2 adapter 生成 VAL 预测：
1. conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --quantization_method bnb --template qwen3_5_nothink --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 --per_device_eval_batch_size 1 --predict_with_generate True --report_to none --max_new_tokens 4096 --top_p 0.95 --temperature 1 --output_dir saves/Qwen3.5-0.8B-Thinking/lora/eval_exp2_in4096_out4096_p95_t1_k20 --trust_remote_code True --ddp_timeout 180000000 --do_predict True --top_k 20 --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_exp2_qwen35_08b_sft1w_ep3_bs2_ga8_lora64

## exp2_1
### Qwen3.5-0.8B
- bricknet-mm sft 5w
**train**
训练 exp2_1 的 BrickNet-MM-SFT 50k adapter：
1. conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --do_train True --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --template qwen3_5_nothink --flash_attn auto --dataset_dir data --dataset BrickNet-MM-SFT --cutoff_len 4096 --learning_rate 5e-05 --num_train_epochs 3.0 --max_samples 50000 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --lr_scheduler_type cosine --max_grad_norm 1.0 --logging_steps 10 --save_steps 500 --warmup_steps 0 --packing False --enable_thinking False --report_to none --output_dir saves/Qwen3.5-0.8B-Thinking/lora/train_exp2_1_qwen35_08b_sft5w_ep3_bs2_ga8_lora64 --bf16 True --plot_loss True --trust_remote_code True --ddp_timeout 180000000 --include_num_input_tokens_seen True --optim adamw_torch --lora_rank 64 --lora_alpha 128 --lora_dropout 0 --lora_target all --freeze_vision_tower True --freeze_multi_modal_projector True --image_max_pixels 589824 --image_min_pixels 1024 --video_max_pixels 65536 --video_min_pixels 256 --tokenized_path .llamafactory_cache/tokenized_dataset/BrickNet-MM-SFT-50000_qwen35-08b_nothink_len4096_img589824

**eval**
使用 exp2_1 adapter 生成 VAL 预测：
1. conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --quantization_method bnb --template qwen3_5_nothink --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 --per_device_eval_batch_size 1 --predict_with_generate True --report_to none --max_new_tokens 4096 --top_p 0.95 --temperature 1 --output_dir saves/Qwen3.5-0.8B-Thinking/lora/eval_exp2_1_in4096_out4096_p95_t1_k20 --trust_remote_code True --ddp_timeout 180000000 --do_predict True --top_k 20 --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_exp2_1_qwen35_08b_sft5w_ep3_bs2_ga8_lora64

## exp2_2
### Qwen3.5-0.8B
- bricknet-mm sft all
**train**
训练 exp2_2 的 BrickNet-MM-SFT 全量 adapter：
1. conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --do_train True --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --template qwen3_5_nothink --flash_attn auto --dataset_dir data --dataset BrickNet-MM-SFT --cutoff_len 4096 --learning_rate 5e-05 --num_train_epochs 3.0 --max_samples 1000000 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --lr_scheduler_type cosine --max_grad_norm 1.0 --logging_steps 10 --save_steps 2000 --warmup_steps 0 --packing False --enable_thinking False --report_to none --output_dir saves/Qwen3.5-0.8B-Thinking/lora/train_exp2_2_qwen35_08b_sft_ep3_bs2_ga8_lora64 --bf16 True --plot_loss True --trust_remote_code True --ddp_timeout 180000000 --include_num_input_tokens_seen True --optim adamw_torch --lora_rank 64 --lora_alpha 128 --lora_dropout 0 --lora_target all --freeze_vision_tower True --freeze_multi_modal_projector True --image_max_pixels 589824 --image_min_pixels 1024 --video_max_pixels 65536 --video_min_pixels 256 --tokenized_path .llamafactory_cache/tokenized_dataset/BrickNet-MM-SFT_qwen35-08b_nothink_len4096_img589824

**eval**


## exp3
### Qwen3.5-0.8B-PT
- bricknet-mm sft 1w
- epoch 3
**train**
从 PT-exp0 初始化并训练 exp3 的 10k SFT 新 adapter：
1. conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --do_train True --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --template qwen3_5_nothink --flash_attn auto --dataset_dir data --dataset BrickNet-MM-SFT --cutoff_len 4096 --learning_rate 5e-05 --num_train_epochs 3.0 --max_samples 10000 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --lr_scheduler_type cosine --max_grad_norm 1.0 --logging_steps 10 --save_steps 500 --warmup_steps 0 --packing False --enable_thinking False --report_to none --output_dir saves/Qwen3.5-0.8B-Thinking/lora/train_exp3_qwen35_08b_pt_sft1w_ep3_bs2_ga8_lora64 --bf16 True --plot_loss True --trust_remote_code True --ddp_timeout 180000000 --include_num_input_tokens_seen True --optim adamw_torch --lora_rank 64 --lora_alpha 128 --lora_dropout 0 --lora_target all --freeze_vision_tower True --freeze_multi_modal_projector True --image_max_pixels 589824 --image_min_pixels 1024 --video_max_pixels 65536 --video_min_pixels 256 --tokenized_path .llamafactory_cache/tokenized_dataset/BrickNet-MM-SFT-10000_qwen35-08b_nothink_len4096_img589824 --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64 --create_new_adapter

**eval**
仅使用 PT-exp0 adapter 生成对照 VAL 预测：
1. conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --quantization_method bnb --template qwen3_5_nothink --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 --per_device_eval_batch_size 1 --predict_with_generate True --report_to none --max_new_tokens 4096 --top_p 0.95 --temperature 1 --output_dir saves/Qwen3.5-0.8B-Thinking/lora/eval_exp3_PT_in4096_out4096_p95_t1_k20 --trust_remote_code True --ddp_timeout 180000000 --do_predict True --top_k 20 --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64

串联 PT-exp0 与 exp3 SFT adapter 生成 VAL 预测：
2. conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --quantization_method bnb --template qwen3_5_nothink --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 --per_device_eval_batch_size 1 --predict_with_generate True --report_to none --max_new_tokens 4096 --top_p 0.95 --temperature 1 --output_dir saves/Qwen3.5-0.8B-Thinking/lora/eval_exp3_in4096_out4096_p95_t1_k20 --trust_remote_code True --ddp_timeout 180000000 --do_predict True --top_k 20 --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64,saves/Qwen3.5-0.8B-Thinking/lora/train_exp3_qwen35_08b_pt_sft1w_ep3_bs2_ga8_lora64

## exp3_0_1
### Qwen3.5-0.8B-PT
- bricknet-mm sft 1w
- epoch 10
**train**
从 PT-exp0 初始化并训练 exp3_0_1 的 10-epoch SFT adapter：
1. conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --do_train True --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --template qwen3_5_nothink --flash_attn auto --dataset_dir data --dataset BrickNet-MM-SFT --cutoff_len 4096 --learning_rate 5e-05 --num_train_epochs 10 --max_samples 10000 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --lr_scheduler_type cosine --max_grad_norm 1.0 --logging_steps 10 --save_steps 500 --warmup_steps 0 --packing False --enable_thinking False --report_to none --output_dir saves/Qwen3.5-0.8B-Thinking/lora/train_exp3_0_1_qwen35_08b_pt_sft1w_ep10_bs2_ga8_lora64 --bf16 True --plot_loss True --trust_remote_code True --ddp_timeout 180000000 --include_num_input_tokens_seen True --optim adamw_torch --lora_rank 64 --lora_alpha 128 --lora_dropout 0 --lora_target all --freeze_vision_tower True --freeze_multi_modal_projector True --image_max_pixels 589824 --image_min_pixels 1024 --video_max_pixels 65536 --video_min_pixels 256 --tokenized_path .llamafactory_cache/tokenized_dataset/BrickNet-MM-SFT-10000_qwen35-08b_nothink_len4096_img589824 --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64 --create_new_adapter

**eval**
使用 exp3_0_1 adapter 生成 VAL 预测：
1. conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --quantization_method bnb --template qwen3_5_nothink --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 --per_device_eval_batch_size 1 --predict_with_generate True --report_to none --max_new_tokens 4096 --top_p 0.95 --temperature 1 --output_dir saves/Qwen3.5-0.8B-Thinking/lora/eval_exp3_0_1_in4096_out4096_p95_t1_k20 --trust_remote_code True --ddp_timeout 180000000 --do_predict True --top_k 20 --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64,saves/Qwen3.5-0.8B-Thinking/lora/train_exp3_0_1_qwen35_08b_pt_sft1w_ep10_bs2_ga8_lora64

运行 exp3_0_1 的 BrickNet 统一评测：
2. cd ../BrickNet && /home/jiahao/miniconda3/envs/bricknet/bin/python scripts/evaluate_experiment.py --predictions ../LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/eval_exp3_0_1_in4096_out4096_p95_t1_k20/generated_predictions.jsonl --text-metrics ../LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/eval_exp3_0_1_in4096_out4096_p95_t1_k20/predict_results.json --output-dir outputs_val/qwen35_08b/eval_exp3_0_1_in4096_out4096_p95_t1_k20 --prompts-file data/bricknet_datasets/captions_val.jsonl


## exp3_1
### Qwen3.5-0.8B-PT
- bricknet-mm sft 5w
- epoch 3
**train**
从 PT-exp0 初始化并训练 exp3_1 的 50k SFT adapter：
1. conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --do_train True --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --template qwen3_5_nothink --flash_attn auto --dataset_dir data --dataset BrickNet-MM-SFT --cutoff_len 4096 --learning_rate 5e-05 --num_train_epochs 3.0 --max_samples 50000 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --lr_scheduler_type cosine --max_grad_norm 1.0 --logging_steps 10 --save_steps 1000 --warmup_steps 0 --packing False --enable_thinking False --report_to none --output_dir saves/Qwen3.5-0.8B-Thinking/lora/train_exp3_1_qwen35_08b_pt_sft5w_ep3_bs2_ga8_lora64 --bf16 True --plot_loss True --trust_remote_code True --ddp_timeout 180000000 --include_num_input_tokens_seen True --optim adamw_torch --lora_rank 64 --lora_alpha 128 --lora_dropout 0 --lora_target all --freeze_vision_tower True --freeze_multi_modal_projector True --image_max_pixels 589824 --image_min_pixels 1024 --video_max_pixels 65536 --video_min_pixels 256 --tokenized_path .llamafactory_cache/tokenized_dataset/BrickNet-MM-SFT-50000_qwen35-08b_nothink_len4096_img589824 --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64 --create_new_adapter

**eval**
使用 exp3_1 adapter 生成 VAL 预测：
1. `conda run -n llamafactory --no-capture-output llamafactory-cli train examples/train_lora/qwen35_08b_bricknet_mm_exp3_1_predict.yaml`
运行 exp3_1 的 BrickNet 文本、结构和图文评测：
2. `cd ../BrickNet && /home/jiahao/miniconda3/envs/bricknet/bin/python scripts/evaluate_experiment.py --predictions ../LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/eval_exp3_1_in4096_out4096_p95_t1_k20/generated_predictions.jsonl --text-metrics ../LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/eval_exp3_1_in4096_out4096_p95_t1_k20/predict_results.json --output-dir outputs_val/qwen35_08b/eval_exp3_1_in4096_out4096_p95_t1_k20 --prompts-file data/bricknet_datasets/captions_val.jsonl --render-jobs 8 --eval-workers 8 --eval-batch-size 8`
把 exp3_1 预测与标签转换为 alignment worker 输入：
3. `jq -c '{response: .predict, label: .label}' saves/Qwen3.5-0.8B-Thinking/lora/eval_exp3_1_in4096_out4096_p95_t1_k20/generated_predictions.jsonl > ../BrickNet/outputs_val/qwen35_08b/eval_exp3_1_in4096_out4096_p95_t1_k20/alignment_input.jsonl`
运行 exp3_1 的 pose/alignment 指标评测：
4. `cd ../ms-swift && /home/jiahao/miniconda3/envs/bricknet/bin/python examples/train/grpo/plugin/bricknet/evaluate_experiment.py alignment-worker --results ../BrickNet/outputs_val/qwen35_08b/eval_exp3_1_in4096_out4096_p95_t1_k20/alignment_input.jsonl --dataset ../BrickNet/outputs_preprocess/BrickNet-MM/sharegpt/BrickNet-MM_VAL.jsonl --scored ../BrickNet/outputs_val/qwen35_08b/eval_exp3_1_in4096_out4096_p95_t1_k20/scored.jsonl --metrics-json ../BrickNet/outputs_val/qwen35_08b/eval_exp3_1_in4096_out4096_p95_t1_k20/metrics.json --metrics-md ../BrickNet/outputs_val/qwen35_08b/eval_exp3_1_in4096_out4096_p95_t1_k20/metrics.md --output ../BrickNet/outputs_val/qwen35_08b/eval_exp3_1_in4096_out4096_p95_t1_k20/alignment.jsonl --bricknet-root ../BrickNet`

结果：BLEU-4 `92.1417`，ROUGE-L `56.0046`，Parsable `367/512 (71.68%)`，Clean
`93/512 (18.16%)`，Collision `6.0332`，PE `0.2854`，SigLIP2 `0.8190`，VQA
`0.7666`，Inventory F1 `0.8932`，Pose Match `0.1617`，Dense Reward `0.5765`，
Strict Success `20/512 (3.91%)`。完整结果见
`../BrickNet/outputs_val/qwen35_08b/eval_exp3_1_in4096_out4096_p95_t1_k20/metrics.json`。


## exp3_2
### Qwen3.5-0.8B-PT
- bricknet-mm sft all
- epoch 3
**train**
从 PT-exp0 初始化并训练 exp3_2 的全量 SFT adapter：
1. conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --do_train True --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --template qwen3_5_nothink --flash_attn auto --dataset_dir data --dataset BrickNet-MM-SFT --cutoff_len 4096 --learning_rate 5e-05 --num_train_epochs 3.0 --max_samples 1000000 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --lr_scheduler_type cosine --max_grad_norm 1.0 --logging_steps 10 --save_steps 2000 --warmup_steps 0 --packing False --enable_thinking False --report_to none --output_dir saves/Qwen3.5-0.8B-Thinking/lora/train_exp3_2_qwen35_08b_pt_sft_ep3_bs2_ga8_lora64 --bf16 True --plot_loss True --trust_remote_code True --ddp_timeout 180000000 --include_num_input_tokens_seen True --optim adamw_torch --lora_rank 64 --lora_alpha 128 --lora_dropout 0 --lora_target all --freeze_vision_tower True --freeze_multi_modal_projector True --image_max_pixels 589824 --image_min_pixels 1024 --video_max_pixels 65536 --video_min_pixels 256 --tokenized_path .llamafactory_cache/tokenized_dataset/BrickNet-MM-SFT_qwen35-08b_nothink_len4096_img589824 --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64 --create_new_adapter

**eval**



## Stage 2 exp4–exp4_3（四个训练和推理完成，10k全指标完成）
四个实验均以已完成的 mixed PT-exp1 final 为共同初始化。`exp4/exp4_1` overfit 训练与 VAL512 推理完成，
机械 gate 已批准；`exp4_2/exp4_3` 10k 训练与 VAL512 推理均完成。50k/all 继续暂停且不分配版本号。


### exp4 — NonThinking-Control VAL511 overfit
状态：训练完成，train loss `0.1737931`；VAL512 512/512 推理完成。
**train**
训练 exp4 NonThinking-Control VAL511 overfit adapter：
1. `python scripts/launch_bricknet_stage2_sft.py --action train --variant nonthinking-control --scale overfit511 --execute --stage0-gate-approved`
**predict VAL512**
使用 exp4 adapter 生成完整 VAL512 预测：
1. `python scripts/launch_bricknet_stage2_sft.py --action predict --variant nonthinking-control --scale overfit511 --execute --stage0-gate-approved`
**eval**
统一评测 exp4 的 VAL512 结果：
1. `python scripts/evaluate_bricknet_stage2.py --experiment exp4 --execute`


### exp4_1 — Thinking-Hard VAL511 overfit
状态：训练完成，train loss `0.0859583`；VAL512 512/512 推理完成。
**train**
训练 exp4_1 Thinking-Hard VAL511 overfit adapter：
1. `python scripts/launch_bricknet_stage2_sft.py --action train --variant thinking-hard --scale overfit511 --execute --stage0-gate-approved`
**predict VAL512**
使用 exp4_1 adapter 生成完整 VAL512 trace：
1. `python scripts/launch_bricknet_stage2_sft.py --action predict --variant thinking-hard --scale overfit511 --execute --stage0-gate-approved`
**eval**
提取 path 并统一评测 exp4_1 的 VAL512 结果：
1. `python scripts/evaluate_bricknet_stage2.py --experiment exp4_1 --execute`


### exp4_2 — NonThinking-Control 10k
状态：训练完成，train loss `0.1726632`；VAL512 512/512 推理和全指标完成。parsable
`382/512 (74.61%)`、clean `93/512 (18.16%)`、dense reward `0.58159`、strict success `16/512 (3.12%)`。
**train**
训练 exp4_2 NonThinking-Control 10k adapter：
1. `python scripts/launch_bricknet_stage2_sft.py --action train --variant nonthinking-control --scale 10k --execute --stage0-gate-approved --overfit-gate-approved`
**predict VAL512**
使用 exp4_2 adapter 生成完整 VAL512 预测：
1. `python scripts/launch_bricknet_stage2_sft.py --action predict --variant nonthinking-control --scale 10k --execute --stage0-gate-approved --overfit-gate-approved`
**eval**
统一评测 exp4_2 的 VAL512 结果：
1. `python scripts/evaluate_bricknet_stage2.py --experiment exp4_2 --execute`


### exp4_3 — Thinking-Hard 10k
状态：训练完成，train loss `0.0433687`；VAL512 512/512 推理完成。strict extractor 得到
`360/512 (70.31%)` 完整合法 trace/path，512 条均有非空 extracted prefix；全指标完成：clean
`101/512 (19.73%)`、dense reward `0.57395`、strict success `13/512 (2.54%)`。相对 exp4_2 没有总体优势，
T1-10k 人工推广 gate 未批准。
**train**
训练 exp4_3 Thinking-Hard 10k adapter：
1. `python scripts/launch_bricknet_stage2_sft.py --action train --variant thinking-hard --scale 10k  --execute --stage0-gate-approved --overfit-gate-approved`
**predict VAL512**
使用 exp4_3 adapter 生成完整 VAL512 trace：
1. `python scripts/launch_bricknet_stage2_sft.py --action predict --variant thinking-hard --scale 10k --execute --stage0-gate-approved --overfit-gate-approved`
**eval**
提取 path 并统一评测 exp4_3 的 VAL512 结果：
1. `python scripts/evaluate_bricknet_stage2.py --experiment exp4_3 --execute`

统一 evaluator 会自动完成 strict path 提取、path BLEU/ROUGE、BrickNet 结构/渲染/图文指标和 alignment。默认
dry-run；已有完整结果时 `--execute` 安全复用并退出，只有显式增加 `--force` 才会重算。

结果：Trace format `360/512 (70.31%)`，Connectivity `360/512 (70.31%)`，Clean
`101/512 (19.73%)`，Collision `6.1738`，PE `0.2799`，SigLIP2 `0.7818`，VQAScore
`0.7486`，BLEU-4 `90.8840`，ROUGE-L `55.1668`，Inventory F1 `0.8812`，Pose Match
`0.1452`，Dense Reward `0.5739`，Strict Success `13/512 (2.54%)`。完整结果见
`../BrickNet/outputs_val/qwen35_08b/eval_exp4_3_stage2_thinking_hard_10k_val512_in16384_out16384_p95_t1_k20/metrics.json`。


### exp4_3_1 — Stage2 V2 Thinking-Hard Lean-State 10k

状态：正式 full pool 66,456、train 10,000、VAL512 已构造；两份真实 processor audit 均为 0 error、0 truncation，
train dry-run 已返回 `ready=true, blockers=[]`。训练、推理和评测尚未执行。该实验只运行 10k，不开放
overfit511/50k/all。

数据 SHA-256：train=`b0ee6b1046aaef6290ed7bb4d1b632c0260fbbd65048619c415e8669f9a6bc95`，
VAL512=`f102e74a2462e38af0cfcbcd9fd012772c7b3c0bc6fc44f2746430d57eec1009`；10k ordered-ID SHA-256 与
exp4_2/exp4_3 同为 `2d87ff4c3b918f748dde48721cbec66595ccc17317cf728f77e30efc04230dea`。

以下是当前状态可直接执行的完整剩余序列。严格按顺序运行；两个 dry-run 必须在相应 execute 前通过。

```bash
# 进入 LlamaFactory 仓库。
cd /home/jiahao/task/LlamaFactory

# 检查 exp4_3_1 训练 gate 和最终启动参数，不启动训练。
python scripts/launch_bricknet_stage2_sft.py \
  --action train --variant thinking-hard-v2-lean-state --scale 10k \
  --overfit-gate-approved

# 通过 gate 后正式训练 exp4_3_1 Lean-State 10k。
python scripts/launch_bricknet_stage2_sft.py \
  --action train --variant thinking-hard-v2-lean-state --scale 10k \
  --execute --stage0-gate-approved --overfit-gate-approved

# 检查 exp4_3_1 VAL512 推理 gate 和最终启动参数。
python scripts/launch_bricknet_stage2_sft.py \
  --action predict --variant thinking-hard-v2-lean-state --scale 10k

# 正式运行 exp4_3_1 的 VAL512 推理。
python scripts/launch_bricknet_stage2_sft.py \
  --action predict --variant thinking-hard-v2-lean-state --scale 10k \
  --execute --stage0-gate-approved

# 检查 exp4_3_1 统一评测计划，不实际执行。
python scripts/evaluate_bricknet_stage2.py --experiment exp4_3_1
# 正式运行 exp4_3_1 的统一评测。
python scripts/evaluate_bricknet_stage2.py --experiment exp4_3_1 --execute
```

训练输出固定为
`saves/Qwen3.5-0.8B-Thinking/lora/train_exp4_3_1_qwen35_08b_mixedpt_stage2_thinking_hard_v2_lean_state_10k_ep3_bs1_ga16_lora64_len16384`；
预测输出固定为
`saves/Qwen3.5-0.8B-Thinking/lora/eval_exp4_3_1_stage2_thinking_hard_v2_lean_state_10k_val512_in16384_out16384_p95_t1_k20`。
构造和 token audit 的可复现命令、报告路径与完整 schema 见 BrickNet 的
`Stage 2 V2 Lean-State Auto-Annotation.md`；当前不要重建或拿旧 Thinking-Hard v1 报告代替。



## Stage 3 exp5（dormant preparation，未训练）
### exp5 — Thinking-Semantic 10k
配置：

- train：`examples/train_lora/qwen35_08b_bricknet_stage3_exp5_thinking_semantic_10k.yaml`
- predict：`examples/train_lora/qwen35_08b_bricknet_stage3_exp5_thinking_semantic_predict.yaml`
- gate launcher：`scripts/launch_bricknet_stage3_sft.py`

exp5 与 exp4_3 使用同一 mixed PT-exp1 final 初始化、相同 10k ID、LoRA/optimizer/epoch/effective batch、
vision/projector freeze、`qwen3_5_nothink` 和 16,384 token 协议。唯一数据变量是 T2 trace 在 T1 硬事实中增加
经审计的 `cue/semantic_role`。

当前只准备配置，没有激活 `BrickNet-Stage3-Thinking-Semantic-10k` dataset registry。Qwen3.6-27B Pilot
配置和 teacher approval 已冻结，但首个 action 的两次教师输出均不是合法 JSON，只得到 fallback；ledger 为
`completed=1, pending=251, fallback=1`。Stage 3 暂停在单 action output gate，不能开始剩余 Pilot 或10k。train dry-run：

```bash
# 检查 Stage3 exp5 训练 gate；当前应报告尚未满足的阻塞项。
python scripts/launch_bricknet_stage3_sft.py --action train
```

正式执行必须同时通过 Stage 0 final、Stage 3 Pilot 人工 approval、exp4_3/T1-10k gate、T2-10k 完整 hard replay、
真实 Qwen processor paired token audit、10k dataset/registry hash 和输出目录 gate。预测还需要 exp5 adapter。
Stage 0 已通过；当前 dry-run 仍应被 Pilot 人工 approval、exp4_3/T1-10k paired gate、T2-10k 数据/replay/token
和 registry 阻断，`training_started=false`。50k/all 不分配版本号或配置。

## PT-exp2 & exp4_4/exp4_7（以及 dormant exp4_5/exp4_6）

当前 MM 活动序列为 `PT-exp2-text8m → PT-exp2-mm-e1/e2/e3-v2 → PT-exp2-v2 alias`，三轮和 alias
选择均已完成。旧 `PT-exp2 alias → exp4_4 10k → exp4_5 50k → exp4_6 all` 配置保持冻结，不会静默改绑 v2；
若下游采用 v2，必须另行批准并显式绑定 `PT-exp2-v2`。不创建 PT-exp2 VAL511 训练或验证。详细数据 hash、配置与
gate 见 [PT-exp2 runbook](bricknet-pt-exp2.md)。

### PT-exp2

MM consolidation 使用三个顺序训练配置：e1 从 text8m adapter 开始，e2 从 e1 final adapter 继续，e3 从 e2
final adapter 继续。冻结 v1 数据的 MM/replay `meta` 异构会使 Arrow 在第一条 replay 失败；首次 e1 因此
标记为 `invalidated`，在 tokenization/optimizer 前停止。v1 JSONL、registry、YAML、launcher、batch 和 nohup 日志均
原样保留。当前活动入口是新增 `v2-no-meta` 训练投影，只移除不输入模型的 `meta`，逐行保持
`id/messages/images` 与 v1 一致。三轮分别使用不重叠的 replay slice（`15,617/15,586/15,667` 条），target tokens 为
`26,285,287/26,285,922/26,284,707`，replay/MM ratio 为
`1.0000053/1.0000294/0.9999832`。每次训练 1 epoch，当前固定物理 CUDA 1 单卡 BS2/GA8/global batch 16；
e1 LR=`2e-5`，e2/e3 LR=`1e-5`。三轮训练 global/max 为 `9417/9417`、`9415/9415`、`9420/9420`，
每组 prediction/scored/alignment 均 `512/512`；按 `strict → dense → clean → parsable` 选择 e1。selection
record 为 `ready=true, executed=true, selected=recommended=mm-e1`，alias 精确指向 e1；完整 metrics/adapter
hash 见 [experiment_results](experiment_results.md)。

- text8m final：`saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp2_text8m_qwen35_08b_path7698261_steps250k_bs4_gbs32_lora64_len6401_nopack`
- MM e1 v2：`saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp2_mm_e1_v2_nometa_qwen35_08b_text8m_mm135k_replay1to1_ep1_bs2_gbs16_lora64_len6400`
- MM e2 v2：`saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp2_mm_e2_v2_nometa_qwen35_08b_text8m_mm135k_replay1to1_ep1_bs2_gbs16_lora64_len6400`
- MM e3 v2：`saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp2_mm_e3_v2_nometa_qwen35_08b_text8m_mm135k_replay1to1_ep1_bs2_gbs16_lora64_len6400`
- final alias v2：`saves/Qwen3.5-0.8B-Thinking/lora/PT-exp2-v2`
- 下游版本：同初始化 10k paired 为 `exp4_4=NonThinking-Control`、`exp4_7=Lean-State`；dormant Control 扩容为
  `exp4_5=50k`、`exp4_6=all 66,456`

下面是已完成链路的可复现手动执行序列；text8m 与 v2 MM 的训练、推理和评测全部使用物理 CUDA 1。
v2 使用独立 dataset registry、tokenized cache、adapter、prediction、evaluation 和 alias，不覆盖 v1 或任何旧实验。
text8m 的 with-length cache 现为 7,698,261 行，含 `input_ids/attention_mask/length`，launcher 复查
`eligible=true/build_required=false`。通用 PT/SFT 工具为
`scripts/build_tokenized_cache_with_length.py`：新实验 YAML
配置 `tokenized_path`、`train_sampling_strategy: group_by_length` 和 `length_column_name: length` 后，可单独执行
`python scripts/build_tokenized_cache_with_length.py --config <yaml>`；

    新实验第一次生成缓存时，直接运行：
```bash
cd /data/jiahao/task/LlamaFactory

conda run -n llamafactory --no-capture-output python \
scripts/build_tokenized_cache_with_length.py \
--config examples/train_lora/<实验配置>.yaml \
--num-proc 4
```

    迁移已有缓存时使用：
```bash
conda run -n llamafactory --no-capture-output python \
scripts/build_tokenized_cache_with_length.py \
--config examples/train_lora/<实验配置>.yaml \
--source-cache <旧缓存目录> \
--output-cache <新的带length缓存目录> \
--num-proc 4
```

    只执行缓存加载和 schema 验证，不构建缓存，独立运行：
```bash
python scripts/build_tokenized_cache_with_length.py \
--config <yaml> \
--check-only
```

也可直接执行实验 launcher。launcher 在
`--execute` 前发现目标不存在时，会以单进程自动预构建、复检，然后才启动单卡/DDP 训练。

YAML 缓存策略无需加入框架外参数：保留 `length_column_name: length` 选择 with-length 自动构建/校验；删除该字段
选择 LlamaFactory 普通缓存流程。dry-run 不构建，只报告 `build_required`；`--execute` 才会构建。已有目标若缺列
不会被自动覆盖，必须用上面的迁移命令生成新目录。自动构建默认 `--cache-num-proc 4`，必要时可从 launcher
命令行覆盖。

2026-08-13 补充 `datasets==4.0.0` 兼容：其 `dataset["length"]` 是懒加载 `Column`，原生 grouped sampler 的
随机标量索引会使 7,698,261 行排序长期停在 `0/250000`。PT/SFT trainer 现在只对“group_by_length 且实际存在
length 列”的数据，将 Arrow scalar column 一次性转为 NumPy 数组；实测数组 `29.37 MiB`、转换 `0.061s`、
全量 grouped index 构造 `4.384s`。普通无 length cache 仍由 Transformers 从 `input_ids` 推断，其他采样策略
也继续走父类实现。已在修改前运行的 trainer 需要重启才能加载新代码。

先重现或复核 v2 数据和 loader gate：

```bash
python scripts/build_bricknet_pt_exp2_mm_train_view_v2.py --verify-only
python scripts/launch_bricknet_pt_exp2_mm_v2.py --gpus 1 --action train --run mm-e1
```

```bash
cd /data/jiahao/task/LlamaFactory

# 已完成：从 text8m adapter 继续训练 MM e1。
python scripts/launch_bricknet_pt_exp2_mm_v2.py --gpus 1 --action train --run mm-e1 --execute
# 已完成：使用 MM e1 adapter 运行验证集推理。
python scripts/launch_bricknet_pt_exp2_mm_v2.py --gpus 1 --action predict --run mm-e1 --execute
# 已完成：评测 MM e1（base evaluator + alignment）。
python scripts/launch_bricknet_pt_exp2_mm_v2.py --gpus 1 --action evaluate --run mm-e1 --execute
# 已完成：从 MM e1 adapter 继续训练 MM e2。
python scripts/launch_bricknet_pt_exp2_mm_v2.py --gpus 1 --action train --run mm-e2 --execute
# 已完成：使用 MM e2 adapter 运行验证集推理。
python scripts/launch_bricknet_pt_exp2_mm_v2.py --gpus 1 --action predict --run mm-e2 --execute
# 已完成：e2 评测 preflight，确认 512 条 prediction 与标准 VAL512 reference 逐行对齐。
python scripts/launch_bricknet_pt_exp2_mm_v2.py --gpus 1 --action evaluate --run mm-e2
# 已完成：评测 MM e2（base evaluator + alignment）。
python scripts/launch_bricknet_pt_exp2_mm_v2.py --gpus 1 --action evaluate --run mm-e2 --execute
# 已完成：从 MM e2 adapter 继续训练 MM e3。
python scripts/launch_bricknet_pt_exp2_mm_v2.py --gpus 1 --action train --run mm-e3 --execute
# 已完成：使用 MM e3 adapter 运行验证集推理。
python scripts/launch_bricknet_pt_exp2_mm_v2.py --gpus 1 --action predict --run mm-e3 --execute
# 已完成：评测 MM e3（base evaluator + alignment）。
python scripts/launch_bricknet_pt_exp2_mm_v2.py --gpus 1 --action evaluate --run mm-e3 --execute
# 已完成：按固定排序选择 v2 final alias（e1 胜出）。
python scripts/launch_bricknet_pt_exp2_mm_v2.py --action select-final --run mm-e3 --execute --approve
```

MM v2 的 evaluate 不再以只有 `metrics.json/structure` 作为完成条件。完整结果必须同时包含 512-row
`task_alignment` 和 `condition_generation`，其中 Dense Reward、Strict Success、固定 reward weights 与 pose
tolerance 均通过校验；`alignment_manifest.json` 还绑定 prediction、scored、标准
`BrickNet-MM_VAL.jsonl`、base manifest、alignment evaluator 及产物 hash。当前 e1/e2/e3 三轮均已通过该 gate，
selection record 已执行并创建 alias。2026-08-20 e1 曾由同一 launcher 执行 alignment-only 回填；随后 e2/e3
也完成了各自 prediction/evaluate。原 2026-08-21 14:06 batch 在 e3 PE 阶段因并发显存 OOM 退出码 1，
但 e3 于 14:37 完成补评、14:48 完成选择；旧失败日志不覆盖当前结果。

也可用 fail-closed 串行脚本复现 `e1 train/predict/evaluate → e2 → e3`。该入口按用户批准将九个阶段全部
固定到物理 CUDA 1；训练为单卡 BS2/GA8/global batch 16，推理和评测也使用 CUDA 1。脚本中的 GPU 空闲等待
已注释，不以现有 compute process 阻断；每步仍先运行 launcher dry-run、完整阶段自动跳过，并把输出追加到
同目录 nohup 日志。该历史入口不执行最终 alias 人工选择；当前 alias 已由独立 selection record 冻结。

```bash
cd /data/jiahao/task/LlamaFactory
nohup bash tmp_bash/run_pt_exp2_mm_e1_e2_e3_v2.sh >/dev/null 2>&1 &
tail -f tmp_bash/pt_exp2_mm_e1_e2_e3_v2.nohup.log
```

用户已冻结官方 first-round + cross-pool exact-dedup 的 7,698,261 条为本机规范，8,092,423 仅保留为发布
provenance。`finalize-existing` 已重扫全部 38,485,631 个源行并逐行比对 31 shard，ordered corpus
SHA-256=`985b8473...07d0ab6`；未重写 34 GiB 数据。官方式 seed-0 first-round VAL1000 已生成。text PT 配置为
non-packing、`cutoff_len=6401`、250k steps、global batch 32。

全量 7,698,261 条 parse 已通过。确定性 10k collision replay 的 94/10,000 命中仍完整记录，前 20 个明细中
18 个来自 PT 首轮、2 个来自 SFT 首轮，含 4 个完整 component。论文与官方 sampler 说明发布 path 在采样阶段
做 collision filtering，官方 `train.py` 在训练加载阶段不再次 parse/collision 筛除。依用户决策，当前
`collision_findings_block_training=false`、`audit.eligible=true`，不删除样本；31-shard `text8m_train` 视图已创建。
当前 v2 三份 JSONL 已通过 Arrow 完整物化、逐行语义投影、真实 LlamaFactory `get_dataset`、Qwen
processor 全池零截断及 launcher 绑定检查。三轮训练、推理和评测均已完成，alias 选择 e1；launcher 报告
CUDA 1 上的进程但不以占用作为 blocker。原 e3 PE OOM 只作历史故障 provenance，后续补评 artifact 为当前状态。

### PT-exp2-mm-rowbal-cont3（连续三 epoch，ep1/ep2/ep3 prediction/evaluation 完成，推荐 checkpoint-33764）

状态：`training complete / valid; ep1/ep2/ep3 prediction/evaluation complete/valid; recommended=checkpoint-33764 (point-estimate rule; no alias)`。
固定数据集、配置和
launcher 已建立，标准 tokenized cache 已于 `2026-08-26 16:04 +08:00` 完成并通过校验；实际 Arrow train split
为 `270102` 行，列含 `input_ids/attention_mask/labels/images/videos/audios/length`，`length mismatch=0`。
新 launcher 的 `prepare-cache` dryrun 为 `ready=true, blockers=[]`；`train --gpus 1` dryrun 为
`executed=false`，内部 gate 全部通过；在 handoff 启动前唯一 blocker 是当时的 CUDA1 计算进程 PID=`2773643`。
该历史 dryrun 时训练输出目录不存在。
该实验从冻结 text8m 250k adapter 直接开始，在同一 Trainer 进程中以相同的 `270102` 行数据连续训练 3 个
epoch（每 epoch `135051` MM + `135051` text），并按 epoch 保存完整 checkpoint。正式训练于
`2026-08-27 07:28:30 +08:00` 开始并已有效完成；ep1/ep2/ep3 prediction/evaluation 均已完成并验证，最终按
`strict → dense → clean → parsable` 的点估计固定规则推荐 `checkpoint-33764`。这是事实排序，不宣称显著性；不创建
alias 或下游绑定。

2026-08-27 07:27:45 +08:00 上游 Text250k wrapper 已写出完成证据和 `exit status=0`，PID=`2773476` 已退出并清理
pidfile。handoff PID=`2909872` 于 07:28:01 严格验证 `exp4_4_2`/`exp4_7_2` evaluate dryrun gates（`ready=true`、
`already_complete=true`、`checks.output_complete=true`、`blockers=[]`）后记录 `stage_start rowbal_train`；07:28:30
实际启动 rowbal 训练。启动参数为 `270102` examples、3 epochs、`16882` update steps/epoch、总 `50646` steps、
device BS2、GA8、global16、CUDA1。07:28:52 有一次 `CUDACachingAllocator allocation failed` warning，训练继续且
至少推进到 step 90；该 warning 非 fatal，不表示实验失败。当时尚未填最终结果或 ETA，后续 checkpoint 状态见下。

2026-08-27 17:53:57 +08:00，epoch1 的 `checkpoint-16882` 已完整落盘并经复核有效：
`/data/jiahao/task/LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp2_mm_rowbal_cont3_qwen35_08b_text8m250k_mm135051_text135051_ep3_bs2_gbs16_lora64_len6400/checkpoint-16882`。
`adapter_model.safetensors`、`adapter_config.json`、`trainer_state.json`、`optimizer.pt`、`scheduler.pt`、
`rng_state.pth` 均齐全；`trainer_state` 为 `global_step=16882`、`epoch=1.0`、`max_steps=50646`。
保存后训练正常继续。2026-08-28 04:23:04 +08:00，epoch2 的 `checkpoint-33764` 已完整落盘并由主代理复核有效：
`/data/jiahao/task/LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp2_mm_rowbal_cont3_qwen35_08b_text8m250k_mm135051_text135051_ep3_bs2_gbs16_lora64_len6400/checkpoint-33764`。
该目录的 `adapter_model.safetensors`、`adapter_config.json`、`trainer_state.json`、`optimizer.pt`、
`scheduler.pt`、`rng_state.pth` 均齐全且稳定可读；`trainer_state` 为 `global_step=33764`、`epoch=2.0`、
`max_steps=50646`。保存后训练继续，04:30 已推进到 step=`33956`；当前仍为
`training/running`，最终 ETA 约为 2026-08-28 14:50 +08:00；该段为训练中的历史快照。最新完成状态见下方。

2026-08-28 14:55:45 +08:00，连续三 epoch 训练以 `stage_complete` 成功完成：
`trainer_state.global_step=50646`、`epoch=3`、`max_steps=50646`，`train_loss=0.2540496625`、
`runtime=113205.8521s`、`train_steps_per_second=0.447`。`checkpoint-50646` 和根输出目录的 adapter
均已完整落盘并通过训练完成校验；这部分状态为 `valid`，不是失败。

原 handoff 随即尝试进入 ep1，但因可用磁盘空间低于 `40 GiB` 的安全门退出（exit `1`），没有启动 prediction；
这是当时的历史状态。为解除该磁盘门，主代理删除了一个明确可重建且没有活动引用的旧
cache：`/data/jiahao/task/LlamaFactory/.llamafactory_cache/tokenized_dataset/PT-exp2-text7698261-qwen35-08b-len6401-nopack`
（`64,769,798,144` bytes）；正式使用的独立 `...nopack-with-length` cache 未删除，原始数据仍保留，该 cache 可按既有
prepare-cache 命令重建。可用空间约由 `30.5 GiB` 恢复至 `90.7 GiB`。

删除后 ep1 dry-run 已返回 `ready=true`；随后两次恢复被物理 CUDA1 上 mingyang 的短/长任务安全门阻止，
这些均为当时的历史状态，不能终止或干扰他人进程。

#### ep1 prediction/evaluation 已完成（2026-08-28 23:58:23 +08:00）

- 使用已验证的 `checkpoint-16882`；prediction 为 `512/512`，runtime=`11238.3337s`，BLEU-4=`79.8114`、
  ROUGE-1=`91.6656`、ROUGE-2=`62.759`、ROUGE-L=`53.2726`；`generated_predictions.jsonl` SHA-256=
  `ae573f7970b7cee7888ab600a232a2dee3f4b853b7caf64d091f5c51b69b2172`。
- base evaluation fully parsable=`404/512`、collision-free=`105/512`、mean actions before first failure=
  `8.421875`（约 `8.4`）、complete render=`404/404`、render failure=`0`。
- alignment `samples=512`：parse prefix=`0.9102900774232086`、inventory F1=`0.8155414811184989`、length=
  `0.809655785706696`、collision prefix=`0.5570950990310708`、pose=`0.12373974525207188`、dense reward=
  `0.5746728336608469`、strict=`2/512=0.00390625`；PE=`0.27744618028697399`、SigLIP2=
  `0.76755156375394007`、VQA=`0.74573100058564745`（图像指标样本均为 `404`）。
- ep1 evaluator exit=`0` at `2026-08-28 23:58:23 +08:00`；`metrics.json` SHA-256=
  `1a5814a6820e70eaf0ccf667e316765cc4626d88195d5dc3d20c614fd6508bfc`，`alignment_manifest.json` SHA-256=
  `8243d937d775a44b9f6b73a11c12401b60c6a64eb57891eba2eb1caba288da27`，status=`complete`。
- 截至该历史条目，ep2 prediction 于 `2026-08-28 23:59:44 +08:00` 在物理 CUDA1 启动并仍在运行，尚未
  complete/valid；下方 ep2 完成条目已覆盖该快照，不据此推断当前状态。

#### ep2 prediction/evaluation 已完成（2026-08-29 02:51:07 +08:00）

- 使用已验证的 `checkpoint-33764`；prediction 为 `512/512`、exit=`0`，完成于 `2026-08-29 02:40:54 +08:00`，runtime=`9611.6262s`；
  BLEU-4=`83.8182900390625`、ROUGE-1=`93.14293828125`、ROUGE-2=`64.2137451171875`、ROUGE-L=`53.681630664062496`；
  `generated_predictions.jsonl` SHA-256=`c156da9d485a2f398478c82742eace44b2adb4d499c7228c09c580dc2f46f8b1`。
- evaluation exit=`0` at `2026-08-29 02:51:07 +08:00`；fully parsable=`402/512`、clean/collision-free=`113/512`、collision/mean actions before
  failure=`8.033203125`、render=`402/402`、render failure=`0`。
- alignment `samples=512`：parse prefix=`0.9183373919040974`、inventory F1=`0.8566068334354298`、length=
  `0.8518979466795903`、collision prefix=`0.5456691592547537`、pose=`0.13199846714201502`、dense reward=
  `0.5889120117294198`、strict=`8/512=0.015625`、exact path=`3`；PE=`0.27715973355876866`、SigLIP2=
  `0.7880970399771163`、VQA=`0.7456277761960504`（图像指标样本均为 `402`）。
- `metrics.json` SHA-256=`950a8356ca0285cb53494f9b4a8846b90e92c73b7f12dfe4bcf187f7e0c4a17d`；
  `alignment_manifest.json` SHA-256=`ec1f3aa21cad2e24523396d0b6c724890325ee9856cf74f35e4070882c1be2d5`，status=`complete`。
  完成后的 evaluate dry-run 仅报告 `EVALUATION_ALREADY_COMPLETE`，这是已完成 artifact 的预期 blocker，不是失败。
- 相对 ep1，ep2 的 BLEU/ROUGE、clean、parse prefix、inventory F1、length、pose、dense、strict、SigLIP2 均为更高的点估计
  （ep2 strict `8/512` 对 ep1 `2/512`），但 fully parsable 为 `402/512` 对 `404/512`，collision prefix、mean actions、PE 和
  VQA 不更高；这是事实比较，不是最终推荐或显著性结论。
- ep3 prediction/evaluation 已有效完成；最终按固定规则推荐 `checkpoint-33764`，不创建或绑定 alias。

#### ep3 prediction/evaluation 已完成（2026-08-29 05:47:32 +08:00）

- 使用已验证的 `checkpoint-50646`；prediction 为 `512/512`，runtime=`9785.7039s`，BLEU-4=`84.08433886718751`、
  ROUGE-1=`93.0461208984375`、ROUGE-2=`64.09938632812501`、ROUGE-L=`53.8440525390625`；
  `generated_predictions.jsonl` SHA-256=`031a7ad92cd0bee16e685e0a02270e8e4b63f3189e50abe416f9cd783991324a`；
  `predict_results.json` SHA-256=`adaa00298f46a2a3bc7555d6facf2cfc09e9bbcd0c750fbba46fa1d640721d4d`。
- evaluation 于 `2026-08-29 05:47:32 +08:00` 以 exit=`0` 完成且有效；fully parsable=`405/512`、clean/collision-free=
  `105/512`、skipped=`107`、render=`405/405`、render failure=`0`。
- alignment `samples=512`：parse=`0.9130015178424298`、inventory F1=`0.8555066953271355`、length=`0.8465148582783681`、
  collision prefix=`0.5459210124616339`、pose=`0.1257369235575623`、dense=`0.5852584080213454`、
  strict=`5/512=0.009765625`、exact path=`0`；PE=`0.27697331934799385`、SigLIP2=`0.77552385447937766`、
  VQA=`0.73738091385658877`（图像指标样本均为 `405`）。
- `metrics.json` SHA-256=`0da1cde9c94709dbee596ddfd93a0dd0349535b01c22668d8be2d43c3328ec6e`；
  `alignment.jsonl` SHA-256=`815fcd28e52a637b88947890ccc4ceaafbb9926b3649d1fb640aff448bba4805`；
  `alignment_manifest.json` SHA-256=`414e2340bcbc67c740ca28d99a5d5994c4d7fcaadc4e2e6cbf86bdc91ec67d58`，status=`complete`。
- ep1/ep2/ep3 的固定选择 tuple（strict→dense→clean→parsable）为
  `ep1=(2/512, 0.5746728336608469, 105, 404)`、
  `ep2=(8/512, 0.5889120117294198, 113, 402)`、
  `ep3=(5/512, 0.5852584080213454, 105, 405)`；因此推荐 `checkpoint-33764`。这是点估计排序，不宣称显著性，且不创建 alias。

```bash
cd /data/jiahao/task/LlamaFactory

# 三 epoch 训练及 ep1/ep2/ep3 prediction/evaluation 已完成；推荐 checkpoint-33764。不要再次执行 train/predict/evaluate，
# 不创建 alias；以下命令仅保留为已执行顺序记录。

# 首次生成并发布完整 CPU loader/processor 审计；已有完整报告时不要重复执行。
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3.py \
  --action prepare-audits --processor-workers 4 --processor-chunksize 8 --execute

# 生成带 length 列的标准 tokenized cache；此步骤仅 CPU，不启动 Trainer。
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3.py \
  --action prepare-cache --cache-num-proc 1 --cache-batch-size 10000 --execute

# 只读训练 preflight；确认 cache、parent adapter、输出目录和 CUDA1 gate。
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3.py \
  --action train --gpus 1

# 正式连续三 epoch 训练；保存 checkpoint-16882/33764/50646 的完整状态。
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3.py \
  --action train --gpus 1 --execute

# ep1 已完成 prediction/evaluation；以下 ep1 命令仅保留为已执行的顺序记录。
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3.py \
  --action predict --run ep1 --gpus 1 --execute
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3.py \
  --action evaluate --run ep1 --gpus 1 --execute

# ep2 prediction/evaluation 已完成；以下 ep2 命令仅保留为已执行顺序记录，不要重复启动。
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3.py \
  --action predict --run ep2 --gpus 1 --execute
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3.py \
  --action evaluate --run ep2 --gpus 1 --execute

# ep3 prediction/evaluation 已完成；以下命令仅保留为已执行顺序记录，不要重复启动。
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3.py \
  --action predict --run ep3 --gpus 1 --execute
# ep3 evaluation 已于 2026-08-29 05:47:32 +08:00 以 exit=0 完成并通过完整性验证。
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3.py \
  --action evaluate --run ep3 --gpus 1 --execute
```

### PT-exp2-mm-rowbal-cont3 自动 handoff（训练已完成，ep1/ep2/ep3 完成，推荐 checkpoint-33764；禁止重复启动）

2026-08-26 16:32 +08:00 首次以 nohup 启动自动 handoff 监督器
`tmp_bash/run_pt_exp2_mm_rowbal_cont3_after_text250k_cuda1.sh`，PID=`2905400`。该进程在 16:38 前异常消失，
没有写出 EXIT 日志；锁已释放，未触碰 CUDA1，也没有启动 rowbal。其旧 PID 文件已移至
`tmp_bash/run_pt_exp2_mm_rowbal_cont3_after_text250k_cuda1.pid.stale-2905400-20260826T163848`。

2026-08-26 16:39 +08:00 曾通过受工具会话托管的方式恢复 handoff：当时 PID=`2909872`，统一 exec session=`70789`，
锁处于 held 状态；监督器绑定上游 Text250k wrapper PID=`2773476`，读取其日志从字节偏移 `453661` 开始的本轮
完成证据。当时仅等待 `exp4_4_2/exp4_7_2` 完整 train→predict→evaluate，不占用 CUDA1，rowbal
`training_started=false`。只有上游 exit status=`0`、CUDA1 为空闲且两组 evaluate dry-run 均严格返回
`ready=true/already_complete=true/output_complete=true` 时，监督器才会按顺序执行 rowbal train 以及
ep1/ep2/ep3 prediction/evaluate；任一 gate 失败都会 fail-closed，不会静默训练。

该 handoff 后续已完成上游 gate、启动并完成 rowbal 训练，随后在 ep1 磁盘安全门处退出；禁止再次启动，尤其不要重新使用
nohup 重启，以免产生重复 handoff。本文不再提供 handoff 启动命令；如需恢复预测，应先核对日志、锁、外部 CUDA1
占用和 fail-closed 证据，再由主代理决定后续操作。

当前 handoff 已完成上游 completion gate 并启动、完成 rowbal 训练；原 handoff 在 ep1 的磁盘安全门处退出。
ep1 prediction/evaluation 已于 `2026-08-28 23:58:23 +08:00` 完成，ep2 prediction/evaluation 已于
`2026-08-29 02:51:07 +08:00` 有效完成，ep3 prediction/evaluation 已于 `2026-08-29 05:47:32 +08:00` 有效完成。
按固定点估计规则推荐 `checkpoint-33764`；禁止重复启动 handoff，不创建 alias。

只读查看 handoff 日志、进程和物理 CUDA1 占用：

```bash
cd /data/jiahao/task/LlamaFactory
tail -f tmp_bash/run_pt_exp2_mm_rowbal_cont3_after_text250k_cuda1.log
ps -o pid,ppid,stat,etime,%cpu,%mem,cmd -p 2909872,2773476
nvidia-smi -i 1 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
```

### exp4_4 / exp4_7（PT-exp2-v2，物理 CUDA1 单卡）

2026-08-25 已新增隔离的 `PT-exp2-v2` 下游入口。`exp4_4` 使用 NonThinking-Control 10k；`exp4_7`
只把监督数据替换为 Stage2 V2 Thinking-Hard Lean-State 10k。两组都从同一个 `PT-exp2-v2` e1 alias
独立创建新 LoRA，不互相串接，也不修改旧 `PT-exp2`/`exp4_4` 模板。训练协议均为 BS1、GA16、global
batch 16、3 epochs、LR `5e-5`、LoRA 64/128、`cutoff_len=16384`，并固定物理 CUDA1。

当前两份训练 processor audit（10,000+10,000）和两份 VAL audit（512+512）均为 0 error、0 truncation，
Control/Lean-State 的 ordered IDs 分别一致。以下 audit 命令只在报告缺失或数据/processor 漂移时重建；
现有报告上重建必须显式增加 `--overwrite`。

```bash
cd /data/jiahao/task/LlamaFactory

CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/jiahao/miniconda3/envs/llamafactory/bin/python \
  scripts/audit_bricknet_reasoning_tokens.py \
  --stage 2 --audit-purpose pt_exp2_v2_downstream_train_pair \
  --dataset NonThinking-Control=/data/jiahao/task/LlamaFactory/data/bricknet_stage2/10k/BrickNet-Stage2-NonThinking-Control.jsonl \
  --dataset Thinking-Hard-V2-Lean-State=/data/jiahao/task/LlamaFactory/data/bricknet_stage2_v2/10k/BrickNet-Stage2-ThinkingHard-V2-LeanState.jsonl \
  --bricknet-root /data/jiahao/task/BrickNet \
  --output-dir /data/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Reasoning/pt_exp2_v2_downstream/reports/token_audit/train10k \
  --cutoff-len 16384 --workers 4 --chunksize 16

CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/jiahao/miniconda3/envs/llamafactory/bin/python \
  scripts/audit_bricknet_reasoning_tokens.py \
  --stage 2 --audit-purpose pt_exp2_v2_downstream_eval_pair \
  --dataset NonThinking-Control-VAL512=/data/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Reasoning/validation/datasets/BrickNet-Stage2-NonThinking-Control-VAL512-Eval.jsonl \
  --dataset Thinking-Hard-V2-Lean-State-VAL512=/data/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Reasoning/stage2_v2/validation/datasets/BrickNet-Stage2-ThinkingHard-V2-LeanState-VAL512-Eval.jsonl \
  --bricknet-root /data/jiahao/task/BrickNet \
  --output-dir /data/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Reasoning/pt_exp2_v2_downstream/reports/token_audit/eval_val512 \
  --cutoff-len 16384 --workers 4 --chunksize 16
```

#### 2026-08-25 exp4_4 alignment manifest 修复与安全回填

原 batch 已完成 `exp4_4` 的 1,875/1,875-step 训练、512/512 推理和数值评测，但当时的通用 Stage2
evaluator 没有生成新下游 launcher 要求的 `alignment_manifest.json`，因此在进入 `exp4_7` 前 fail-closed
停止。修复后的 evaluator/launcher 使用同一完整 manifest 契约。下面命令先只读识别“数值完整、manifest 缺失”，
再对标准 VAL512 reference、alignment/scored 逐行关系、输入输出 hash 和聚合指标做严格验证；验证通过时只原子
回填 manifest，不调用评测子进程，也不重跑 train、predict、render 或 numeric evaluation。当前再次执行时会因
完整 gate 已通过而安全 no-op。

```bash
cd /data/jiahao/task/LlamaFactory

# 只读检查：回填前实际为 numeric_evaluation_complete=true、alignment_manifest_complete=false。
conda run -n llamafactory --no-capture-output \
  python scripts/evaluate_bricknet_stage2.py --experiment exp4_4

# 已执行的安全回填；当前重复执行会复用完整结果，不重算数值评测。
conda run -n llamafactory --no-capture-output \
  python scripts/evaluate_bricknet_stage2.py --experiment exp4_4 --execute

# 固定 manifest 身份，并重新检查 evaluator 与下游 launcher 的完整完成 gate。
EXP44_ALIGNMENT_MANIFEST=/data/jiahao/task/BrickNet/outputs_val/qwen35_08b/eval_exp4_4_PT_exp2_v2_nonthinking_control_10k_val512_in16384_out16384_p95_t1_k20/alignment_manifest.json
test -f "$EXP44_ALIGNMENT_MANIFEST"
sha256sum "$EXP44_ALIGNMENT_MANIFEST"
test "$(sha256sum "$EXP44_ALIGNMENT_MANIFEST" | awk '{print $1}')" = \
  c8b064186ddc3fc7c4cc8aeba88712c0f5d9f10018722a8a9e7a2ab70f184ba8

conda run -n llamafactory --no-capture-output \
  python scripts/evaluate_bricknet_stage2.py --experiment exp4_4
/home/jiahao/miniconda3/envs/llamafactory/bin/python \
  scripts/launch_bricknet_pt_exp2_v2_downstream.py \
  --run exp4_4 --action evaluate --gpus 1
```

当前 manifest schema 为 `bricknet-stage2-alignment-v1`，
SHA-256=`c8b064186ddc3fc7c4cc8aeba88712c0f5d9f10018722a8a9e7a2ab70f184ba8`，
`freeze.mode=verified_numeric_backfill`。回填后 evaluator 为 `already_complete=true`，下游 launcher 为
`output_complete=true`。冻结指标为 train loss=`0.1477434707`、parsable=`407/512`、clean=`122/512`、
Dense Reward=`0.6062629319`、Strict Success=`11/512`。

#### 2026-08-26 exp4_7 全链完成与冻结

`exp4_7` 训练于 `2026-08-25 19:01:05~19:01:24 +08:00` 完成并通过 train gate。`trainer_state.global_step=max_steps=1875`，
train_results/all_results 一致，train loss=`0.10565751036008199`，train runtime=`10756.4909s`，
`num_input_tokens_seen=90109152`；derived throughput=`8377.19 token/s`、samples/s=`2.789`、steps/s=`0.174`。
adapter_config SHA-256=`30898fb95cb02bb51edbd28c48f38f203825e8956bc1226e111eabceefc51ef6`，
adapter_model.safetensors SHA-256=`c8ad92be7a6091fc5f2198fc6efe8391d6e32e321d79156e16df2a294d1d69f6`。

prediction 目录为
`saves/Qwen3.5-0.8B-Thinking/lora/eval_exp4_7_PT_exp2_v2_thinking_hard_v2_lean_state_10k_val512_in16384_out16384_p95_t1_k20`，
512 rows，runtime=`30150.9915s`、samples/s=`0.017`；`generated_predictions.jsonl` SHA-256=
`c6800800648b9e234632f579c8ddf6507a4c4bca2186f8c5bb7016c9cf0ae92e`，`path_predictions.jsonl` SHA-256=
`d0e9fc35e50d9d87644aa6f51bac3057b23a2087399a3baec7b1765f09881c19`，`trace_extraction_report.json` SHA-256=
`adc42d5826ddfdce543b24783f999b8b487c612373e3b2acf33c3cd731f1f41c`。

Stage-2 evaluation 目录为
`/data/jiahao/task/BrickNet/outputs_val/qwen35_08b/eval_exp4_7_PT_exp2_v2_thinking_hard_v2_lean_state_10k_val512_in16384_out16384_p95_t1_k20`；
`scored.jsonl`、`alignment_input.jsonl`、`alignment.jsonl` 均为 512 rows，renders/PE/SigLIP2/VQA 均为 409，fully
parsable 为 409 且无 render failure。核心指标：parsable=`409/512=0.798828125`、clean=`125/512=0.244140625`、
Dense Reward=`0.6046798092343368`、Strict=`13/512=0.025390625`、inventory F1=`0.8982252851027394`、
length=`0.8977633625091882`、pose=`0.1457130516560686`、exact path=`0`、BLEU4=`90.033721875`、
ROUGE1/2/L=`94.8887908203125/65.453983984375/55.00614921875`、PE/SigLIP2/VQA=
`0.2794995296263753/0.8000668227526845/0.7507266265438064`。

`metrics.json` SHA-256=`5d042bd093234860f926fd459e78243e60729c612bc6dbc18270ba245fec852d`；
`evaluation_manifest.json` SHA-256=`7bdd6168c74414ae64fbc817785e49cba3030c11e7acaf88ca88b86506ad07dd`；
`alignment_manifest.json` SHA-256=`d748e37596c1f523a817ede44cc59b499e42da5aafe31c16da529447c4585b7c`，schema=
`bricknet-stage2-alignment-v1`、status=`complete`、freeze=`post_evaluation_freeze`。两个只读 launcher 最终审计
均为 `ready=true`、`already_complete=true`、无 blockers；manifest identity/artifact hash 契约通过，相关回归
`10 passed`。Transformers docstring `[ERROR]` 仅为非致命 warning，不影响 wrapper exit 0。

`trace_extraction_report.json` 的 `trace_contract=lean-state-v1-internal-consistency`，`count=512`、
`trace_format_valid=34`、`trace_format_rate=0.06640625`、`nonempty_extracted_prefix=511`、
`reference_labels_valid=512`，trace errors 总计 478；主要为 action1 mismatch 261、action0 mismatch 84、
action2 mismatch 17、action3 mismatch 10。canonical path 的 512 条数值评测与 alignment manifest 契约有效，
但 Lean-State 内部一致性很差，不能声称可靠状态追踪；这是模型输出质量信号，不是 evaluator/launcher 失败。

同初始化 `exp4_4` 对照为 parsable=`407`、clean=`122`、dense=`0.6062629319122218`、strict=`11`、
inventory F1=`0.9049767466390838`、length=`0.884054956309081`、pose=`0.13778809824321342`；
`exp4_7-exp4_4` 的 parsable/clean/strict/length/pose 点估计上升而 dense/inventory 下降。尚未运行 paired
bootstrap，结论只能写成混合点估计，不能宣称显著优胜或选出赢家；`exp4_5/exp4_6` 继续 dormant。

串行 wrapper `/data/jiahao/task/LlamaFactory/tmp_bash/run_exp4_4_exp4_7_pt_exp2_v2_cuda1.sh` 及同名 `.log`
于 `2026-08-26 03:34:07 +08:00` 记录 `complete`、exit status=`0`；PID=`2099550` 已退出，物理 CUDA1 空闲。

修复的定向测试和静态检查可直接执行如下；最终完整套件实测 pytest 为 `10 passed`，两个 Python 文件的
`py_compile` 与相关 diff check 均通过：

```bash
cd /data/jiahao/task/LlamaFactory

/home/jiahao/miniconda3/envs/openpi/bin/python -m pytest -q \
  --confcutdir=tests/eval \
  tests/eval/test_bricknet_stage2_alignment_manifest.py

/home/jiahao/miniconda3/envs/llamafactory/bin/python -m py_compile \
  scripts/evaluate_bricknet_stage2.py \
  scripts/launch_bricknet_pt_exp2_v2_downstream.py

git diff --check -- \
  scripts/evaluate_bricknet_stage2.py \
  scripts/launch_bricknet_pt_exp2_v2_downstream.py \
  tests/eval/test_bricknet_stage2_alignment_manifest.py
```

#### 六阶段恢复入口

推荐入口先执行全链静态 preflight，再按 `exp4_4 train→predict→evaluate→exp4_7 train→predict→evaluate`
自动顺序运行。preflight 和正式入口都会 fail-closed 检查物理 GPU1；它们不会自行终止占用进程。
初次六阶段 preflight 于 `2026-08-25T01:12:52+08:00` 通过；原 batch PID=`1696595` 在完成 exp4_4
数值 artifact 后因上述 manifest 缺口退出。契约修复和回填后，全链 preflight 再次通过；串行 launcher 于
`2026-08-25T15:54:51+08:00` 恢复并于 `2026-08-26 03:34:07 +08:00` 完成，exit status=`0`，PID=`2099550`
已退出、CUDA1 空闲。恢复入口对三个完整 `exp4_4` stage 均安全 no-op，随后完成 `exp4_7` 的 train/predict/evaluate；
两个只读 launcher 最终 `ready=true/already_complete=true`、无 blockers，回归 `10 passed`。

```bash
cd /data/jiahao/task/LlamaFactory

bash tmp_bash/run_exp4_4_exp4_7_pt_exp2_v2_cuda1.sh --preflight-only

# 可靠独立会话启动。当前 batch 存活时拒绝重启，避免改写其 PID 文件。
EXP47_BATCH_PID_FILE=tmp_bash/run_exp4_4_exp4_7_pt_exp2_v2_cuda1.pid
if test -s "$EXP47_BATCH_PID_FILE" && \
   kill -0 "$(cat "$EXP47_BATCH_PID_FILE")" 2>/dev/null; then
  printf 'batch already running: PID=%s\n' "$(cat "$EXP47_BATCH_PID_FILE")"
else
  setsid -f bash -c '
    printf "%s\n" "$$" > tmp_bash/run_exp4_4_exp4_7_pt_exp2_v2_cuda1.pid
    exec bash tmp_bash/run_exp4_4_exp4_7_pt_exp2_v2_cuda1.sh
  ' </dev/null >/dev/null 2>&1
fi

# 已完成 batch：exp4_4 三阶段均安全 no-op；exp4_7 已完成 train/predict/evaluate，wrapper exit status=0，
# PID=2099550 已退出，CUDA1 空闲。
BATCH_PID=2099550
cat tmp_bash/run_exp4_4_exp4_7_pt_exp2_v2_cuda1.pid
ps -o pid,ppid,pgid,sid,stat,etime,cmd -p "$BATCH_PID"
test "$(cat tmp_bash/run_exp4_4_exp4_7_pt_exp2_v2_cuda1.pid)" = "$BATCH_PID"
tail -F tmp_bash/run_exp4_4_exp4_7_pt_exp2_v2_cuda1.log
```

需要逐阶段手工执行时，严格使用下面六条；每一条成功后再执行下一条：

```bash
cd /data/jiahao/task/LlamaFactory

/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_v2_downstream.py \
  --run exp4_4 --action train --gpus 1 --pt-exp2-v2-approved --execute
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_v2_downstream.py \
  --run exp4_4 --action predict --gpus 1 --execute
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_v2_downstream.py \
  --run exp4_4 --action evaluate --gpus 1 --execute

/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_v2_downstream.py \
  --run exp4_7 --action train --gpus 1 --pt-exp2-v2-approved --execute
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_v2_downstream.py \
  --run exp4_7 --action predict --gpus 1 --execute
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_v2_downstream.py \
  --run exp4_7 --action evaluate --gpus 1 --execute
```

旧 `scripts/launch_bricknet_pt_exp2.py --run exp4_4` 仍绑定历史 `PT-exp2` alias，不能用于本轮。
`exp4_5/exp4_6` 继续 dormant；只有本轮 exp4_4 完整结果经独立收益 gate 批准后，才讨论 50k/all。

### exp4_5
```bash
# 正式训练 exp4_5 50k。
python scripts/launch_bricknet_pt_exp2.py --gpus 0 1 --action train --run exp4_5 --execute
# 使用 exp4_5 adapter 运行 VAL512 推理。
python scripts/launch_bricknet_pt_exp2.py --gpus 0 --action predict --run exp4_5 --execute
# 统一评测 exp4_5 的 VAL512 结果。
python scripts/launch_bricknet_pt_exp2.py --gpus 0 --action evaluate --run exp4_5 --execute
# 人工确认 exp4_5 收益并批准扩展到 all。
python scripts/launch_bricknet_pt_exp2.py --action approve-scale --run exp4_5 --execute --approve
```

### exp4_6
```bash
# 正式训练 exp4_6 全量 66,456。
python scripts/launch_bricknet_pt_exp2.py --gpus 0 1 --action train --run exp4_6 --execute
# 使用 exp4_6 adapter 运行 VAL512 推理。
python scripts/launch_bricknet_pt_exp2.py --gpus 0 --action predict --run exp4_6 --execute
# 统一评测 exp4_6 的 VAL512 结果。
python scripts/launch_bricknet_pt_exp2.py --gpus 0 --action evaluate --run exp4_6 --execute
```

### PT-exp2-100k 权重

用户在另一台服务器用 2× RTX PRO 6000 训练的 100,000-step PT checkpoint 已同步。当前四份
exp4_4_1 / exp4_7_1 YAML 实际固定读取：

`saves/Qwen3.5-0.8B-Thinking/lora/backup/train_PT_exp2_text8m_qwen35_08b_path7698261_steps250k_bs4_gbs32_lora64_len6401_nopack-checkpoint-100000`。

该目录的 `trainer_state.global_step=100000`；`adapter_model.safetensors` SHA-256 为
`f211e68913ae77d8320ef6d4d9108085741b7dac78fb3191cbcda10ae0f2c6c9`。复现或重跑前执行：

```bash
cd /home/jiahao/task/LlamaFactory
PT100K=saves/Qwen3.5-0.8B-Thinking/lora/backup/train_PT_exp2_text8m_qwen35_08b_path7698261_steps250k_bs4_gbs32_lora64_len6401_nopack-checkpoint-100000
jq '.global_step' "$PT100K/trainer_state.json"
sha256sum "$PT100K/adapter_model.safetensors"
ls "$PT100K/adapter_config.json" "$PT100K/adapter_model.safetensors"
```

该 100k 权重与 `PT-exp2-text8m(250k) → mm-e1/e2/e3 → PT-exp2` 主链相互独立；exp4_4_1 / exp4_7_1
直接从上述 checkpoint-100000 初始化，不等待 text8m/MM-e1/e2/e3 或 final alias。两组训练、
VAL512 推理和统一评测均已完成；结果与 hash 见 `experiment_results.md`。

#### exp4_4_1 = PT-exp2-100k + NonThinking-Control 10k

配置：train=`examples/train_lora/qwen35_08b_bricknet_stage2_exp4_4_1_nonthinking_control_10k_pt_exp2_100k.yaml`，
predict=`examples/train_lora/qwen35_08b_bricknet_stage2_exp4_4_1_nonthinking_control_predict_pt_exp2_100k.yaml`。
与 exp4_4 相同的 LoRA/LR/epoch/batch/长度协议，仅初始化换成 `PT-exp2-100k`。
当前状态：训练 1,875/1,875 steps、推理 512/512 和统一评测完成。

```bash
cd /home/jiahao/task/LlamaFactory

# 单卡训练（GA 在 CLI 覆盖为 16，global batch 16）。
# conda run -n llamafactory --no-capture-output
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
  examples/train_lora/qwen35_08b_bricknet_stage2_exp4_4_1_nonthinking_control_10k_pt_exp2_100k.yaml \
  gradient_accumulation_steps=16

# 双卡训练（与 launcher 相同的 DDP 注入）。
FORCE_TORCHRUN=1 NPROC_PER_NODE=2 NNODES=1 CUDA_VISIBLE_DEVICES=0,1 \
  conda run -n llamafactory --no-capture-output llamafactory-cli train \
  examples/train_lora/qwen35_08b_bricknet_stage2_exp4_4_1_nonthinking_control_10k_pt_exp2_100k.yaml \
  gradient_accumulation_steps=8

# VAL512 推理（单卡）。
conda run -n llamafactory --no-capture-output llamafactory-cli train \
  examples/train_lora/qwen35_08b_bricknet_stage2_exp4_4_1_nonthinking_control_predict_pt_exp2_100k.yaml

# 统一评测（提取 path + BLEU/ROUGE + 渲染/图文指标 + alignment）。
conda run -n llamafactory --no-capture-output python \
  scripts/evaluate_bricknet_stage2.py --experiment exp4_4_1 --execute
```

#### exp4_7_1 = PT-exp2-100k + Stage-2 V2 Thinking-Hard Lean-State 10k

配置：train=`examples/train_lora/qwen35_08b_bricknet_stage2_exp4_7_1_thinking_hard_10k_pt_exp2_100k.yaml`，
predict=`examples/train_lora/qwen35_08b_bricknet_stage2_exp4_7_1_thinking_hard_predict_pt_exp2_100k.yaml`。
与 exp4_3_1 相同的 Stage-2 V2 Lean-State 数据（`BrickNet-Stage2-ThinkingHard-V2-LeanState-10k` 与
`BrickNet-Stage2-ThinkingHard-V2-LeanState-VAL512-Eval`）与训练协议，仅初始化换成 `PT-exp2-100k`；
不使用效果不佳的旧 Thinking-Hard 数据。评测自动走 strict trace 提取（variant=thinking-hard-v2-lean-state）。
当前状态：训练 1,875/1,875 steps、推理 512/512、strict extraction 和统一评测完成。

```bash
cd /home/jiahao/task/LlamaFactory

# 单卡训练。
#  conda run -n llamafactory --no-capture-output
CUDA_VISIBLE_DEVICES=1 llamafactory-cli train \
  examples/train_lora/qwen35_08b_bricknet_stage2_exp4_7_1_thinking_hard_10k_pt_exp2_100k.yaml \
  gradient_accumulation_steps=16

# 双卡训练。
FORCE_TORCHRUN=1 NPROC_PER_NODE=2 NNODES=1 CUDA_VISIBLE_DEVICES=0,1 \
  conda run -n llamafactory --no-capture-output llamafactory-cli train \
  examples/train_lora/qwen35_08b_bricknet_stage2_exp4_7_1_thinking_hard_10k_pt_exp2_100k.yaml \
  gradient_accumulation_steps=8

# VAL512 推理（单卡）。
conda run -n llamafactory --no-capture-output llamafactory-cli train \
  examples/train_lora/qwen35_08b_bricknet_stage2_exp4_7_1_thinking_hard_predict_pt_exp2_100k.yaml

# 统一评测（strict trace 提取 + 全指标 + alignment）。
conda run -n llamafactory --no-capture-output python \
  scripts/evaluate_bricknet_stage2.py --experiment exp4_7_1 --execute
```

## exp4_2 直通 Stage5–8（action-only）

固定链：`Qwen/Qwen3.5-0.8B + PT-exp1 + exp4_2`。以下命令已预填当前数据、adapter、VAL512 和报告路径，
仍可直接执行或用于同协议复跑。Stage5 full replay 已通过；当前有效状态、冻结结果和指标解释见
`/home/jiahao/task/BrickNet/BrickNet-MM Agentic LEGO Planner/Stage 6-7 Agentic Evaluation.md`。

> 2026-08-20 新服务器 canonical-path refreeze（历史 v1 身份）：config SHA-256=
> `f4eb8c9b14e9f5919d45925d84d385e42c60595e57da179b77d63d63f08c28b2`，canonical contract SHA-256=
> `76067f38eecd3ec02b4b5e3cc9bf2a44e5b704beeb0a206c668b9eeae5d1f825`。
> 现有 Stage5 full replay report 已在新服务器校验通过，无需重新运行。
> 下方 Stage5 full replay 命令仅用于报告缺失、输入/mesh hash 漂移或环境实现变化时的重建。

> **2026-08-21 A1-only 协议修订（覆盖下方 2026-08-18 的“五组待重跑”历史说明）：**
> B1/V1/V2/A0 已各完成 512/512，绑定 canonical contract `76067f38…f825`，四组 controller
> preflight 与已有八套 final/diagnostic 评测均通过。它们按文件 hash 采用，禁止重跑或改写 provenance。
> 当前 per-run config SHA-256=`371a60a1e55294a3a7d7b0917c4d48954009c2fe1337d3f5ae7df4b45c35a987`，
> compact-A1 active contract SHA-256=`f0effcc8fcc4fea35fde5b2aa82e6a6ecb9418d038a4ecb54d00f6d5ab38f611`；
> 新服务器逐字节模板为 `configs/agentic_stage67_exp4_2.new-server.371a60a1.json`。
> A1-v1 两次都在 row index 2 的 prefill 因完整失败子树历史膨胀而 CUDA OOM，2-row partial 不是 artifact。
> 当前同名 A1 改为“父节点每个失败 direct child 一条摘要；完整后代轨迹只进 audit；processor 实际
> `input_ids`（含图像展开）最多 16,384 tokens，超限在搬入 CUDA 前让当前样本确定性失败并继续”。后续只运行 A1，
> 不运行 V2+A1，也不串行重跑五组。A1 完成后只重建 A1 两层，再用已有八层生成五模式×两层 manifest/bootstrap。

> **历史记录（已被上方 2026-08-21 决策取代）：2026-08-18 inference-contract v1 冻结后曾要求五组重跑。** 旧 V1/V2 的 raw first-choice
> 会在首选 proposal 被 verifier 拒绝时错误写为空串，且旧 V2/A1 在首批
> 合法候选全部走入死路后不会继续该 state 的后续轮次，因而不是完整 bounded DFS；旧 A0/A1 还使用
> `stage8-act` 替换了 exp4_2 system/user prompt，不满足“首轮与 V1 相同”的冻结定义。修复后的 Stage6–7 统一使用
> `exp4_2-stepwise`：V1 静默重试，V2=V1+多候选完整 bounded DFS，A0 首轮与 V1 字节相同且之后显示
> proposal/observation、只重试 rejected action，A1=A0+多候选 DFS、允许回退 accepted prefix。每个进程仍只在
> 启动时执行一次 `set_seed(42)`。这段“五组覆盖”授权已经失效，不能据此执行旧命令；当前唯一覆盖授权是上方所述 A1-only。
> 旧异常数值仍可从本文与统一结果账本的历史记录追溯。
>
> **输出已改为逐样本流式持久化。** 每条完成后立即写 audit body、final 和 diagnostic 三个 `*.partial.jsonl`，正常结束
> 时补齐全局 hash/provenance，并按 final→diagnostic→audit 的顺序提升正式文件；不提供 resume。若进程中断，partial
> 保留且 Eval 会 fail closed；重新执行同一条 controller 命令会清空该组旧 partial 并从第 0 条重新开始。

### 双服务器配置激活与正式启动

不同服务器路径身份对应不同 canonical config。当前 A1-only refreeze 只在新服务器完成；正式入口始终是
`configs/agentic_stage67_exp4_2.json`，`tmp_bash` 不再保存配置副本。当前新服务器可执行模板是 `371a60a1`；旧
`f4eb8c9b/606dcd5f` 模板必须保留：前者是 B1/V1/V2/A0 adoption 的不可变 v1 证据，后者是旧服务器迁移历史，
均不能拿来启动 compact A1。

#### 当前新服务器（canonical root `/data/jiahao/task`）

```bash
# 激活新服务器配置、固定 SHA，并只读运行官方 preflight；不写 tmp_bash 配置副本。
(
  set -Eeuo pipefail
  if pgrep -f \
    '[r]un_stage67_base_exp4_2.sh|[r]un_bricknet_agentic(_a1_compact)?_inference.py|[e]valuate_bricknet_agentic_stage67.py' \
    >/dev/null; then
    printf 'Stage6–7 is already running; do not switch its config.\n' >&2
    exit 1
  fi

  cd /data/jiahao/task/BrickNet
  install -m 0644 \
    configs/agentic_stage67_exp4_2.new-server.371a60a1.json \
    configs/agentic_stage67_exp4_2.json
  printf '%s  %s\n' \
    '371a60a1e55294a3a7d7b0917c4d48954009c2fe1337d3f5ae7df4b45c35a987' \
    'configs/agentic_stage67_exp4_2.json' | sha256sum --check -
  cmp -s \
    configs/agentic_stage67_exp4_2.json \
    configs/agentic_stage67_exp4_2.new-server.371a60a1.json
  /data/jiahao/task/LlamaFactory/tmp_bash/run_stage67_base_exp4_2.sh \
    --preflight-only
)
```

```bash
# 新服务器正式后台启动：固定单卡 CUDA 0；只覆盖 A1，再仅评测 A1，并重建共享 manifest/bootstrap。
nohup env CUDA_VISIBLE_DEVICES=0 \
  /data/jiahao/task/LlamaFactory/tmp_bash/run_stage67_base_exp4_2.sh \
  > /data/jiahao/task/LlamaFactory/tmp_bash/run_stage67_base_exp4_2.log 2>&1 &
echo $!
```

#### 旧服务器（canonical root `/home/jiahao/task`，历史模板；当前不要启动）

旧服务器的 `old-server.606dcd5f.json` 只冻结了历史 v1 五模式协议，不含 compact-A1，也没有当前旧服务器上四组
artifact 的 adoption 复核证据，因此这里不提供可执行启动命令。若以后切回旧服务器，应先同步 compact planner/runner/
evaluator，按 `/home/jiahao/task` 重新生成 per-run config，核对该机 B1/V1/V2/A0 的 512 行及文件 SHA，再创建带新 SHA
的旧服务器模板并运行 `--preflight-only`；完成这些步骤前不要用 `606dcd5f` 模板启动 A1。

### 当前不执行的 Stage5 与旧四组

现有 Stage5 full replay report 已在当前 data/mesh 路径通过校验；默认流程只读取和核验它，不重新生成。只有报告缺失、
输入/mesh hash 漂移或环境实现改变时，才先停下 A1、确认漂移原因，再单独重建 Stage5 report。

B1/V1/V2/A0 的执行命令已从当前手动入口移除，避免误覆盖 adopted artifact。其冻结身份如下；需要历史复现时应另建
输出目录和新 provenance，不能使用当前正式路径：

| run | mode | controller audit SHA-256 |
| --- | --- | --- |
| B1 | `b1-post-hoc` | `6a4fdd321997b1d979b1ba08768961141e834f37d8d930649249d3a228d8b6ca` |
| V1 | `v1-silent-retry` | `992b17babd72a8d67b77aecf795d1d26d2d867b61bb3120779b7d2291a233f2e` |
| V2 | `v2-silent-dfs` | `b788c2794cbfce0cd722d6ad3e8bb5c72e2567ba22ae7be9a923e66be1692871` |
| A0 | `a0-act-feedback` | `c5da7f4336c7ca92663d71a3c4de2d748182437059a5fcdf31178b80a6238e00` |

### A1
```bash
# 当前唯一需要运行的 controller：同名覆盖 A1，失败 direct child 只向父节点暴露一条摘要；完整轨迹仍写 audit。
# prompt 以 processor 实际 input_ids（含图像 token）计数，硬限 16,384；超限不会进入 CUDA/model.generate。
cd /data/jiahao/task/BrickNet
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  /home/jiahao/miniconda3/envs/bricknet/bin/python scripts/run_bricknet_agentic_a1_compact_inference.py \
  --input /data/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Reasoning/validation/datasets/BrickNet-Stage2-NonThinking-Control-VAL512-Eval.jsonl \
  --output /data/jiahao/task/BrickNet/outputs_val/qwen35_08b/agentic_exp4_2_a1/controller_audit.jsonl \
  --mode a1-feedback-search --backend hf --prompt-protocol exp4_2-stepwise --seed 42 \
  --model Qwen/Qwen3.5-0.8B --model-revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --contract-config /data/jiahao/task/BrickNet/configs/agentic_stage67_exp4_2.json \
  --candidates-per-round 8 --max-rounds-per-state 4 --max-backtrack-depth 3 \
  --stage5-report /data/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/stage5/Stage5-full-replay-report.json
```

每个 controller 命令自动同时生成 `<stem>.predictions.jsonl` 与一个未修复模型层文件：B1 为
`<stem>.raw_first_choice_predictions.jsonl`（`full_path_raw`），V1/A0 为
`<stem>.greedy_observed_candidate0_prefix_predictions.jsonl`，V2/A1 为
`<stem>.search_observed_candidate0_prefix_predictions.jsonl`。stepwise 的 candidate 0 只是 batch 返回顺序，
不代表概率最高；被拒绝时保留原文并立即终止，未被主 DFS 展开的 child 记为 partial，不启动 shadow rollout。

### Eval
下面的统一入口会校验 512 条 ID/order、HF provenance、generation error、文件 hash、首轮 exp4_2 prompt 字节、
完整 chat turn framing、silent-feedback 隔离、accepted-prefix prefill，并从 audit 逐行重建 final 与对应 observed-prefix
文本，分别评测 `final_system` 与分模式诊断层 artifact。统计层只对配置显式允许的 layer 运行
seed-42、10,000 次 paired bootstrap：
V2→A1 只比较 `final_system`，不对 `search_observed_candidate0_prefix` 形成 paired 主结论。

```bash
# 进入 BrickNet；后续所有 Stage6–7 评测命令均从这里执行。
cd /data/jiahao/task/BrickNet

# A1 完成后做 mixed-contract fail-closed 预检；四组 adopted hash 与新 A1 都必须通过。
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  scripts/evaluate_bricknet_agentic_stage67.py --action preflight

# 打印完整执行计划但不运行，便于先检查解释器、输入、输出和外部评测命令。
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  scripts/evaluate_bricknet_agentic_stage67.py --action all --runs a1

# 不要再执行旧的五组全量评测入口；仅重建 A1 final/diagnostic。
BRICKNET_DATA=/data/jiahao/task/BrickNet/data/bricknet_datasets \
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  scripts/evaluate_bricknet_agentic_stage67.py \
  --action evaluate --runs a1 --layers final raw --execute --force

# 采用已有 B1/V1/V2/A0 八层与新 A1 两层，冻结共享 manifest。
BRICKNET_DATA=/data/jiahao/task/BrickNet/data/bricknet_datasets \
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  scripts/evaluate_bricknet_agentic_stage67.py --action manifest --execute

# 基于 mixed-contract manifest 运行 paired bootstrap 与 Markdown 报告。
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  scripts/evaluate_bricknet_agentic_stage67.py --action summarize --execute
```

统一配置位于 `/data/jiahao/task/BrickNet/configs/agentic_stage67_exp4_2.json`；最终产物仍为
`agentic_exp4_2_stage67_manifest.json`、`agentic_exp4_2_stage67_statistics.json` 和
`agentic_exp4_2_stage67_results.md`，本轮直接覆盖同名旧文件。主选择指标是 evaluator 的
pose-aware `task_strict_success`，次指标为
`dense_reward`，最后比较 token/latency/verifier-call 成本；`controller_hard_valid_success` 只证明系统输出满足
硬约束，不能代替 task strict success。

### Experiments Results
B1/V1/V2/A0 已按旧 v1 contract/hash 采用；A1 绑定 active compact revision。A1 与 mixed-contract
manifest/bootstrap 完成后，再按 task strict → dense → hard-valid coverage → 成本更新统一结果账本；
不得把旧四组 audit 的 embedded contract 改写成 active A1 hash。


### exp4_2 固定原始预测重放（B1 同款 post-hoc）

把 exp4_2 冻结的 512 条原始预测作为 replay 输入，走与 B1 完全相同的 post-hoc 验证（不重新采样、纯 CPU、不占
GPU），并在 exp4_2 结果目录的 `post_hoc/` 下产出与 B1 目录同结构的结果文件。该 replay 产物不进入五组 HF 的
`agentic_exp4_2_stage67_manifest`，与「新随机种子的 B1」是两个独立口径。

```bash
# 一键：prepare（冻结预测→replay+provenance）→ run（replay post-hoc）→ check（fail-closed）→ evaluate（final/raw 全套评测）。
cd /home/jiahao/task/BrickNet
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  scripts/exp4_2_post_hoc.py --action all --execute

# 只重放+校验（约 15 秒，不跑渲染/图文指标）：
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python scripts/exp4_2_post_hoc.py --action prepare --execute
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python scripts/exp4_2_post_hoc.py --action run --execute
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python scripts/exp4_2_post_hoc.py --action check
```

输出根为 `outputs_val/qwen35_08b/eval_exp4_2_stage2_nonthinking_control_10k_val512_in16384_out16384_p95_t1_k20/post_hoc/`：
`controller_audit.jsonl`、final/raw 两份 predictions、`replay.jsonl`、`provenance.json`、`consistency_report.json`
与 `eval_final/`、`eval_raw/`。raw 与冻结 `generated_predictions.jsonl` 的 `predict` 逐字节一致；final 只保留通过
全部硬检查的原路径，失败样本为空串。同时 `run_bricknet_agentic_inference.py` 的 replay 分支现在也支持
`--stage5-report` 自动绑定 `BRICKNET_DATA`。


### R1-S
R1-S 64 smoke 已完成：第一次 processor audit 生成 boundary plan，随后由 BrickNet 在 accepted-action 边界重新物化，
第二次 audit 得到 `cutoff_hits=0`。Stage5 full report 现已通过；下一步应重新执行 preflight 刷新 eligibility，
再按下方命令物化并训练正式 R1-S 10k。

#### R1-S smoke test
```bash
# 进入 BrickNet 仓库，构造 R1-S 64 条 protocol smoke 数据。
cd /home/jiahao/task/BrickNet
# 使用通过的 Stage5 report 构造未经切窗的 R1-S 64 条数据。
BRICKNET_DATA=/home/jiahao/task/BrickNet/data/bricknet_datasets \
PYTHONPATH=src:data_preprocess /home/jiahao/miniconda3/envs/bricknet/bin/python \
  data_preprocess/prepare_bricknet_stage8_act_sft.py \
  --size 64 --variants R1-S \
  --stage5-report /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/stage5/Stage5-full-replay-report.json \
  --overwrite

# 创建 64 条 smoke 的 processor audit 输出目录。
mkdir -p /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/smoke64/token_audit
# 切换到 LlamaFactory 运行真实 processor audit。
cd /home/jiahao/task/LlamaFactory
# 首次审计 raw R1-S 64 数据，并生成 accepted-action boundary plan。
PYTHONPATH=src /home/jiahao/miniconda3/envs/llamafactory/bin/python \
  scripts/audit_bricknet_stage8_act_tokens.py \
  examples/train_lora/qwen35_08b_bricknet_stage8_r1_s_act_success_10k.yaml \
  /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/smoke64/BrickNet-Stage8-R1-S.jsonl \
  /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/smoke64/token_audit/BrickNet-Stage8-R1-S.json \
  --dataset-name BrickNet-Stage8-R1-S-64

# 返回 BrickNet，按首次审计产生的 boundary plan 物化切窗数据。
cd /home/jiahao/task/BrickNet
# 在 accepted-action 边界重新构造 R1-S 64，禁止静默截断。
BRICKNET_DATA=/home/jiahao/task/BrickNet/data/bricknet_datasets \
PYTHONPATH=src:data_preprocess /home/jiahao/miniconda3/envs/bricknet/bin/python \
  data_preprocess/prepare_bricknet_stage8_act_sft.py \
  --size 64 --variants R1-S \
  --stage5-report /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/stage5/Stage5-full-replay-report.json \
  --window-plan R1-S=/home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/smoke64/token_audit/BrickNet-Stage8-R1-S.boundary_plan.jsonl \
  --overwrite

# 返回 LlamaFactory，对切窗后的 64 数据进行二次 processor audit。
cd /home/jiahao/task/LlamaFactory
# 验证切窗数据为零错误、零截断且监督 token 完整保留。
PYTHONPATH=src /home/jiahao/miniconda3/envs/llamafactory/bin/python \
  scripts/audit_bricknet_stage8_act_tokens.py \
  examples/train_lora/qwen35_08b_bricknet_stage8_r1_s_act_success_10k.yaml \
  /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/smoke64/BrickNet-Stage8-R1-S.jsonl \
  /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/smoke64/token_audit/BrickNet-Stage8-R1-S.json \
  --dataset-name BrickNet-Stage8-R1-S-64

# 检查 R1-S 64 初始化、数据、token 和 Stage5 gates，不启动训练。
python scripts/launch_bricknet_stage8_act_sft.py --run R1-S --scale 64 --refresh-initialization-audit
# 所有 gate 通过后正式运行 R1-S 64 overfit/protocol smoke 训练。
python scripts/launch_bricknet_stage8_act_sft.py --run R1-S --scale 64 --execute
```

#### R1-S 10k
64 smoke 验收后，用相同的两遍 audit/切窗协议构造并训练正式 R1-S 10k：

```bash
# 进入 BrickNet 仓库，构造正式 R1-S 10k 数据。
cd /home/jiahao/task/BrickNet
# 按 seed-42 manifest 构造未经切窗的 R1-S 10k success-only 数据。
BRICKNET_DATA=/home/jiahao/task/BrickNet/data/bricknet_datasets \
PYTHONPATH=src:data_preprocess /home/jiahao/miniconda3/envs/bricknet/bin/python \
  data_preprocess/prepare_bricknet_stage8_act_sft.py \
  --size 10000 --variants R1-S \
  --stage5-report /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/stage5/Stage5-full-replay-report.json \
  --overwrite

# 创建 R1-S 10k 的 processor audit 输出目录。
mkdir -p /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/10k/token_audit
# 切换到 LlamaFactory 运行 R1-S 10k processor audit。
cd /home/jiahao/task/LlamaFactory
# 首次审计 raw R1-S 10k，并生成 accepted-action boundary plan。
PYTHONPATH=src /home/jiahao/miniconda3/envs/llamafactory/bin/python \
  scripts/audit_bricknet_stage8_act_tokens.py \
  examples/train_lora/qwen35_08b_bricknet_stage8_r1_s_act_success_10k.yaml \
  /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/10k/BrickNet-Stage8-R1-S.jsonl \
  /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/10k/token_audit/BrickNet-Stage8-R1-S.json \
  --dataset-name BrickNet-Stage8-R1-S-10k

# 返回 BrickNet，按 boundary plan 物化 R1-S 10k 切窗数据。
cd /home/jiahao/task/BrickNet
# 在 accepted-action 边界重新构造正式 R1-S 10k。
BRICKNET_DATA=/home/jiahao/task/BrickNet/data/bricknet_datasets \
PYTHONPATH=src:data_preprocess /home/jiahao/miniconda3/envs/bricknet/bin/python \
  data_preprocess/prepare_bricknet_stage8_act_sft.py \
  --size 10000 --variants R1-S \
  --stage5-report /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/stage5/Stage5-full-replay-report.json \
  --window-plan R1-S=/home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/10k/token_audit/BrickNet-Stage8-R1-S.boundary_plan.jsonl \
  --overwrite

# 返回 LlamaFactory，对切窗后的 R1-S 10k 运行二次审计。
cd /home/jiahao/task/LlamaFactory
# 验证 R1-S 10k 为零错误、零截断且 token mix 合法。
PYTHONPATH=src /home/jiahao/miniconda3/envs/llamafactory/bin/python \
  scripts/audit_bricknet_stage8_act_tokens.py \
  examples/train_lora/qwen35_08b_bricknet_stage8_r1_s_act_success_10k.yaml \
  /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/10k/BrickNet-Stage8-R1-S.jsonl \
  /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/10k/token_audit/BrickNet-Stage8-R1-S.json \
  --dataset-name BrickNet-Stage8-R1-S-10k
```

#### R1-S 10k Train & Eval
```bash
# 刷新并检查 R1-S 10k 的 logits 等价、adapter 冻结和全部训练 gate。
python scripts/launch_bricknet_stage8_act_sft.py --run R1-S --scale 10k --refresh-initialization-audit
# 所有 gate 通过后正式训练 R1-S 10k 新 LoRA。
python scripts/launch_bricknet_stage8_act_sft.py --run R1-S --scale 10k --execute
# 检查独立命名的 exp4_2 stage8-act 贪心比较器及其 Stage5/adapter/output gates。
python scripts/launch_bricknet_stage8_controller_eval.py --run S8-ZS-Greedy
# 所有 gate 通过后生成 S8-ZS-Greedy VAL512 final/raw artifact。
python scripts/launch_bricknet_stage8_controller_eval.py --run S8-ZS-Greedy --execute
# 检查 R1-S 与 S8-ZS-Greedy 的 controller/prompt/预算绑定及比较器 artifact。
python scripts/launch_bricknet_stage8_controller_eval.py --run R1-S
# 所有 gate 通过后正式执行 R1-S VAL512 Stage8-act controller 评测。
python scripts/launch_bricknet_stage8_controller_eval.py --run R1-S --execute
# 进入 BrickNet，检查成对 artifact 并预览 final/raw、pose-aware 评测和 paired bootstrap 命令。
cd /home/jiahao/task/BrickNet
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  scripts/evaluate_bricknet_agentic_stage8.py --treatment R1-S --action all
# 执行两层统一评测、冻结 hash manifest 并运行 seed-42 的 10,000 次 paired bootstrap。
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  scripts/evaluate_bricknet_agentic_stage8.py --treatment R1-S --action all --execute
# 返回 LlamaFactory 继续后续 Stage8 命令。
cd /home/jiahao/task/LlamaFactory
```

### R1-C
R1-S 通过 `S8-ZS-Greedy` paired promotion gate 后，按以下顺序收集并构造 R1-C；collector 在每个 GT prefix 只采一个真实 proposal，
accepted proposal 会 rollback 后继续 GT teacher forcing，最终每个 source 最多保留两个稳定 hash 选择的 rejection。

#### R1-C 10k
```bash
# 进入 BrickNet 仓库，使用已训练的 R1-S policy 收集真实 rejection。
cd /home/jiahao/task/BrickNet
# 在每个 GT prefix 上采样 R1-S proposal，并保存真实 rejected proposal 与完整 provenance。
BRICKNET_DATA=/home/jiahao/task/BrickNet/data/bricknet_datasets \
PYTHONPATH=src:data_preprocess /home/jiahao/miniconda3/envs/bricknet/bin/python \
  scripts/collect_bricknet_stage8_r1c.py \
  --base /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-RL/datasets/BrickNet-MM-RL.jsonl \
  --selection-manifest /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Reasoning/stage2/manifests/stage2_train_10k_seed42.jsonl \
  --size 10000 --backend hf --seed 42 \
  --r1s-adapter /home/jiahao/task/LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/train_stage8_r1_s_act_success_10k_ep3_bs1_ga16_lora64_len16384 \
  --output /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/10k/R1-S-policy-rejections.jsonl

# 使用真实 rejection 日志构造未经切窗的 R1-C 10k，目标 token mix 为 80/20。
BRICKNET_DATA=/home/jiahao/task/BrickNet/data/bricknet_datasets \
PYTHONPATH=src:data_preprocess /home/jiahao/miniconda3/envs/bricknet/bin/python \
  data_preprocess/prepare_bricknet_stage8_act_sft.py \
  --size 10000 --variants R1-C \
  --rejection-logs /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/10k/R1-S-policy-rejections.jsonl \
  --stage5-report /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/stage5/Stage5-full-replay-report.json \
  --overwrite

# 切换到 LlamaFactory，首次审计 raw R1-C 10k。
cd /home/jiahao/task/LlamaFactory
# 检查 R1-C message loss、80/20 supervised-token mix，并生成 boundary plan。
PYTHONPATH=src /home/jiahao/miniconda3/envs/llamafactory/bin/python \
  scripts/audit_bricknet_stage8_act_tokens.py \
  examples/train_lora/qwen35_08b_bricknet_stage8_r1_c_act_correction_10k.yaml \
  /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/10k/BrickNet-Stage8-R1-C.jsonl \
  /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/10k/token_audit/BrickNet-Stage8-R1-C.json \
  --dataset-name BrickNet-Stage8-R1-C-10k \
  --baseline-report /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/10k/token_audit/BrickNet-Stage8-R1-S.json

# 返回 BrickNet，按 R1-C boundary plan 重新物化切窗数据。
cd /home/jiahao/task/BrickNet
# 在 accepted-action 边界重新构造 R1-C 10k，保留 rejection/correction 上下文。
BRICKNET_DATA=/home/jiahao/task/BrickNet/data/bricknet_datasets \
PYTHONPATH=src:data_preprocess /home/jiahao/miniconda3/envs/bricknet/bin/python \
  data_preprocess/prepare_bricknet_stage8_act_sft.py \
  --size 10000 --variants R1-C \
  --rejection-logs /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/10k/R1-S-policy-rejections.jsonl \
  --stage5-report /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/stage5/Stage5-full-replay-report.json \
  --window-plan R1-C=/home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/10k/token_audit/BrickNet-Stage8-R1-C.boundary_plan.jsonl \
  --overwrite

# 返回 LlamaFactory，对切窗后的 R1-C 10k 进行二次审计。
cd /home/jiahao/task/LlamaFactory
# 验证 R1-C 零截断、80/20 token mix 和与 R1-S token budget 的绑定。
PYTHONPATH=src /home/jiahao/miniconda3/envs/llamafactory/bin/python \
  scripts/audit_bricknet_stage8_act_tokens.py \
  examples/train_lora/qwen35_08b_bricknet_stage8_r1_c_act_correction_10k.yaml \
  /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/10k/BrickNet-Stage8-R1-C.jsonl \
  /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/10k/token_audit/BrickNet-Stage8-R1-C.json \
  --dataset-name BrickNet-Stage8-R1-C-10k \
  --baseline-report /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/10k/token_audit/BrickNet-Stage8-R1-S.json
```

#### R1-C 10k Train & Eval
```bash
# 刷新并检查 R1-C 的初始化隔离、matched max_steps 和全部训练 gate。
python scripts/launch_bricknet_stage8_act_sft.py --run R1-C --scale 10k --refresh-initialization-audit
# 所有 gate 通过后从 exp4_2 独立初始化并正式训练 R1-C。
python scripts/launch_bricknet_stage8_act_sft.py --run R1-C --scale 10k --execute
# 检查 R1-C 与已有 S8-ZS-Greedy artifact 的 controller/prompt/预算绑定。
python scripts/launch_bricknet_stage8_controller_eval.py --run R1-C
# 所有 gate 通过后正式执行 R1-C VAL512 Stage8-act controller 评测。
python scripts/launch_bricknet_stage8_controller_eval.py --run R1-C --execute
# 进入 BrickNet，检查 R1-C 与 S8-ZS-Greedy 的成对评测契约。
cd /home/jiahao/task/BrickNet
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  scripts/evaluate_bricknet_agentic_stage8.py --treatment R1-C --action all
# 执行 final/raw 统一评测、hash 冻结和 seed-42/10,000 paired bootstrap。
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  scripts/evaluate_bricknet_agentic_stage8.py --treatment R1-C --action all --execute
# 返回 LlamaFactory 继续后续 Stage8 命令。
cd /home/jiahao/task/LlamaFactory
```

### R1-B
R1-S 的 `S8-ZS-Greedy` gate 未改善前停止。R1-C 训练完成并通过同一比较器 gate 后，才检查修复后 A1 日志是否达到
1,000 rollback transitions/100 sources 以构造 R1-B。三个 launcher 均会从 exp4_2 新建 LoRA，并拒绝把
R1-S/R1-C 串行当作下一实验初始化。完成真实 A1 日志转换、R1-B 70/20/10 token-mix 和两遍切窗审计后，入口为：

```bash
# 进入 LlamaFactory；仅在 R1-B rollback 数据和两遍审计 gate 已通过后继续。
cd /home/jiahao/task/LlamaFactory
# 刷新并检查 R1-B 初始化隔离、70/20/10 token mix 和训练 gate。
python scripts/launch_bricknet_stage8_act_sft.py --run R1-B --scale 10k --refresh-initialization-audit
# 所有条件满足后从 exp4_2 独立初始化并正式训练 R1-B。
python scripts/launch_bricknet_stage8_act_sft.py --run R1-B --scale 10k --execute
# 检查独立命名的 exp4_2 stage8-act DFS 比较器及其 Stage5/adapter/output gates。
python scripts/launch_bricknet_stage8_controller_eval.py --run S8-ZS-DFS
# 所有 gate 通过后生成 S8-ZS-DFS VAL512 final/raw artifact。
python scripts/launch_bricknet_stage8_controller_eval.py --run S8-ZS-DFS --execute
# 检查 R1-B 与 S8-ZS-DFS 的 controller/prompt/预算绑定及比较器 artifact。
python scripts/launch_bricknet_stage8_controller_eval.py --run R1-B
# 所有 gate 通过后正式执行 R1-B VAL512 Stage8-act DFS 评测。
python scripts/launch_bricknet_stage8_controller_eval.py --run R1-B --execute
# 进入 BrickNet，检查 R1-B 与 S8-ZS-DFS 的成对评测契约。
cd /home/jiahao/task/BrickNet
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  scripts/evaluate_bricknet_agentic_stage8.py --treatment R1-B --action all
# 执行 final/raw 统一评测、hash 冻结和 seed-42/10,000 paired bootstrap。
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  scripts/evaluate_bricknet_agentic_stage8.py --treatment R1-B --action all --execute
# 返回 LlamaFactory 保持本文后续命令的默认目录。
cd /home/jiahao/task/LlamaFactory
```

这些入口在 rejection/rollback 数据、80/20 或 70/20/10 supervised-token mix、matched max_steps、Stage5 report、
processor、initialization 和 dataset hash 任一条件缺失时都会退出，不会静默训练。Stage8 成对评测还会在
experiment ID、adapter 顺序、首轮 prompt、sampling、seed、预算、Stage5 provenance 或 512 条顺序不一致时拒绝算分。

## PT-exp2 Text250k-only downstream（complete / rowbal 三 epoch 与 ep1/ep2/ep3 完成，推荐 checkpoint-33764，2026-08-29）

`exp4_4_2`（NonThinking-Control）和 `exp4_7_2`（Thinking-Hard V2 Lean-State）已由同一六阶段 wrapper
按顺序完成 train→predict→evaluate；rowbal handoff 随后已完成三 epoch 训练，ep1/ep2/ep3 prediction/evaluation 均已有效完成。
按固定点估计规则推荐 `checkpoint-33764`；不创建 rowbal alias。
两组 train YAML 直接绑定冻结的
`saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp2_text8m_qwen35_08b_path7698261_steps250k_bs4_gbs32_lora64_len6401_nopack`；
其四项输入 hash 与 `global_step=max_steps=250000` 由 launcher fail-closed 校验。两组均为 single-seed
`seed=42`；统计分析、显著性或 paired 比较等待用户后续明确指令。

先运行只读 preflight（默认不执行；GPU gate 固定物理 CUDA 1）：

```bash
cd /data/jiahao/task/LlamaFactory
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_text250k_downstream.py --run exp4_4_2 --action train --gpus 1 --text250k-approved
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_text250k_downstream.py --run exp4_4_2 --action predict --gpus 1 --text250k-approved
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_text250k_downstream.py --run exp4_4_2 --action evaluate --gpus 1 --text250k-approved
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_text250k_downstream.py --run exp4_7_2 --action train --gpus 1 --text250k-approved
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_text250k_downstream.py --run exp4_7_2 --action predict --gpus 1 --text250k-approved
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_text250k_downstream.py --run exp4_7_2 --action evaluate --gpus 1 --text250k-approved
bash tmp_bash/run_exp4_4_2_exp4_7_2_pt_exp2_text250k_cuda1.sh --preflight-only
```

逐阶段正式命令（历史逐阶段入口；当前由同一 wrapper 串行执行，禁止手动重复；训练还必须保留
`--text250k-approved`）：

```bash
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_text250k_downstream.py --run exp4_4_2 --action train --gpus 1 --text250k-approved --execute
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_text250k_downstream.py --run exp4_4_2 --action predict --gpus 1 --text250k-approved --execute
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_text250k_downstream.py --run exp4_4_2 --action evaluate --gpus 1 --text250k-approved --execute
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_text250k_downstream.py --run exp4_7_2 --action train --gpus 1 --text250k-approved --execute
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_text250k_downstream.py --run exp4_7_2 --action predict --gpus 1 --text250k-approved --execute
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_text250k_downstream.py --run exp4_7_2 --action evaluate --gpus 1 --text250k-approved --execute
```

完整六阶段后台入口（历史启动入口；当前 wrapper 已运行，禁止重复启动；wrapper 自带 lock/log/PID 和 fail-stop）：

```bash
cd /data/jiahao/task/LlamaFactory
nohup bash tmp_bash/run_exp4_4_2_exp4_7_2_pt_exp2_text250k_cuda1.sh >/dev/null 2>&1 &
```

### 正式启动状态（2026-08-26 13:40 +08:00；当前 handoff 状态另见上文）

- 六阶段 wrapper preflight exit=`0` 后已正式启动，wrapper PID=`2773476`；固定顺序仍为
  `exp4_4_2 train -> predict -> evaluate -> exp4_7_2 train -> predict -> evaluate`。
- 当时执行 `exp4_4_2` train 的首次 token-cache 构建；`exp4_7_2` 尚未启动并由同一 wrapper 排队。
- 当时没有新 VAL512 denominator 或指标；paired bootstrap、显著性检验和跨实验统计均未启动，等待两个实验
  的评测指标完成后由用户另行指示。

### `exp4_4_2` 完成、`exp4_7_2` 启动（2026-08-26 19:46 +08:00）

- `exp4_4_2` train 完成：`global_step=max_steps=1875`、train loss=`0.15205654106140137`、runtime=
  `11362.8679s`；最终 adapter config/model SHA-256 分别为
  `75eb487d1fa1bb7fc48a1bdb489c1c4e570442d6df4132ea8cd4b34d605f3808` /
  `ae1b4b386ff779f0e6886e16287db2779147de7b1778e3c56194aaf9f73fc77a`。
- prediction 为 512 rows，runtime=`9453.7019s`；`generated_predictions.jsonl` SHA-256=
  `14f07b478c8847f7706d7b2f94ed9033df9b9a39df028950cd039a48906cb0b7`。
- evaluation 的 generated/path/scored/alignment_input/alignment 均为 512 rows；409 个 fully parsable 样本均
  完成渲染和三项图像指标，render failure=`0`。核心指标：parsable=`409/512`、clean=`117/512`、Dense=
  `0.5984927319348795`、Strict=`14/512`、inventory F1=`0.8837379571053656`、length=
  `0.8651899315810889`、collision prefix=`0.5350824952247927`、pose=`0.14479307335952754`。
- `metrics.json` SHA-256=`bb2ced3a221d3a38d94bbf2c7396b36ed1532514b9d28ce2366d180cf6b917c0`；
  `alignment_manifest.json` SHA-256=`b29c5dc929464f5d95a7a90358c1e30c4e97845966f06f8ed533fdcfd6b9fea8`，
  status=`complete`、freeze=`post_evaluation_freeze`。结果有效。
- wrapper 已通过 completion gate 进入 `exp4_7_2` train；当前尚无 `exp4_7_2` 指标。未运行跨实验统计、
  paired bootstrap 或显著性检验。

### Text250k-only 六阶段完成（2026-08-27 07:27 +08:00）

- `exp4_7_2` train 完成：`global_step=max_steps=1875`、train loss=`0.10840269915262858`、runtime=
  `10780.8035s`；最终 adapter config/model SHA-256 分别为
  `e90bb5e9a75d722954d97b2845813d2af640586bc5ddbdf6735fdcf5823a1dd2` /
  `f909b6ce6c6f33cb8474fbe432ac68d4a7a2cdcf1d0483eefb73750f34c964c9`。
- prediction 为 512 rows，runtime=`30490.9119s`；generated/path SHA-256 分别为
  `ec1a6002bed26936e681aacc4ad49f007803b13a1b9c4ff69e774ee29eae396a` /
  `86618878002cf475c55a81c64d150950a78be5252f490f4eadc33ef78fc93237`。
- evaluation 的五个核心 JSONL 均为 512 rows；423 个 fully parsable 样本均有完整 8-view render 与
  PE/SigLIP2/VQA，render failure=`0`。核心指标：parsable=`423/512`、clean=`130/512`、Dense=
  `0.608682876136042`、Strict=`11/512`、inventory F1=`0.900108828017569`、length=
  `0.9150935723166516`、collision prefix=`0.5436165505760034`、pose=`0.1396889334201416`。
- `metrics.json` SHA-256=`4ae6829f4cc65f14e0a855adf334fd0b8e63e7d43661c0a23f56c824046ab716`；
  `alignment_manifest.json` SHA-256=`41ca88501c1403fafce3c80478cd5e017b1b6609d62b6856e40979950947ff42`，
  status=`complete`、freeze=`post_evaluation_freeze`。trace-format-valid=`36/512` 作为输出质量字段保留；
  canonical 512-row 评测与 manifest 契约有效。
- Text250k wrapper 于 `2026-08-27 07:27:45 +08:00` complete、exit status=`0`，PID=`2773476` 已退出。
  正式日志中没有 paired bootstrap、McNemar、statistics 或显著性检验命令；本轮到此停止并等待用户指令。

## rowbal-cont3 ep3 downstream（`exp4_4_3` / `exp4_7_3`，complete / valid，2026-09-01）

最终状态：两组均已完成 `train → predict → evaluate` 并通过独立验证；详细数值和 canonical 结果记录见
[`experiment_results.md`](experiment_results.md)。完整串行链由 waiter 于 `2026-08-31 16:34:00 +08:00` 交接，
于 `2026-09-01 10:14:33 +08:00` 以 exit status=`0` 完成，总耗时约 `17:40:33`；运行 PID 已清理，当前无本项目进程。
本轮使用物理 CUDA0；完成后 CUDA0 的其他用户占用不影响结果，也未被干预。不要重复启动该 wrapper。

下文的 preflight、早停和资源等待内容均为历史 provenance；其中的 queued/waiting 只描述当时快照，当前状态以本节
“最终完成证据”为准。

新增两个独立 Stage2 SFT 端点敏感性实验：`exp4_4_3` 使用 NonThinking-Control 10k，
`exp4_7_3` 使用 Thinking-Hard V2 Lean-State 10k。两者均直接绑定用户指定的
`PT-exp2-mm-rowbal-cont3` ep3 `checkpoint-50646`，各自新建 SFT LoRA；不使用 alias，不重复加载
text250k adapter，也不串接两个 SFT adapter。该 ep3 分支是用户指定的 endpoint sensitivity arm；
rowbal 内部按固定点估计规则推荐的仍是 ep2 `checkpoint-33764`，不得将 ep3 写成“rowbal 最优”。

准备阶段已完成配置、launcher、测试、命令与文档准备。2026-08-30 05:10:50 +08:00 的正式
preflight exit=`0`，启动前 focused regression=`53 passed`。随后于 05:11:03 +08:00 曾启动
CUDA0 wrapper `tmp_bash/run_exp4_4_3_exp4_7_3_pt_exp2_mm_rowbal_cont3_cuda0.sh`，wrapper PID=`1588945`，
exec session 由主代理持有；该历史尝试的首阶段为 `exp4_4_3 train`，后续状态更正见下文。启动时物理
CUDA0 为空闲，可用磁盘约 `74 GiB`；训练入口保留显式 `--rowbal-ep3-approved` 要求。

此前准备阶段使用的 CUDA1 wrapper 仅保留作历史入口，不用于本轮；本轮统一使用 CUDA0 launcher 和 wrapper。

逐阶段只读 preflight（本轮已执行并通过；需要复核时可直接执行，不启动实验）：

```bash
cd /data/jiahao/task/LlamaFactory
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3_downstream.py --run exp4_4_3 --action train --gpus 0 --rowbal-ep3-approved
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3_downstream.py --run exp4_4_3 --action predict --gpus 0
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3_downstream.py --run exp4_4_3 --action evaluate --gpus 0
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3_downstream.py --run exp4_7_3 --action train --gpus 0 --rowbal-ep3-approved
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3_downstream.py --run exp4_7_3 --action predict --gpus 0
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3_downstream.py --run exp4_7_3 --action evaluate --gpus 0
bash tmp_bash/run_exp4_4_3_exp4_7_3_pt_exp2_mm_rowbal_cont3_cuda0.sh --preflight-only
```

逐阶段正式入口（CUDA0 空闲后由 wrapper 按顺序执行；当前不要单独执行）：

```bash
cd /data/jiahao/task/LlamaFactory
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3_downstream.py --run exp4_4_3 --action train --gpus 0 --rowbal-ep3-approved --execute
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3_downstream.py --run exp4_4_3 --action predict --gpus 0 --execute
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3_downstream.py --run exp4_4_3 --action evaluate --gpus 0 --execute
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3_downstream.py --run exp4_7_3 --action train --gpus 0 --rowbal-ep3-approved --execute
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3_downstream.py --run exp4_7_3 --action predict --gpus 0 --execute
/home/jiahao/miniconda3/envs/llamafactory/bin/python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3_downstream.py --run exp4_7_3 --action evaluate --gpus 0 --execute
```

本轮历史启动的六阶段串行入口（早停后待重启，不要重复启动另一实例）：

```bash
cd /data/jiahao/task/LlamaFactory
nohup bash tmp_bash/run_exp4_4_3_exp4_7_3_pt_exp2_mm_rowbal_cont3_cuda0.sh >/dev/null 2>&1 &
```

历史持久等待入口（已启动并在完成后退出；不要重复执行）：

```bash
cd /data/jiahao/task/LlamaFactory
nohup setsid bash tmp_bash/wait_for_exp4_4_3_exp4_7_3_pt_exp2_mm_rowbal_cont3_cuda0.sh </dev/null >/dev/null 2>&1 &
```

该 waiter 于 `2026-08-31 14:10 +08:00` 启动，PID=`2197540`（PPID=`1`、PGID=SID=`2197540`），
已在最终串行链完成后退出。不要再次执行上面的启动命令，以免产生重复 waiter。以下均为历史只读状态检查：

```bash
cd /data/jiahao/task/LlamaFactory
WAIT_PID_FILE=tmp_bash/wait_for_exp4_4_3_exp4_7_3_pt_exp2_mm_rowbal_cont3_cuda0.pid
WAIT_PID=$(sed -n '1p' "$WAIT_PID_FILE")
printf 'wait PID: %s\n' "$WAIT_PID"
ps -o pid,ppid,pgid,sid,stat,etime,cmd -p "$WAIT_PID"
tail -n 40 tmp_bash/wait_for_exp4_4_3_exp4_7_3_pt_exp2_mm_rowbal_cont3_cuda0.log
tail -n 40 tmp_bash/run_exp4_4_3_exp4_7_3_pt_exp2_mm_rowbal_cont3_cuda0.log
```

wrapper 固定顺序为 `exp4_4_3 train -> predict -> evaluate -> exp4_7_3 train -> predict -> evaluate`，
并带有 lock/log/PID、`>=40 GiB` 磁盘门、CUDA0 占用门和 fail-stop。它不会杀死或修改已有 GPU
进程；任一门失败时停止，不继续下一阶段。上次尝试实例 PID=`1588945` 已早停；不得把它视为当前运行实例，
也不得预填 `_3` 的 prediction/evaluation/metrics 结果。

### 当前状态更正（2026-08-31）

- 上述 `2026-08-30 05:11:03` 启动是一次早停的历史尝试：日志观测到初始 tokenizer/cache 处理约
  `2500/10000`，训练进度仍为 `0/1875`；该尝试于 `05:15:19` 结束。日志末尾的 `exit status=0`
  仅是外层 wrapper 返回值，不能作为训练完成证据。
- 该尝试未形成可用的 train adapter、完整 token-cache、prediction 或 evaluation 产物；
  `exp4_7_3` 未启动。因此不执行任何中间状态恢复，也不登记新结果。
- 截至 `2026-08-31`，物理 CUDA0 被其他用户 `lingyu` 的 `PID=1908540`（`python cog5b.py`）占用。
  不杀死、不暂停、不干预该进程；当前实验状态为 `queued / waiting for CUDA0`。
- 2026-08-31 14:10 +08:00 已实际启动独立 `nohup+setsid` CUDA0 waiter：PID=`2197540`、PPID=`1`、
  PGID=SID=`2197540`，日志为 `tmp_bash/wait_for_exp4_4_3_exp4_7_3_pt_exp2_mm_rowbal_cont3_cuda0.log`。
  waiter 每 60 秒执行只读检查，不杀死或干预任何进程；仅在 CUDA0 空闲、可用磁盘至少 `40 GiB` 且
  preflight 通过后，才会 `exec` 原完整串行 wrapper。
- CUDA0 空闲后，应从 `exp4_4_3 train` 重新运行完整的 fail-stop 串行链
  `exp4_4_3 train → predict → evaluate → exp4_7_3 train → predict → evaluate`；旧 PID=`1588945`
  不代表当前仍有运行实例。

### 最终完成证据（2026-09-01 10:14:33 +08:00）

- waiter 于 `2026-08-31 16:34:00 +08:00` 通过 CUDA0 空闲、磁盘和 preflight gate 后交接完整串行链；
  `exp4_4_3 train → predict → evaluate → exp4_7_3 train → predict → evaluate` 于
  `2026-09-01 10:14:33 +08:00` 以 exit status=`0` 完成，总耗时约 `17:40:33`。运行 PID 已清理，当前无本项目进程。
- `exp4_4_3` train=`global_step=max_steps=1875`、epoch=`3`、loss=`0.14706837952931723`、runtime=`9550.3512s`；
  prediction=`512/512`、runtime=`2:34:38.41`；parsable=`406/512`、clean=`123/512`、dense=`0.6059268408483144`、
  strict=`10/512`、trace-format-valid=`512/512`。
- `exp4_7_3` train=`global_step=max_steps=1875`、epoch=`3`、loss=`0.10613786784807841`、runtime=`11413.0456s`；
  prediction=`512/512`、runtime=`8:43:33.35`；parsable=`415/512`、clean=`116/512`、dense=`0.609875326115232`、
  strict=`12/512`、trace-format-valid=`31/512`、nonempty=`511/512`。该 trace 数字是输出质量 warning，不是 evaluator 失败。
- 两组的 generated/path/scored/alignment 均为 `512` rows，evaluation/alignment manifest 均为 `complete`。
  `exp4_4_3` 的 `metrics.json` / `alignment_manifest.json` SHA-256 分别为
  `fb14f0a334beb0d0fc1ad2321c0e319a04ca5941f83f0127a276b77793b44d74` /
  `55538e77f948976114e562b6a5e732043adbe3bad2cb5428fcde686209f50208`；`exp4_7_3` 分别为
  `c922e214c01cfd6c4d8b69e05fbef7b25259d5801cf19313dc19a863f4ea9ff2` /
  `b0815881ca2a5395936f092dfc7092366c50542b02d6876ffd90dfa5cc296daa`。
- `exp4_4_3` / `exp4_7_3` 已从 queued/waiting 状态转为 `complete / valid`，不创建 alias；尚未运行 paired
  bootstrap/statistics，不宣称跨初始化显著 winner。历史早停记录和旧 wrapper PID 仅保留作 provenance。

## Official SFT media render handoff（2026-09-06；validated media，downstream pending）

正式官方 SFT media layer 已完成并通过 strict verify/completion gate。正式状态文件为
`/data/jiahao/task/LlamaFactory/tmp_bash/supervise_official_sft_render.status`，其结果为
`COMPLETE/OK`；strict verify=`PASS`，completion gate=`PASS`；共 `67178` rows、`537424` raw views、
`67178` collages、`failed=0`。本轮 PT/VAL 不在范围内；本轮仅完成媒体层与 provenance，
projection、dataset registry、token cache 和训练接线仍为 pending，现有 PT/SFT/评测仍使用 v1 images。

正式数据路径：

```text
/data/jiahao/task/BrickNet/outputs_gt/sft_8view_renders_v2_rowids
/data/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM/image_v2_official/SFT
```

正式 metadata、配置、launcher 和证据：

```text
/data/jiahao/task/BrickNet/outputs_gt/.bricknet_render_v2_official/sft_identity.json
/data/jiahao/task/BrickNet/outputs_gt/.bricknet_render_v2_official/sft_progress.json
/data/jiahao/task/BrickNet/outputs_gt/.bricknet_render_v2_official/sft_summary.json
/data/jiahao/task/BrickNet/outputs_gt/.bricknet_render_v2_official/sft_timings.jsonl
/data/jiahao/task/BrickNet/configs/bricknet_mm_image_v2_official_render.json
/data/jiahao/task/BrickNet/scripts/render_bricknet_render_8views.py
/data/jiahao/task/BrickNet/scripts/render_bricknet_render_8views_official.sh
/data/jiahao/task/BrickNet/scripts/generate_bricknet_sft_completion_gate.py
/data/jiahao/task/LlamaFactory/tmp_bash/supervise_official_sft_render.status
/data/jiahao/task/LlamaFactory/tmp_bash/supervise_official_sft_render-20260902T194605Z-381805.log
/data/jiahao/task/LlamaFactory/tmp_bash/supervise_official_sft_render-20260902T194605Z-381805.gpu-peaks.tsv
/data/jiahao/task/LlamaFactory/tmp_bash/sft_official_verify_20260906.log
/data/jiahao/task/BrickNet/outputs_gt/.bricknet_render_v2_official/sft_completion_gate.json
```

最终证据：verify log SHA-256=`9afecbc4717484dd55331b93f496045eff24b2079881e133242b59625d22166a`；
completion gate SHA-256=`7b68c82e1116aa72d7168c6087858168ca886c57413f841ac3f93b465e325f97`，生成于
`2026-09-06 02:39 +08:00`。raw/collage logical bytes 分别为 `119394717747` / `22019373017`；
selected row IDs SHA-256=`c9d740a06f550b750e1fa6af58d3f874e8ec75be87864c0272cb7483b28a62b8`。

渲染身份是 CYCLES/OPTIX、GPU `0,1`、8 views、`512x512`、256 samples、seed `0`。config
默认 `workers_per_gpu=8`，正式 CLI/supervisor 实际覆盖为 `16/GPU`；迁移和结果解释必须保留这个
差异。pilot `/data/jiahao/task/BrickNet-Render/tmp/official_sft_pilot_20260902T194605Z-381805`
只有 32 rows，不是正式全量数据。

### Strict verify 和 completion gate（已通过）

只有严格 verify 日志出现下面的完整 pass marker，才能生成并声称 gate 通过：

```text
[SFT] verify passed: 67178 rows, exactly 8 views and valid 1024x512 RGB collages
```

上述 marker 已出现且 strict verify 已 exit `0`；当前 completion gate 已为 `PASS`。verify log 与 gate 的
SHA-256 和生成时间见上方最终证据。下面命令保留为重建/复核入口；它仍要求 supervisor status、render log、
GPU peaks 和 strict marker 全部存在，任何证据缺失时 gate 程序都会 fail-closed，不会写出误导性的 gate：

```bash
set -euo pipefail

export SFT_BRICKNET_ROOT=/data/jiahao/task/BrickNet
export SFT_VERIFY_LOG=/data/jiahao/task/LlamaFactory/tmp_bash/sft_official_verify_20260906.log
export SFT_SUPERVISOR_STATUS=/data/jiahao/task/LlamaFactory/tmp_bash/supervise_official_sft_render.status
export SFT_RENDER_LOG=/data/jiahao/task/LlamaFactory/tmp_bash/supervise_official_sft_render-20260902T194605Z-381805.log
export SFT_GPU_PEAKS=/data/jiahao/task/LlamaFactory/tmp_bash/supervise_official_sft_render-20260902T194605Z-381805.gpu-peaks.tsv
export SFT_META_ROOT="${SFT_BRICKNET_ROOT}/outputs_gt/.bricknet_render_v2_official"
export SFT_GATE_PATH="${SFT_META_ROOT}/sft_completion_gate.json"

test -s "${SFT_VERIFY_LOG}"
grep -F \
  '[SFT] verify passed: 67178 rows, exactly 8 views and valid 1024x512 RGB collages' \
  "${SFT_VERIFY_LOG}"

cd "${SFT_BRICKNET_ROOT}"
python scripts/generate_bricknet_sft_completion_gate.py \
  --config "${SFT_BRICKNET_ROOT}/configs/bricknet_mm_image_v2_official_render.json" \
  --metadata-root "${SFT_META_ROOT}" \
  --verify-log "${SFT_VERIFY_LOG}" \
  --supervisor-status "${SFT_SUPERVISOR_STATUS}" \
  --render-log "${SFT_RENDER_LOG}" \
  --gpu-peaks "${SFT_GPU_PEAKS}" \
  --effective-workers-per-gpu 16 \
  --output "${SFT_GATE_PATH}"

jq -e \
  '.status == "PASS" and .scope.split == "SFT" and
   .scope.out_of_scope == ["PT", "VAL"] and
   .effective_runtime.config_workers_per_gpu == 8 and
   .effective_runtime.effective_workers_per_gpu == 16 and
   .downstream.projection_activated == false and
   (.metadata_files | has("config") and has("identity") and has("progress") and has("summary") and has("timings")) and
   .strict_verify.status == "PASS" and
   .strict_verify.pass_marker_count >= 1 and
   .supervisor_provenance.status_values.stage == "COMPLETE" and
   .supervisor_provenance.status_values.result == "OK" and
   (.supervisor_provenance.gpu_peaks.rows | map(.gpu_id) | sort) == [0, 1]' \
  "${SFT_GATE_PATH}"
```

`sft.lock` 是可变 sentinel，不能进入 gate identity 或迁移清单；即使目标端已有 stale lock，也要先
确认无 render/verify 进程后再由人工决定是否清理，禁止把 lock 当成完成证据。

### 从当前工作区同步到另一台服务器

代码、配置和文档不通过 rsync 整个 dirty worktree：源端先审查相关 diff，选择性提交并 push；目标端
先查看并保存/提交自身变更，再 `fetch` 和 `pull --ff-only`。这样不会覆盖目标机同名的未提交修改。
下面的数据 rsync 不使用 `--delete`，源端和目标端路径均显式设置，变量名不依赖 `HOME`。目标目录
应先确认建议至少 `160 GiB` 可用空间（raw 约 113G、collages 约 21G，metadata/evidence 另计）。

#### 默认 Git 主方案（推荐；不要与 patch fallback 同时执行）

当前两个源仓库实测分支均为 `main`。下面代码块自足：先审查，再选择性 staging，提交前检查 staged
内容，最后 push；目标端先审查并保存自身变更，再只允许 `pull --ff-only origin main`。如果用户明确选择
其他分支，统一替换 `SFT_MIG_GIT_BRANCH`，并同步修改 push/pull 参数。

```bash
set -euo pipefail

export SFT_MIG_TARGET_HOST='jiahao@10.119.46.67'
export SFT_MIG_SOURCE_TASK_ROOT=/data/jiahao/task
export SFT_MIG_TARGET_TASK_ROOT=/home/jiahao/task
export SFT_MIG_SOURCE_BRICKNET="${SFT_MIG_SOURCE_TASK_ROOT}/BrickNet"
export SFT_MIG_SOURCE_LLAMAF="${SFT_MIG_SOURCE_TASK_ROOT}/LlamaFactory"
export SFT_MIG_TARGET_BRICKNET="${SFT_MIG_TARGET_TASK_ROOT}/BrickNet"
export SFT_MIG_TARGET_LLAMAF="${SFT_MIG_TARGET_TASK_ROOT}/LlamaFactory"
export SFT_MIG_GIT_BRANCH=main
export SFT_STAGE_ONE_SHOT_LAUNCHER=0

# Source: inspect first, then stage only the reviewed code/config/docs files.
git -C "${SFT_MIG_SOURCE_BRICKNET}" status --short
git -C "${SFT_MIG_SOURCE_BRICKNET}" diff --stat
git -C "${SFT_MIG_SOURCE_BRICKNET}" diff -- \
  data_preprocess \
  'BrickNet-MM Agentic LEGO Planner' \
  scripts/render_bricknet_render_8views.py \
  tests/test_render_bricknet_render_8views.py
git -C "${SFT_MIG_SOURCE_LLAMAF}" status --short
git -C "${SFT_MIG_SOURCE_LLAMAF}" diff --stat
git -C "${SFT_MIG_SOURCE_LLAMAF}" diff -- SERVER_MIGRATION.md record.md experiment_results.md bricknet-pt-exp2.md

test "$(git -C "${SFT_MIG_SOURCE_BRICKNET}" branch --show-current)" = "${SFT_MIG_GIT_BRANCH}"
test "$(git -C "${SFT_MIG_SOURCE_LLAMAF}" branch --show-current)" = "${SFT_MIG_GIT_BRANCH}"
# Existing dirty docs/driver/tests require interactive hunk review; do not stage them wholesale.
git -C "${SFT_MIG_SOURCE_BRICKNET}" add -p -- \
  data_preprocess \
  'BrickNet-MM Agentic LEGO Planner' \
  scripts/render_bricknet_render_8views.py \
  tests/test_render_bricknet_render_8views.py
# New official config/launcher/gate implementation and its test are exact, separately reviewed paths.
git -C "${SFT_MIG_SOURCE_BRICKNET}" add -- \
  configs/bricknet_mm_image_v2_official_render.json \
  scripts/render_bricknet_render_8views_official.sh \
  scripts/generate_bricknet_sft_completion_gate.py \
  tests/test_generate_bricknet_sft_completion_gate.py
# LlamaFactory's four dirty handoff docs also require interactive hunk review.
git -C "${SFT_MIG_SOURCE_LLAMAF}" add -p -- \
  SERVER_MIGRATION.md record.md experiment_results.md bricknet-pt-exp2.md
# Optional provenance: opt in only if this relevant one-shot launcher is intentionally versioned.
if [[ "${SFT_STAGE_ONE_SHOT_LAUNCHER}" == 1 ]]; then
  git -C "${SFT_MIG_SOURCE_LLAMAF}" add -- tmp_bash/supervise_official_sft_render.sh
fi
# Never stage generated status/log/TSV/GPU-error/PID/lock evidence: supervise_official_sft_render.status,
# supervise_official_sft_render-*.log, supervise_official_sft_render-*.tsv, *.gpu.error, *.pid, and *.lock.

# Required pre-commit review: inspect all staged changes before either commit.
for SFT_SOURCE_REPO in "${SFT_MIG_SOURCE_BRICKNET}" "${SFT_MIG_SOURCE_LLAMAF}"; do
  git -C "${SFT_SOURCE_REPO}" diff --cached --check
  git -C "${SFT_SOURCE_REPO}" diff --cached --stat
  git -C "${SFT_SOURCE_REPO}" diff --cached
done
# Pause here for human review of the printed staged diff, then commit and push the two main branches.
git -C "${SFT_MIG_SOURCE_BRICKNET}" commit -m 'Add official SFT render completion gate and handoff'
git -C "${SFT_MIG_SOURCE_BRICKNET}" push origin "${SFT_MIG_GIT_BRANCH}"
git -C "${SFT_MIG_SOURCE_LLAMAF}" commit -m 'Document official SFT migration handoff'
git -C "${SFT_MIG_SOURCE_LLAMAF}" push origin "${SFT_MIG_GIT_BRANCH}"

# Target: inspect its worktrees first. Save/review and commit target-owned changes before pulling.
ssh "${SFT_MIG_TARGET_HOST}" bash -s -- \
  "${SFT_MIG_TARGET_BRICKNET}" "${SFT_MIG_TARGET_LLAMAF}" "${SFT_MIG_GIT_BRANCH}" <<'REMOTE'
set -euo pipefail
SFT_TARGET_GIT_BRANCH=$3
for SFT_TARGET_REPO in "$1" "$2"; do
  cd "${SFT_TARGET_REPO}"
  test "$(git branch --show-current)" = "${SFT_TARGET_GIT_BRANCH}"
  git status --short
  git diff --stat
  # After the target owner has saved/committed its own changes, continue:
  test -z "$(git status --porcelain)"
  git fetch origin "${SFT_TARGET_GIT_BRANCH}"
  git pull --ff-only origin "${SFT_TARGET_GIT_BRANCH}"
done
REMOTE
```

#### Patch fallback（仅主方案未执行时；不要在 primary 成功后执行）

该代码块与上面的 Git 主方案互斥、且自足。它只生成 tracked 文件 patch；`git diff HEAD` 为空通常表示
主方案已经提交，遇到这种情况会 fail-closed。未跟踪的 config/launcher/gate/test 仍按下面的显式路径
另行传输，不能期待它们出现在 patch 中。

```bash
set -euo pipefail

export SFT_MIG_TARGET_HOST='user@target-server'
export SFT_MIG_SOURCE_TASK_ROOT=/data/jiahao/task
export SFT_MIG_TARGET_TASK_ROOT=/data/jiahao/task
export SFT_MIG_SOURCE_BRICKNET="${SFT_MIG_SOURCE_TASK_ROOT}/BrickNet"
export SFT_MIG_SOURCE_LLAMAF="${SFT_MIG_SOURCE_TASK_ROOT}/LlamaFactory"
export SFT_MIG_TARGET_BRICKNET="${SFT_MIG_TARGET_TASK_ROOT}/BrickNet"
export SFT_MIG_TARGET_LLAMAF="${SFT_MIG_TARGET_TASK_ROOT}/LlamaFactory"
export SFT_MIG_GIT_BRANCH=main
export SFT_STAGE_ONE_SHOT_LAUNCHER=0

test "$(git -C "${SFT_MIG_SOURCE_BRICKNET}" branch --show-current)" = "${SFT_MIG_GIT_BRANCH}"
test "$(git -C "${SFT_MIG_SOURCE_LLAMAF}" branch --show-current)" = "${SFT_MIG_GIT_BRANCH}"
if git -C "${SFT_MIG_SOURCE_BRICKNET}" diff HEAD --quiet -- \
    data_preprocess 'BrickNet-MM Agentic LEGO Planner' \
    scripts/render_bricknet_render_8views.py tests/test_render_bricknet_render_8views.py; then
  echo 'No BrickNet tracked diff: do not run patch fallback after a successful primary commit.' >&2
  exit 2
fi
if git -C "${SFT_MIG_SOURCE_LLAMAF}" diff HEAD --quiet -- \
    SERVER_MIGRATION.md record.md experiment_results.md bricknet-pt-exp2.md; then
  echo 'No LlamaFactory tracked diff: do not run patch fallback after a successful primary commit.' >&2
  exit 2
fi

export SFT_MIG_PATCH_OUT=/tmp/bricknet_sft_handoff_reviewed.patch
git -C "${SFT_MIG_SOURCE_BRICKNET}" diff HEAD --binary -- \
  data_preprocess \
  'BrickNet-MM Agentic LEGO Planner' \
  scripts/render_bricknet_render_8views.py \
  tests/test_render_bricknet_render_8views.py > "${SFT_MIG_PATCH_OUT}"
test -s "${SFT_MIG_PATCH_OUT}"
export SFT_MIG_LLAMAF_PATCH_OUT=/tmp/llamafactory_sft_handoff_reviewed.patch
git -C "${SFT_MIG_SOURCE_LLAMAF}" diff HEAD --binary -- \
  SERVER_MIGRATION.md record.md experiment_results.md bricknet-pt-exp2.md > "${SFT_MIG_LLAMAF_PATCH_OUT}"
test -s "${SFT_MIG_LLAMAF_PATCH_OUT}"
export SFT_MIG_TARGET_BRICKNET_PATCH=/tmp/bricknet_official_sft_handoff_20260906.patch
export SFT_MIG_TARGET_LLAMAF_PATCH=/tmp/llamafactory_official_sft_handoff_20260906.patch
ssh "${SFT_MIG_TARGET_HOST}" "test ! -e '${SFT_MIG_TARGET_BRICKNET_PATCH}' && test ! -e '${SFT_MIG_TARGET_LLAMAF_PATCH}'"
scp "${SFT_MIG_PATCH_OUT}" \
  "${SFT_MIG_TARGET_HOST}:${SFT_MIG_TARGET_BRICKNET_PATCH}"
scp "${SFT_MIG_LLAMAF_PATCH_OUT}" \
  "${SFT_MIG_TARGET_HOST}:${SFT_MIG_TARGET_LLAMAF_PATCH}"

# Target: inspect and save/commit target-owned changes before applying either patch.
ssh "${SFT_MIG_TARGET_HOST}" bash -s -- \
  "${SFT_MIG_TARGET_BRICKNET}" "${SFT_MIG_TARGET_LLAMAF}" \
  "${SFT_MIG_TARGET_BRICKNET_PATCH}" "${SFT_MIG_TARGET_LLAMAF_PATCH}" \
  "${SFT_MIG_GIT_BRANCH}" <<'REMOTE'
set -euo pipefail
SFT_TARGET_BRICKNET=$1
SFT_TARGET_LLAMAF=$2
SFT_TARGET_BRICKNET_PATCH=$3
SFT_TARGET_LLAMAF_PATCH=$4
SFT_TARGET_GIT_BRANCH=$5
for SFT_TARGET_REPO in "${SFT_TARGET_BRICKNET}" "${SFT_TARGET_LLAMAF}"; do
  cd "${SFT_TARGET_REPO}"
  test "$(git branch --show-current)" = "${SFT_TARGET_GIT_BRANCH}"
  git status --short
  git diff --stat
  # The target owner must save/commit its own changes before continuing.
  test -z "$(git status --porcelain)"
done
git -C "${SFT_TARGET_BRICKNET}" apply --check "${SFT_TARGET_BRICKNET_PATCH}"
git -C "${SFT_TARGET_LLAMAF}" apply --check "${SFT_TARGET_LLAMAF_PATCH}"
git -C "${SFT_TARGET_BRICKNET}" apply "${SFT_TARGET_BRICKNET_PATCH}"
git -C "${SFT_TARGET_LLAMAF}" apply "${SFT_TARGET_LLAMAF_PATCH}"
REMOTE

echo "Both patches passed target git apply --check and were applied; tracked patch files contain no untracked files."
echo "If either check fails, the target is not on the same baseline; do not force-apply."

ssh "${SFT_MIG_TARGET_HOST}" "mkdir -p \
  '${SFT_MIG_TARGET_BRICKNET}/configs' \
  '${SFT_MIG_TARGET_BRICKNET}/scripts' \
  '${SFT_MIG_TARGET_BRICKNET}/tests' \
  '${SFT_MIG_TARGET_LLAMAF}/tmp_bash'"
for SFT_UNTRACKED_PATH in \
  configs/bricknet_mm_image_v2_official_render.json \
  scripts/render_bricknet_render_8views_official.sh \
  scripts/generate_bricknet_sft_completion_gate.py \
  tests/test_generate_bricknet_sft_completion_gate.py; do
  test -f "${SFT_MIG_SOURCE_BRICKNET}/${SFT_UNTRACKED_PATH}"
  rsync -a --partial \
    "${SFT_MIG_SOURCE_BRICKNET}/${SFT_UNTRACKED_PATH}" \
    "${SFT_MIG_TARGET_HOST}:${SFT_MIG_TARGET_BRICKNET}/${SFT_UNTRACKED_PATH}"
done
if [[ "${SFT_STAGE_ONE_SHOT_LAUNCHER}" == 1 && \
      -f "${SFT_MIG_SOURCE_LLAMAF}/tmp_bash/supervise_official_sft_render.sh" ]]; then
  rsync -a --partial \
    "${SFT_MIG_SOURCE_LLAMAF}/tmp_bash/supervise_official_sft_render.sh" \
    "${SFT_MIG_TARGET_HOST}:${SFT_MIG_TARGET_LLAMAF}/tmp_bash/supervise_official_sft_render.sh"
fi
```

#### 大数据与 evidence rsync（独立执行；不包含代码仓库）

该代码块只同步 raw、official SFT collage、显式 metadata/gate/evidence，不使用 `--delete`；它可在
Git 主方案或 patch fallback 完成后独立执行。

```bash
set -euo pipefail

export SFT_MIG_TARGET_HOST='user@target-server'
export SFT_MIG_SOURCE_TASK_ROOT=/data/jiahao/task
export SFT_MIG_TARGET_TASK_ROOT=/data/jiahao/task
export SFT_MIG_SOURCE_BRICKNET="${SFT_MIG_SOURCE_TASK_ROOT}/BrickNet"
export SFT_MIG_SOURCE_LLAMAF="${SFT_MIG_SOURCE_TASK_ROOT}/LlamaFactory"
export SFT_MIG_TARGET_BRICKNET="${SFT_MIG_TARGET_TASK_ROOT}/BrickNet"
export SFT_MIG_TARGET_LLAMAF="${SFT_MIG_TARGET_TASK_ROOT}/LlamaFactory"
export SFT_MIG_SOURCE_RAW="${SFT_MIG_SOURCE_BRICKNET}/outputs_gt/sft_8view_renders_v2_rowids"
export SFT_MIG_SOURCE_COLLAGE="${SFT_MIG_SOURCE_BRICKNET}/outputs_preprocess/BrickNet-MM/image_v2_official/SFT"
export SFT_MIG_SOURCE_META="${SFT_MIG_SOURCE_BRICKNET}/outputs_gt/.bricknet_render_v2_official"
export SFT_MIG_TARGET_RAW="${SFT_MIG_TARGET_BRICKNET}/outputs_gt/sft_8view_renders_v2_rowids"
export SFT_MIG_TARGET_COLLAGE="${SFT_MIG_TARGET_BRICKNET}/outputs_preprocess/BrickNet-MM/image_v2_official/SFT"
export SFT_MIG_TARGET_META="${SFT_MIG_TARGET_BRICKNET}/outputs_gt/.bricknet_render_v2_official"
export SFT_MIG_SOURCE_EVIDENCE="${SFT_MIG_SOURCE_LLAMAF}/tmp_bash"
export SFT_MIG_TARGET_EVIDENCE="${SFT_MIG_TARGET_LLAMAF}/tmp_bash"

ssh "${SFT_MIG_TARGET_HOST}" "mkdir -p \
  '${SFT_MIG_TARGET_RAW}' \
  '${SFT_MIG_TARGET_COLLAGE}' \
  '${SFT_MIG_TARGET_META}' \
  '${SFT_MIG_TARGET_EVIDENCE}'"

# Formal raw views and collages. No --delete: pre-existing target files are not removed.
rsync -aH --partial --info=progress2 \
  "${SFT_MIG_SOURCE_RAW}/" \
  "${SFT_MIG_TARGET_HOST}:${SFT_MIG_TARGET_RAW}/"
rsync -aH --partial --info=progress2 \
  "${SFT_MIG_SOURCE_COLLAGE}/" \
  "${SFT_MIG_TARGET_HOST}:${SFT_MIG_TARGET_COLLAGE}/"

# Metadata is explicit. sft.lock is intentionally excluded; the validated gate is mandatory for migration.
for SFT_METADATA_NAME in sft_identity.json sft_progress.json sft_summary.json sft_timings.jsonl; do
  test -f "${SFT_MIG_SOURCE_META}/${SFT_METADATA_NAME}"
  rsync -a --partial --info=progress2 \
    "${SFT_MIG_SOURCE_META}/${SFT_METADATA_NAME}" \
    "${SFT_MIG_TARGET_HOST}:${SFT_MIG_TARGET_META}/${SFT_METADATA_NAME}"
done
test -s "${SFT_MIG_SOURCE_META}/sft_completion_gate.json"
rsync -a --partial --info=progress2 \
  "${SFT_MIG_SOURCE_META}/sft_completion_gate.json" \
  "${SFT_MIG_TARGET_HOST}:${SFT_MIG_TARGET_META}/sft_completion_gate.json"

# Evidence is explicit; run after strict verify has stopped writing its log.
for SFT_EVIDENCE_NAME in \
  supervise_official_sft_render.status \
  supervise_official_sft_render-20260902T194605Z-381805.log \
  supervise_official_sft_render-20260902T194605Z-381805.gpu.tsv \
  supervise_official_sft_render-20260902T194605Z-381805.gpu-peaks.tsv \
  sft_official_verify_20260906.log; do
  test -f "${SFT_MIG_SOURCE_EVIDENCE}/${SFT_EVIDENCE_NAME}"
  rsync -a --partial --info=progress2 \
    "${SFT_MIG_SOURCE_EVIDENCE}/${SFT_EVIDENCE_NAME}" \
    "${SFT_MIG_TARGET_HOST}:${SFT_MIG_TARGET_EVIDENCE}/${SFT_EVIDENCE_NAME}"
done
# Optional GPU error evidence; an empty .gpu.error is valid and means no recorded GPU error.
if [[ -f "${SFT_MIG_SOURCE_EVIDENCE}/supervise_official_sft_render-20260902T194605Z-381805.gpu.error" ]]; then
  rsync -a --partial --info=progress2 \
    "${SFT_MIG_SOURCE_EVIDENCE}/supervise_official_sft_render-20260902T194605Z-381805.gpu.error" \
    "${SFT_MIG_TARGET_HOST}:${SFT_MIG_TARGET_EVIDENCE}/supervise_official_sft_render-20260902T194605Z-381805.gpu.error"
fi
```

如果目标服务器没有同样的 `/home/jiahao/task -> /data/jiahao/task` symlink，媒体副本仍须保留源
metadata 作为不可变 provenance；目标路径不同只用 rsync checksum/结构 gate 验证同步，禁止手工改写
已完成 run 的 `sft_identity.json`、`sft_summary.json` 或 gate，也不能通过改路径冒充同一 run。
`/data/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM/images/SFT`、
`/data/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM/sharegpt/BrickNet-MM_SFT.json`、
`/data/jiahao/task/BrickNet/data/bricknet_datasets/sft.npz` 和
`/home/jiahao/.local/share/bricknet/glb` 仅在要继续复现/重新验证时，按 `sft_identity.json` 中的 hash、
数量和版本显式补齐，不属于本次默认同步。对应 BrickNet-Render commit、renderer 环境、GPU 和
dataset/token-cache 配置也只在该场景下按 identity 复核；若要在目标机继续 render 或运行
identity-bound strict verify，必须保持相同 canonical layout，或新建目标机专用 config 并走明确的
refreeze/new identity 流程。以上数据 rsync 不传播源端删除；目标端若要清理旧文件，必须先人工审阅
差异后单独执行。

### rsync dry-run 和目标端计数

先做不读内容的快速 dry-run，再按需用 `-c` 做全量 checksum dry-run；两种命令都不使用 `--delete`。
正常情况下应没有待传输项。raw views 的 `-c` 会读取约 113G，运行时间可能较长。

```bash
set -euo pipefail
export SFT_MIG_TARGET_HOST='user@target-server'
export SFT_MIG_SOURCE_TASK_ROOT=/data/jiahao/task
export SFT_MIG_TARGET_TASK_ROOT=/data/jiahao/task
export SFT_MIG_SOURCE_BRICKNET="${SFT_MIG_SOURCE_TASK_ROOT}/BrickNet"
export SFT_MIG_TARGET_BRICKNET="${SFT_MIG_TARGET_TASK_ROOT}/BrickNet"
export SFT_MIG_SOURCE_RAW="${SFT_MIG_SOURCE_BRICKNET}/outputs_gt/sft_8view_renders_v2_rowids"
export SFT_MIG_SOURCE_COLLAGE="${SFT_MIG_SOURCE_BRICKNET}/outputs_preprocess/BrickNet-MM/image_v2_official/SFT"
export SFT_MIG_SOURCE_META="${SFT_MIG_SOURCE_BRICKNET}/outputs_gt/.bricknet_render_v2_official"
export SFT_MIG_TARGET_RAW="${SFT_MIG_TARGET_BRICKNET}/outputs_gt/sft_8view_renders_v2_rowids"
export SFT_MIG_TARGET_COLLAGE="${SFT_MIG_TARGET_BRICKNET}/outputs_preprocess/BrickNet-MM/image_v2_official/SFT"
export SFT_MIG_TARGET_META="${SFT_MIG_TARGET_BRICKNET}/outputs_gt/.bricknet_render_v2_official"

# 快速结构/size/mtime dry-run。
rsync -aHn --itemize-changes --out-format='%i %n%L' \
  "${SFT_MIG_SOURCE_RAW}/" "${SFT_MIG_TARGET_HOST}:${SFT_MIG_TARGET_RAW}/"
rsync -aHn --itemize-changes --out-format='%i %n%L' \
  "${SFT_MIG_SOURCE_COLLAGE}/" "${SFT_MIG_TARGET_HOST}:${SFT_MIG_TARGET_COLLAGE}/"

# 内容 checksum dry-run（需要完整读取文件，但仍不写入、不删除）。
rsync -aHnc --itemize-changes --out-format='%i %n%L' \
  "${SFT_MIG_SOURCE_RAW}/" "${SFT_MIG_TARGET_HOST}:${SFT_MIG_TARGET_RAW}/"
rsync -aHnc --itemize-changes --out-format='%i %n%L' \
  "${SFT_MIG_SOURCE_COLLAGE}/" "${SFT_MIG_TARGET_HOST}:${SFT_MIG_TARGET_COLLAGE}/"

for SFT_METADATA_NAME in sft_identity.json sft_progress.json sft_summary.json sft_timings.jsonl; do
  rsync -nc --itemize-changes --out-format='%i %n%L' \
    "${SFT_MIG_SOURCE_META}/${SFT_METADATA_NAME}" \
    "${SFT_MIG_TARGET_HOST}:${SFT_MIG_TARGET_META}/${SFT_METADATA_NAME}"
done
test -s "${SFT_MIG_SOURCE_META}/sft_completion_gate.json"
rsync -nc --itemize-changes --out-format='%i %n%L' \
  "${SFT_MIG_SOURCE_META}/sft_completion_gate.json" \
  "${SFT_MIG_TARGET_HOST}:${SFT_MIG_TARGET_META}/sft_completion_gate.json"
```

目标端计数和命名检查（只读；应输出 `status=PASS`）如下。脚本会 fail-closed：raw/collage root
本身必须是非 symlink 目录；raw root 的任何非数字目录、非目录项或 symlink，row 目录中的额外/缺失/
非文件/symlink 项，以及 collage root 的任何非文件、symlink 或非 `<numeric>.png` 项都会令检查失败，
不会被静默忽略。

```bash
set -euo pipefail
export SFT_MIG_TARGET_HOST='user@target-server'
export SFT_MIG_TARGET_TASK_ROOT=/data/jiahao/task
export SFT_MIG_TARGET_BRICKNET="${SFT_MIG_TARGET_TASK_ROOT}/BrickNet"
export SFT_MIG_TARGET_RAW="${SFT_MIG_TARGET_BRICKNET}/outputs_gt/sft_8view_renders_v2_rowids"
export SFT_MIG_TARGET_COLLAGE="${SFT_MIG_TARGET_BRICKNET}/outputs_preprocess/BrickNet-MM/image_v2_official/SFT"

ssh "${SFT_MIG_TARGET_HOST}" bash -s -- \
  "${SFT_MIG_TARGET_RAW}" "${SFT_MIG_TARGET_COLLAGE}" <<'PY'
set -euo pipefail
raw_root=$1
collage_root=$2
python - "$raw_root" "$collage_root" <<'PYTHON'
from pathlib import Path
import re, sys

raw = Path(sys.argv[1])
collage = Path(sys.argv[2])
if raw.is_symlink() or not raw.is_dir():
    raise SystemExit(f"raw root must be an existing non-symlink directory: {raw}")
if collage.is_symlink() or not collage.is_dir():
    raise SystemExit(f"collage root must be an existing non-symlink directory: {collage}")
raw_entries = list(raw.iterdir())
bad_raw_root_entries = [
    p for p in raw_entries if p.is_symlink() or not (p.is_dir() and re.fullmatch(r"\d+", p.name))
]
raw_dirs = [
    p for p in raw_entries if not p.is_symlink() and p.is_dir() and re.fullmatch(r"\d+", p.name)
]
raw_pngs = []
bad_rows = []
row_ids = set()
for row in raw_dirs:
    rid = int(row.name)
    row_ids.add(rid)
    entries = list(row.iterdir())
    names = {p.name for p in entries}
    expected = {f"{rid}_{view:04d}.png" for view in range(8)}
    if (
        names != expected
        or len(entries) != 8
        or any(p.is_symlink() or not p.is_file() or p.name not in expected for p in entries)
    ):
        bad_rows.append((rid, sorted(names - expected), sorted(expected - names)))
    raw_pngs.extend(p for p in entries if not p.is_symlink() and p.is_file() and p.name.endswith(".png"))
collage_entries = list(collage.iterdir())
bad_collage_entries = [
    p for p in collage_entries if p.is_symlink() or not (p.is_file() and re.fullmatch(r"\d+\.png", p.name))
]
collages = {
    p.name for p in collage_entries
    if not p.is_symlink() and p.is_file() and re.fullmatch(r"\d+\.png", p.name)
}
expected_collages = {f"{rid}.png" for rid in row_ids}
ok = (
    len(row_ids) == 67178
    and len(raw_pngs) == 537424
    and not bad_raw_root_entries
    and not bad_rows
    and not bad_collage_entries
    and collages == expected_collages
    and len(collages) == 67178
)
print(
    f"raw_rows={len(row_ids)} raw_pngs={len(raw_pngs)} "
    f"collages={len(collages)} bad_raw_root_entries={len(bad_raw_root_entries)} "
    f"bad_rows={len(bad_rows)} bad_collage_entries={len(bad_collage_entries)}"
)
print("status=PASS" if ok else "status=FAIL")
if not ok:
    raise SystemExit(1)
PYTHON
PY
```

目标端还要检查 metadata 文件和已验证 gate 均存在且 gate 为 PASS；不要把 `sft.lock` 计入
metadata 完整性：

```bash
set -euo pipefail
export SFT_MIG_TARGET_HOST='user@target-server'
export SFT_MIG_TARGET_TASK_ROOT=/data/jiahao/task
export SFT_MIG_TARGET_BRICKNET="${SFT_MIG_TARGET_TASK_ROOT}/BrickNet"
export SFT_MIG_TARGET_META="${SFT_MIG_TARGET_BRICKNET}/outputs_gt/.bricknet_render_v2_official"

ssh "${SFT_MIG_TARGET_HOST}" bash -s -- "${SFT_MIG_TARGET_META}" <<'PY'
set -euo pipefail
meta_root=$1
for name in sft_identity.json sft_progress.json sft_summary.json sft_timings.jsonl; do
  test -s "${meta_root}/${name}"
done
test -s "${meta_root}/sft_completion_gate.json"
jq -e '
  .status == "PASS" and
  .scope.split == "SFT" and
  .scope.out_of_scope == ["PT", "VAL"] and
  (.metadata_files | has("config") and has("identity") and has("progress") and has("summary") and has("timings")) and
  .strict_verify.status == "PASS" and
  .supervisor_provenance.status_values.stage == "COMPLETE" and
  .supervisor_provenance.status_values.result == "OK"
' \
  "${meta_root}/sft_completion_gate.json"
echo 'sft.lock: intentionally excluded from migration identity'
PY
```

同步后若要在目标端继续复现渲染，还需重新检查目标的 renderer、GLB inventory、config SHA-256、
launcher/source SHA-256、GPU 可见性、实际 workers/GPU、dataset registry 和 token cache；这些
配置/gate/代码任一不同步，都只能把目标端当作媒体查看副本，不能宣称可复现或可继续 downstream。
