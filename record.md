
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

## PT-exp2 & exp4_4~exp4_6

固定序列为 `PT-exp2-text8m → PT-exp2-mm-e1/e2/e3 → PT-exp2 alias → exp4_4 10k → exp4_5 50k → exp4_6 all`。
不创建 PT-exp2 VAL511 训练或验证。详细数据 hash、配置与 gate 见 [PT-exp2 runbook](bricknet-pt-exp2.md)。

### PT-exp2

MM consolidation 使用三个顺序训练配置：e1 从 text8m adapter 开始，e2 从 e1 final adapter 继续，e3 从 e2
final adapter 继续。三轮分别使用不重叠的 replay slice（`15,617/15,586/15,667` 条），target tokens 为
`26,285,287/26,285,922/26,284,707`，replay/MM ratio 为
`1.0000053/1.0000294/0.9999832`。每次训练 1 epoch、每卡 BS2，单/双卡 GA=`8/4`（global batch 16）；
e1 LR=`2e-5`，e2/e3 LR=`1e-5`。

- text8m final：`saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp2_text8m_qwen35_08b_path7698261_steps250k_bs4_gbs32_lora64_len6401_nopack`
- MM e1：`saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp2_mm_e1_qwen35_08b_text8m_mm135k_replay1to1_ep1_bs2_gbs16_lora64_len6400`
- MM e2：`saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp2_mm_e2_qwen35_08b_text8m_mm135k_replay1to1_ep1_bs2_gbs16_lora64_len6400`
- MM e3：`saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp2_mm_e3_qwen35_08b_text8m_mm135k_replay1to1_ep1_bs2_gbs16_lora64_len6400`
- final alias：`saves/Qwen3.5-0.8B-Thinking/lora/PT-exp2`
- 下游版本：`exp4_4=10k`、`exp4_5=50k`、`exp4_6=all 66,456`

下面是从 text8m 训练到 all-66,456 下游评测的完整手动执行序列。命令已经预填 action、run、approval 和工作
目录；按顺序逐条运行即可。所有 `train` 命令通过 `--gpus 0 1` 同时使用本机两张 RTX PRO 6000，launcher
显式注入 `FORCE_TORCHRUN=1`、`NPROC_PER_NODE=2`、`NNODES=1`，由 DDP 每卡启动一个进程。launcher 根据
`--gpus` 数量动态覆盖 GA：单卡 text/MM/downstream 为 `8/8/16`，双卡为 `4/4/8`，因此有效 global batch
始终为 32/16/16，训练 steps、epochs 和 LR schedule 均不变。输出目录使用 `gbs32/gbs16`，不再绑定某个
world size；`ddp_find_unused_parameters=false` 已显式冻结。
后续恢复计划使用带 `int32 length` 的 text8m `tokenized_path`，grouped sampler 直接读取长度列，不再在每次启动时
扫描 7,698,261 条 `input_ids`。但 2026-08-13 复查发现当前同名目录被普通流程重新生成，实际缺少 `length`；
launcher 现会阻断该伪 with-length cache，需迁移到新的输出目录并更新 YAML。通用 PT/SFT 工具为
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

预测与评测仍用单卡。`select-final`、
`approve-scale` 是有意保留的人工决策点，运行前应先阅读上一条评测产生的 metrics：

如只使用 GPU 0，把任一训练命令的 `--gpus 0 1` 改为 `--gpus 0` 即可；launcher 会自动恢复单卡 GA，
无需修改 YAML。例如 text8m 单卡 dry-run：

```bash
# 以单 GPU 检查 PT-exp2 text8m 训练 gate 和动态 batch 配置，不启动训练。
python scripts/launch_bricknet_pt_exp2.py --gpus 0 --action train --run text8m
```

```bash
# 进入保存 PT-exp2 配置与 launcher 的 LlamaFactory 仓库。
cd /data/jiahao/task/LlamaFactory

# 使用两张 GPU 正式训练 PT-exp2 text8m 阶段。
python scripts/launch_bricknet_pt_exp2.py --gpus 0 1 --action train --run text8m --execute
# 从 text8m adapter 继续正式训练 MM e1。
python scripts/launch_bricknet_pt_exp2.py --gpus 0 1 --action train --run mm-e1 --execute
# 使用 MM e1 adapter 运行验证集推理。
python scripts/launch_bricknet_pt_exp2.py --gpus 0 --action predict --run mm-e1 --execute
# 评测 MM e1 的推理结果。
python scripts/launch_bricknet_pt_exp2.py --gpus 0 --action evaluate --run mm-e1 --execute
# 从 MM e1 adapter 继续正式训练 MM e2。
python scripts/launch_bricknet_pt_exp2.py --gpus 0 1 --action train --run mm-e2 --execute
# 使用 MM e2 adapter 运行验证集推理。
python scripts/launch_bricknet_pt_exp2.py --gpus 0 --action predict --run mm-e2 --execute
# 评测 MM e2 的推理结果。
python scripts/launch_bricknet_pt_exp2.py --gpus 0 --action evaluate --run mm-e2 --execute
# 从 MM e2 adapter 继续正式训练 MM e3。
python scripts/launch_bricknet_pt_exp2.py --gpus 0 1 --action train --run mm-e3 --execute
# 使用 MM e3 adapter 运行验证集推理。
python scripts/launch_bricknet_pt_exp2.py --gpus 0 --action predict --run mm-e3 --execute
# 评测 MM e3 的推理结果。
python scripts/launch_bricknet_pt_exp2.py --gpus 0 --action evaluate --run mm-e3 --execute
# 人工确认 e3 结果后，将其选为 PT-exp2 final alias。
python scripts/launch_bricknet_pt_exp2.py --action select-final --run mm-e3 --execute --approve
```

用户已冻结官方 first-round + cross-pool exact-dedup 的 7,698,261 条为本机规范，8,092,423 仅保留为发布
provenance。`finalize-existing` 已重扫全部 38,485,631 个源行并逐行比对 31 shard，ordered corpus
SHA-256=`985b8473...07d0ab6`；未重写 34 GiB 数据。官方式 seed-0 first-round VAL1000 已生成。text PT 配置为
non-packing、`cutoff_len=6401`、250k steps、global batch 32。

全量 7,698,261 条 parse 已通过。确定性 10k collision replay 的 94/10,000 命中仍完整记录，前 20 个明细中
18 个来自 PT 首轮、2 个来自 SFT 首轮，含 4 个完整 component。论文与官方 sampler 说明发布 path 在采样阶段
做 collision filtering，官方 `train.py` 在训练加载阶段不再次 parse/collision 筛除。依用户决策，当前
`collision_findings_block_training=false`、`audit.eligible=true`，不删除样本；31-shard `text8m_train` 视图已创建。
当前 launcher 会报告所选 GPU 上的进程，但按此前临时决策不以 GPU 占用作为 blocker；执行前必须人工确认
GPU 0/1 均无其他用户任务。尚未启动任何 PT-exp2 训练。

### exp4_4
```bash
# 从 PT-exp2 final 正式训练 exp4_4 10k。
python scripts/launch_bricknet_pt_exp2.py --gpus 0 1 --action train --run exp4_4 --execute
# 使用 exp4_4 adapter 运行 VAL512 推理。
python scripts/launch_bricknet_pt_exp2.py --gpus 0 --action predict --run exp4_4 --execute
# 统一评测 exp4_4 的 VAL512 结果。
python scripts/launch_bricknet_pt_exp2.py --gpus 0 --action evaluate --run exp4_4 --execute
# 人工确认 exp4_4 收益并批准扩展到 50k。
python scripts/launch_bricknet_pt_exp2.py --action approve-scale --run exp4_4 --execute --approve
# 按冻结 manifest 物化 exp4_5 50k 数据。
python scripts/launch_bricknet_pt_exp2.py --action materialize --run exp4_5 --execute
```

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

### PT-exp2-100k 外部权重（exp4_4_1 / exp4_7_1 的前置）

用户已在另一台服务器用 2× RTX PRO 6000 训练 100,000 steps 的 PT-exp2 权重。本机尚未同步，规范路径固定为
`saves/Qwen3.5-0.8B-Thinking/lora/PT-exp2-100k`。在运行 exp4_4_1 / exp4_7_1 之前，先把该 adapter 拷贝到
该目录，并确认包含 `adapter_config.json` 与 `adapter_model.safetensors`：

```bash
cd /home/jiahao/task/LlamaFactory
# 示例：从远程主机同步（按实际地址替换 <remote> 与 <远程目录>）。
mkdir -p saves/Qwen3.5-0.8B-Thinking/lora/PT-exp2-100k
rsync -av <remote>:<远程目录>/ saves/Qwen3.5-0.8B-Thinking/lora/PT-exp2-100k/
sha256sum saves/Qwen3.5-0.8B-Thinking/lora/PT-exp2-100k/adapter_model.safetensors
ls saves/Qwen3.5-0.8B-Thinking/lora/PT-exp2-100k/adapter_config.json \
   saves/Qwen3.5-0.8B-Thinking/lora/PT-exp2-100k/adapter_model.safetensors
```

该 100k 权重与 `PT-exp2-text8m(250k) → mm-e1/e2/e3 → PT-exp2` 主链相互独立；exp4_4_1 / exp4_7_1
直接从 `PT-exp2-100k` 初始化，不等待 text8m/MM/e1-e3 或 final alias。

### exp4_4_1 — PT-exp2-100k + NonThinking-Control 10k

配置：train=`examples/train_lora/qwen35_08b_bricknet_stage2_exp4_4_1_nonthinking_control_10k_pt_exp2_100k.yaml`，
predict=`examples/train_lora/qwen35_08b_bricknet_stage2_exp4_4_1_nonthinking_control_predict_pt_exp2_100k.yaml`。
与 exp4_4 相同的 LoRA/LR/epoch/batch/长度协议，仅初始化换成 `PT-exp2-100k`。

```bash
cd /home/jiahao/task/LlamaFactory

# 单卡训练（GA 在 CLI 覆盖为 16，global batch 16）。
conda run -n llamafactory --no-capture-output llamafactory-cli train \
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

### exp4_7_1 — PT-exp2-100k + Thinking-Hard 10k

配置：train=`examples/train_lora/qwen35_08b_bricknet_stage2_exp4_7_1_thinking_hard_10k_pt_exp2_100k.yaml`，
predict=`examples/train_lora/qwen35_08b_bricknet_stage2_exp4_7_1_thinking_hard_predict_pt_exp2_100k.yaml`。
与 exp4_3 相同的 Thinking-Hard 数据与训练协议，仅初始化换成 `PT-exp2-100k`；评测自动走 strict trace 提取。

```bash
cd /home/jiahao/task/LlamaFactory

# 单卡训练。
conda run -n llamafactory --no-capture-output llamafactory-cli train \
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
仍可直接执行或用于同协议复跑。截至 2026-08-15，Stage5 full replay 已通过，B1/V1/V2/A0/A1 的 VAL512
推理、final/raw 双层统一评测和 seed-42、10,000 次 paired bootstrap 均已完成；冻结结果和指标解释见
`/home/jiahao/task/BrickNet/BrickNet-MM Agentic LEGO Planner/Stage 6-7 Agentic Evaluation.md`。

> **2026-08-16 协议异常修复后必须重跑。** 上面 2026-08-15 的五组 controller artifact 由旧 HF backend
> 生成：它用 Qwen 原生 `apply_chat_template(enable_thinking=False)` 在 assistant 提示符后注入空 `<think>` 块、
> 且未把模型默认 EOS(`<|endoftext|>`) 覆盖为 `<|im_end|>`，与 exp4_2 冻结的 LlamaFactory
> `qwen3_5_nothink` 协议不等价；B1 还叠加了末尾空行 framing 拒绝。修复已合入
> `BrickNet/scripts/run_bricknet_agentic_inference.py`（字节级 `qwen3_5_nothink` 渲染、EOS/pad 覆盖、图像像素预算、
> 统一空白规范化）。2026-08-17 起随机种子也改为 exp4_2 方案：每个进程只设一次 `set_seed(42)`，已删除
> `fork_rng` 和逐样本/逐调用的种子重新对齐，五个模式共用同一条连续随机流；下面五条命令**不需要修改**。
> GPU 空闲后按原顺序重新执行即可原子覆盖旧的异常输出目录；
> 随后按 Eval 小节重跑 `--action all --execute`（必要时 `--force`）重建 manifest/bootstrap。旧
> `agentic_exp4_2_stage67_{manifest,statistics,results}` 仅作历史，重跑前不得引用其中的五组指标。

```bash
# 进入 BrickNet 仓库，后续运行 Stage5–7 环境和 controller。
cd /home/jiahao/task/BrickNet

# 对 66,456 条 reference 运行 Stage5 真实 mesh 全量事务 replay，并冻结 gate report。
BRICKNET_DATA=/home/jiahao/task/BrickNet/data/bricknet_datasets \
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  scripts/audit_bricknet_assembly_env.py \
  --input /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-RL/datasets/BrickNet-MM-RL.jsonl \
  --output /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/stage5/Stage5-full-replay-report.json \
  --expected-count 66456 --workers 8 --chunksize 8 --progress-every 100
```

### B1
```bash
# 运行 B1：exp4_2 完整 path 一次生成，只做 post-hoc 验证。
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python scripts/run_bricknet_agentic_inference.py \
  --input /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Reasoning/validation/datasets/BrickNet-Stage2-NonThinking-Control-VAL512-Eval.jsonl \
  --output /home/jiahao/task/BrickNet/outputs_val/qwen35_08b/agentic_exp4_2_b1/controller_audit.jsonl \
  --mode b1-post-hoc --backend hf --prompt-protocol exp4_2-stepwise --seed 42 \
  --stage5-report /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/stage5/Stage5-full-replay-report.json
```

### V1
```bash
# 运行 V1：逐 placement 静默拒绝与重试，模型看不到 verifier 原因。
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python scripts/run_bricknet_agentic_inference.py \
  --input /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Reasoning/validation/datasets/BrickNet-Stage2-NonThinking-Control-VAL512-Eval.jsonl \
  --output /home/jiahao/task/BrickNet/outputs_val/qwen35_08b/agentic_exp4_2_v1/controller_audit.jsonl \
  --mode v1-silent-retry --backend hf --prompt-protocol exp4_2-stepwise --seed 42 \
  --stage5-report /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/stage5/Stage5-full-replay-report.json
```

### V2
```bash
# 运行 V2：逐 placement 静默 verifier DFS 与有限深度 rollback。
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python scripts/run_bricknet_agentic_inference.py \
  --input /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Reasoning/validation/datasets/BrickNet-Stage2-NonThinking-Control-VAL512-Eval.jsonl \
  --output /home/jiahao/task/BrickNet/outputs_val/qwen35_08b/agentic_exp4_2_v2/controller_audit.jsonl \
  --mode v2-silent-dfs --backend hf --prompt-protocol exp4_2-stepwise --seed 42 \
  --candidates-per-round 8 --max-rounds-per-state 4 --max-backtrack-depth 3 \
  --stage5-report /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/stage5/Stage5-full-replay-report.json
```

### A0
```bash
# 运行 A0：向模型显式反馈五字段 rejection observation，不进行分支搜索。
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python scripts/run_bricknet_agentic_inference.py \
  --input /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Reasoning/validation/datasets/BrickNet-Stage2-NonThinking-Control-VAL512-Eval.jsonl \
  --output /home/jiahao/task/BrickNet/outputs_val/qwen35_08b/agentic_exp4_2_a0/controller_audit.jsonl \
  --mode a0-act-feedback --backend hf --prompt-protocol stage8-act --seed 42 \
  --stage5-report /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/stage5/Stage5-full-replay-report.json
```

### A1
```bash
# 运行 A1：显式反馈加 snapshot/rollback 分支搜索。
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python scripts/run_bricknet_agentic_inference.py \
  --input /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Reasoning/validation/datasets/BrickNet-Stage2-NonThinking-Control-VAL512-Eval.jsonl \
  --output /home/jiahao/task/BrickNet/outputs_val/qwen35_08b/agentic_exp4_2_a1/controller_audit.jsonl \
  --mode a1-feedback-search --backend hf --prompt-protocol stage8-act --seed 42 \
  --candidates-per-round 8 --max-rounds-per-state 4 --max-backtrack-depth 3 \
  --stage5-report /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/stage5/Stage5-full-replay-report.json
```

每个 controller 命令自动同时生成 `<stem>.predictions.jsonl` 与
`<stem>.raw_first_choice_predictions.jsonl`。

### Eval
下面的统一入口会校验 512 条 ID/order、HF provenance、generation error
和文件 hash，分别评测 final/raw 两层，生成 hash-frozen manifest，再运行 seed-42、10,000 次 paired bootstrap。

```bash
# 进入 BrickNet；后续所有 Stage6–7 评测命令均从这里执行。
cd /home/jiahao/task/BrickNet

# 只做 fail-closed 预检，不重跑任何评测；五组 controller artifact 必须均为 512 条真实 HF 输出。
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  scripts/evaluate_bricknet_agentic_stage67.py --action preflight

# 打印完整执行计划但不运行，便于先检查解释器、输入、输出和外部评测命令。
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  scripts/evaluate_bricknet_agentic_stage67.py --action all

# 一键生成 B1/V1/V2/A0/A1 的 final/raw 全指标、冻结 manifest、bootstrap 统计和 Markdown 结果表。
BRICKNET_DATA=/home/jiahao/task/BrickNet/data/bricknet_datasets \
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  scripts/evaluate_bricknet_agentic_stage67.py --action all --execute

# 可选：仅补跑 V2 final 层；已有同 input hash 的派生产物会复用，确需重建时追加 --force。
BRICKNET_DATA=/home/jiahao/task/BrickNet/data/bricknet_datasets \
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  scripts/evaluate_bricknet_agentic_stage67.py --action evaluate --runs v2 --layers final --execute

# 可选：评测齐全后只重新冻结 manifest，适合确认输入 hash 没有漂移。
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  scripts/evaluate_bricknet_agentic_stage67.py --action manifest --execute

# 可选：基于冻结 manifest 只重跑 paired bootstrap 与 experiment_results 风格 Markdown 报告。
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  scripts/evaluate_bricknet_agentic_stage67.py --action summarize --execute
```

统一配置位于 `/home/jiahao/task/BrickNet/configs/agentic_stage67_exp4_2.json`；最终产物为
`agentic_exp4_2_stage67_manifest.json`、`agentic_exp4_2_stage67_statistics.json` 和
`agentic_exp4_2_stage67_results.md`。主选择指标是 evaluator 的 pose-aware `task_strict_success`，次指标为
`dense_reward`，最后比较 token/latency/verifier-call 成本；`controller_hard_valid_success` 只证明系统输出满足
硬约束，不能代替 task strict success。

### Experiments Results
2026-08-15 结论（V2 final strict=`20/512 (3.906%)`、dense=`0.567096` 最佳；A0/A1 显著劣于 V1/V2 等）
已因上述 `<think>`/EOS/framing 协议异常而**暂缓**，重跑前不得作为相对 exp4_2 的证据。修复协议重跑五组
controller 并重建 manifest/bootstrap 后，再按 task strict → dense → 成本的顺序重新下结论；若届时 A0/A1
仍显著退化，下一步应先训练 R1-S 10k 交互协议 cold start，而不是把未训练的显式 feedback 设为默认。


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
# 检查 R1-S 使用 matched A0 的 VAL512 controller 评测命令。
python scripts/launch_bricknet_stage8_controller_eval.py --run R1-S
# 正式执行 R1-S matched A0 VAL512 controller 评测。
python scripts/launch_bricknet_stage8_controller_eval.py --run R1-S --execute
```

### R1-C
R1-S matched A0 gate 获批后，按以下顺序收集并构造 R1-C；collector 在每个 GT prefix 只采一个真实 proposal，
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
# 检查 R1-C 使用 matched A0 的 VAL512 controller 评测命令。
python scripts/launch_bricknet_stage8_controller_eval.py --run R1-C
# 正式执行 R1-C matched A0 VAL512 controller 评测。
python scripts/launch_bricknet_stage8_controller_eval.py --run R1-C --execute
```

### R1-B
R1-S 的 matched A0 VAL512 未改善前停止。R1-C 训练完成并通过 matched A0 gate 后，才检查 A1 日志是否达到
1,000 rollback transitions/100 sources 以构造 R1-B。三个 launcher 均会从 exp4_2 新建 LoRA，并拒绝把
R1-S/R1-C 串行当作下一实验初始化。完成真实 A1 日志转换、R1-B 70/20/10 token-mix 和两遍切窗审计后，入口为：

```bash
# 进入 LlamaFactory；仅在 R1-B rollback 数据和两遍审计 gate 已通过后继续。
cd /home/jiahao/task/LlamaFactory
# 刷新并检查 R1-B 初始化隔离、70/20/10 token mix 和训练 gate。
python scripts/launch_bricknet_stage8_act_sft.py --run R1-B --scale 10k --refresh-initialization-audit
# 所有条件满足后从 exp4_2 独立初始化并正式训练 R1-B。
python scripts/launch_bricknet_stage8_act_sft.py --run R1-B --scale 10k --execute
# 正式执行 R1-B matched A1 VAL512 controller 评测。
python scripts/launch_bricknet_stage8_controller_eval.py --run R1-B --execute
```

这些入口在 rejection/rollback 数据、80/20 或 70/20/10 supervised-token mix、matched max_steps、Stage5 report、
processor、initialization 和 dataset hash 任一条件缺失时都会退出，不会静默训练。
