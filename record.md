
# notes

- 统一实验进度与结果：[experiment_results.md](experiment_results.md)。
- 唯一总推进计划：`/home/jiahao/task/BrickNet/BrickNet-MM Agentic LEGO Planner/Constructor Plan.md`。
- 执行约定：以下命令均从 `/home/jiahao/task/LlamaFactory` 目录启动；LlamaFactory 训练/预测命令显式通过
  `conda run -n llamafactory --no-capture-output` 使用项目环境，不依赖当前 shell 的 Python/Transformers 版本。
- 已完成实验的 train/predict 行是可复现实验命令；安全 launcher 在对应输出目录已存在时会按设计拒绝覆盖。
  评测命令可直接执行，其中 Stage-2 统一 evaluator 会复用完整结果，显式传入 `--force` 才会重算。
- 2026-08-05 决策：66,456 条基础处理池不再四分；统一使用 512 VAL；policy-specific full hard mining 暂停，
  条件性恢复时最多处理 2,000 prompts。

    Qwen3-VL-2B-Instruct
    Qwen3.5-0.8B
    Qwen3.5-2B
    hf download Qwen/Qwen3.5-2B-Base



    历史 tokenized cache：.llamafactory_cache/tokenized_dataset/BrickNet-MM-SFT_qwen35-08b_nothink_len4096_img589824

    hf download Qwen/Qwen3-VL-4B-Instruct
    hf download Qwen/Qwen3-VL-8B-Instruct
    hf download Qwen/Qwen3-VL-32B-Instruct
    hf download Qwen/Qwen3.5-4B-Base
    hf download Qwen/Qwen3.5-4B
    hf download Qwen/Qwen3.5-9B
    hf download Qwen/Qwen3.5-27B
    hf download Qwen/Qwen3.6-27B


# PT experiments
## exp0
### Qwen3.5-0.8B
**train**
训练输出目录：`saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64`
**eval**
1. `conda run -n llamafactory --no-capture-output llamafactory-cli train examples/train_lora/qwen35_08b_bricknet_mm_pt_predict.yaml`
2. `cd ../BrickNet && /home/jiahao/miniconda3/envs/bricknet/bin/python scripts/evaluate_experiment.py --predictions ../LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20/generated_predictions.jsonl --text-metrics ../LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20/predict_results.json --output-dir outputs_val/qwen35_08b/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20 --prompts-file data/bricknet_datasets/captions_val.jsonl`
3. `jq -c '{response: .predict, label: .label}' saves/Qwen3.5-0.8B-Thinking/lora/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20/generated_predictions.jsonl > ../BrickNet/outputs_val/qwen35_08b/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20/alignment_input.jsonl`
4. `cd ../ms-swift && /home/jiahao/miniconda3/envs/bricknet/bin/python examples/train/grpo/plugin/bricknet/evaluate_experiment.py alignment-worker --results ../BrickNet/outputs_val/qwen35_08b/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20/alignment_input.jsonl --dataset ../BrickNet/outputs_preprocess/BrickNet-MM/sharegpt/BrickNet-MM_VAL.jsonl --scored ../BrickNet/outputs_val/qwen35_08b/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20/scored.jsonl --metrics-json ../BrickNet/outputs_val/qwen35_08b/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20/metrics.json --metrics-md ../BrickNet/outputs_val/qwen35_08b/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20/metrics.md --output ../BrickNet/outputs_val/qwen35_08b/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20/alignment.jsonl --bricknet-root ../BrickNet`

结果：BLEU-4 `91.2174`，ROUGE-L `55.4362`，Parsable `310/512 (60.55%)`，Clean
`78/512 (15.23%)`，Collision `5.2188`，PE `0.2823`，SigLIP2 `0.8007`，VQA
`0.7604`，Inventory F1 `0.8253`，Pose Match `0.1418`，Dense Reward `0.5355`，
Strict Success `14/512 (2.73%)`。完整结果见
`../BrickNet/outputs_val/qwen35_08b/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20/metrics.json`。

## exp1
### Qwen3.5-0.8B
**train**
  conda run -n llamafactory --no-capture-output llamafactory-cli train examples/train_lora/qwen35_08b_bricknet_mixed_pt.yaml
**eval**
1) LlamaFactory 生成预测
conda run -n llamafactory --no-capture-output llamafactory-cli train examples/train_lora/qwen35_08b_bricknet_mm_pt_exp1_predict.yaml
2) BrickNet 文本+渲染评测
cd ../BrickNet && /home/jiahao/miniconda3/envs/bricknet/bin/python scripts/evaluate_experiment.py \
  --predictions ../LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/eval_PT_exp1_ptval_in4096_out4096_p95_t1_k20/generated_predictions.jsonl \
  --text-metrics ../LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/eval_PT_exp1_ptval_in4096_out4096_p95_t1_k20/predict_results.json \
  --output-dir outputs_val/qwen35_08b/eval_PT_exp1_ptval_in4096_out4096_p95_t1_k20 \
  --prompts-file data/bricknet_datasets/captions_val.jsonl
3) 生成 alignment 输入
cd ../LlamaFactory && jq -c '{response: .predict, label: .label}' \
  saves/Qwen3.5-0.8B-Thinking/lora/eval_PT_exp1_ptval_in4096_out4096_p95_t1_k20/generated_predictions.jsonl \
  > ../BrickNet/outputs_val/qwen35_08b/eval_PT_exp1_ptval_in4096_out4096_p95_t1_k20/alignment_input.jsonl
4) ms-swift 对齐评测
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
1. conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3-VL-2B-Instruct --preprocessing_num_workers 16 --finetuning_type lora --quantization_method bnb --template qwen3_vl_nothink --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 --per_device_eval_batch_size 1 --predict_with_generate True --report_to none --max_new_tokens 512 --top_p 0.95 --temperature 1 --output_dir saves/Qwen3-VL-2B-Instruct/lora/eval_exp1_in4096_out512_p095_k20_t1 --trust_remote_code True --ddp_timeout 180000000 --do_predict True --adapter_name_or_path saves/Qwen3-VL-2B-Instruct/lora/train_exp1_qwen3vl_2b_val_ep10_bs1_ga8_lora16 --top_k 20
补充结果目录：`saves/Qwen3-VL-2B-Instruct/lora/eval_exp1_in4097_out512_p09_t095`。

### Qwen3.5-0.8B
**train**
训练输出目录：`saves/Qwen3.5-0.8B-Thinking/lora/train_exp1_qwen35_08b_val_ep10_bs1_ga8_lora16`
**eval**
conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --quantization_method bnb --template qwen3_5_nothink --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 --per_device_eval_batch_size 4 --predict_with_generate True --report_to none --max_new_tokens 512 --top_p 0.95 --temperature 1 --output_dir saves/Qwen3.5-0.8B-Thinking/lora/eval_exp1_in4096_out512_p095_k20_t1 --trust_remote_code True --ddp_timeout 180000000 --do_predict True --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_exp1_qwen35_08b_val_ep10_bs1_ga8_lora16 --top_k 20

### Qwen3.5-2B
**train**
训练输出目录：`saves/Qwen3.5-2B-Thinking/lora/train_exp1_qwen35_2b_val_ep10_bs1_ga8_lora16`
**eval**
conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3.5-2B --preprocessing_num_workers 16 --finetuning_type lora --quantization_method bnb --template qwen3_5_nothink --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 --per_device_eval_batch_size 1 --predict_with_generate True --report_to none --max_new_tokens 512 --top_p 0.95 --temperature 1 --output_dir saves/Qwen3.5-2B-Thinking/lora/eval_exp1_in4096_out512_p095_k20_t1 --trust_remote_code True --ddp_timeout 180000000 --do_predict True --top_k 20 --adapter_name_or_path saves/Qwen3.5-2B-Thinking/lora/train_exp1_qwen35_2b_val_ep10_bs1_ga8_lora16

## exp1_1
### Qwen3.5-2B
**train**
训练输出目录：`saves/Qwen3.5-2B-Thinking/lora/train_exp1_1_qwen35_2b_val_ep20_bs4_ga8_lora32_lr1e5_schdlconstanwarm`
**eval**
1. conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3.5-2B --preprocessing_num_workers 16 --finetuning_type lora --quantization_method bnb --template qwen3_5_nothink --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 --per_device_eval_batch_size 1 --predict_with_generate True --report_to none --max_new_tokens 512 --top_p 0.95 --temperature 1 --output_dir saves/Qwen3.5-2B-Thinking/lora/eval_exp1_1_in4096_out512_p095_k20_t1 --trust_remote_code True --ddp_timeout 180000000 --do_predict True --top_k 20 --adapter_name_or_path saves/Qwen3.5-2B-Thinking/lora/train_exp1_1_qwen35_2b_val_ep20_bs4_ga8_lora32_lr1e5_schdlconstanwarm

2. conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3.5-2B --preprocessing_num_workers 16 --finetuning_type lora --quantization_method bnb --template qwen3_5_nothink --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 --per_device_eval_batch_size 1 --predict_with_generate True --report_to none --max_new_tokens 512 --top_p 0.9 --temperature 0.95 --output_dir saves/Qwen3.5-2B-Thinking/lora/eval_exp1_1_in4097_out512_p09_t095 --trust_remote_code True --ddp_timeout 180000000 --do_predict True --adapter_name_or_path saves/Qwen3.5-2B-Thinking/lora/train_exp1_1_qwen35_2b_val_ep20_bs4_ga8_lora32_lr1e5_schdlconstanwarm

## exp2
### Qwen3.5-0.8B
- bricknet-mm sft 1w
**train**
1. conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --do_train True --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --template qwen3_5_nothink --flash_attn auto --dataset_dir data --dataset BrickNet-MM-SFT --cutoff_len 4096 --learning_rate 5e-05 --num_train_epochs 3.0 --max_samples 10000 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --lr_scheduler_type cosine --max_grad_norm 1.0 --logging_steps 10 --save_steps 500 --warmup_steps 0 --packing False --enable_thinking False --report_to none --output_dir saves/Qwen3.5-0.8B-Thinking/lora/train_exp2_qwen35_08b_sft1w_ep3_bs2_ga8_lora64 --bf16 True --plot_loss True --trust_remote_code True --ddp_timeout 180000000 --include_num_input_tokens_seen True --optim adamw_torch --lora_rank 64 --lora_alpha 128 --lora_dropout 0 --lora_target all --freeze_vision_tower True --freeze_multi_modal_projector True --image_max_pixels 589824 --image_min_pixels 1024 --video_max_pixels 65536 --video_min_pixels 256 --tokenized_path .llamafactory_cache/tokenized_dataset/BrickNet-MM-SFT-10000_qwen35-08b_nothink_len4096_img589824

**eval**
1. conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --quantization_method bnb --template qwen3_5_nothink --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 --per_device_eval_batch_size 1 --predict_with_generate True --report_to none --max_new_tokens 4096 --top_p 0.95 --temperature 1 --output_dir saves/Qwen3.5-0.8B-Thinking/lora/eval_exp2_in4096_out4096_p95_t1_k20 --trust_remote_code True --ddp_timeout 180000000 --do_predict True --top_k 20 --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_exp2_qwen35_08b_sft1w_ep3_bs2_ga8_lora64

## exp2_1
### Qwen3.5-0.8B
- bricknet-mm sft 5w
**train**
1. conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --do_train True --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --template qwen3_5_nothink --flash_attn auto --dataset_dir data --dataset BrickNet-MM-SFT --cutoff_len 4096 --learning_rate 5e-05 --num_train_epochs 3.0 --max_samples 50000 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --lr_scheduler_type cosine --max_grad_norm 1.0 --logging_steps 10 --save_steps 500 --warmup_steps 0 --packing False --enable_thinking False --report_to none --output_dir saves/Qwen3.5-0.8B-Thinking/lora/train_exp2_1_qwen35_08b_sft5w_ep3_bs2_ga8_lora64 --bf16 True --plot_loss True --trust_remote_code True --ddp_timeout 180000000 --include_num_input_tokens_seen True --optim adamw_torch --lora_rank 64 --lora_alpha 128 --lora_dropout 0 --lora_target all --freeze_vision_tower True --freeze_multi_modal_projector True --image_max_pixels 589824 --image_min_pixels 1024 --video_max_pixels 65536 --video_min_pixels 256 --tokenized_path .llamafactory_cache/tokenized_dataset/BrickNet-MM-SFT-50000_qwen35-08b_nothink_len4096_img589824

**eval**
1. conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --quantization_method bnb --template qwen3_5_nothink --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 --per_device_eval_batch_size 1 --predict_with_generate True --report_to none --max_new_tokens 4096 --top_p 0.95 --temperature 1 --output_dir saves/Qwen3.5-0.8B-Thinking/lora/eval_exp2_1_in4096_out4096_p95_t1_k20 --trust_remote_code True --ddp_timeout 180000000 --do_predict True --top_k 20 --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_exp2_1_qwen35_08b_sft5w_ep3_bs2_ga8_lora64

## exp2_2
### Qwen3.5-0.8B
- bricknet-mm sft all
**train**
1. conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --do_train True --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --template qwen3_5_nothink --flash_attn auto --dataset_dir data --dataset BrickNet-MM-SFT --cutoff_len 4096 --learning_rate 5e-05 --num_train_epochs 3.0 --max_samples 1000000 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --lr_scheduler_type cosine --max_grad_norm 1.0 --logging_steps 10 --save_steps 2000 --warmup_steps 0 --packing False --enable_thinking False --report_to none --output_dir saves/Qwen3.5-0.8B-Thinking/lora/train_exp2_2_qwen35_08b_sft_ep3_bs2_ga8_lora64 --bf16 True --plot_loss True --trust_remote_code True --ddp_timeout 180000000 --include_num_input_tokens_seen True --optim adamw_torch --lora_rank 64 --lora_alpha 128 --lora_dropout 0 --lora_target all --freeze_vision_tower True --freeze_multi_modal_projector True --image_max_pixels 589824 --image_min_pixels 1024 --video_max_pixels 65536 --video_min_pixels 256 --tokenized_path .llamafactory_cache/tokenized_dataset/BrickNet-MM-SFT_qwen35-08b_nothink_len4096_img589824

**eval**


## exp3
### Qwen3.5-0.8B-PT
- bricknet-mm sft 1w
- epoch 3
**train**
1. conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --do_train True --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --template qwen3_5_nothink --flash_attn auto --dataset_dir data --dataset BrickNet-MM-SFT --cutoff_len 4096 --learning_rate 5e-05 --num_train_epochs 3.0 --max_samples 10000 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --lr_scheduler_type cosine --max_grad_norm 1.0 --logging_steps 10 --save_steps 500 --warmup_steps 0 --packing False --enable_thinking False --report_to none --output_dir saves/Qwen3.5-0.8B-Thinking/lora/train_exp3_qwen35_08b_pt_sft1w_ep3_bs2_ga8_lora64 --bf16 True --plot_loss True --trust_remote_code True --ddp_timeout 180000000 --include_num_input_tokens_seen True --optim adamw_torch --lora_rank 64 --lora_alpha 128 --lora_dropout 0 --lora_target all --freeze_vision_tower True --freeze_multi_modal_projector True --image_max_pixels 589824 --image_min_pixels 1024 --video_max_pixels 65536 --video_min_pixels 256 --tokenized_path .llamafactory_cache/tokenized_dataset/BrickNet-MM-SFT-10000_qwen35-08b_nothink_len4096_img589824 --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64 --create_new_adapter

**eval**
1. conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --quantization_method bnb --template qwen3_5_nothink --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 --per_device_eval_batch_size 1 --predict_with_generate True --report_to none --max_new_tokens 4096 --top_p 0.95 --temperature 1 --output_dir saves/Qwen3.5-0.8B-Thinking/lora/eval_exp3_PT_in4096_out4096_p95_t1_k20 --trust_remote_code True --ddp_timeout 180000000 --do_predict True --top_k 20 --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64

2. conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --quantization_method bnb --template qwen3_5_nothink --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 --per_device_eval_batch_size 1 --predict_with_generate True --report_to none --max_new_tokens 4096 --top_p 0.95 --temperature 1 --output_dir saves/Qwen3.5-0.8B-Thinking/lora/eval_exp3_in4096_out4096_p95_t1_k20 --trust_remote_code True --ddp_timeout 180000000 --do_predict True --top_k 20 --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64,saves/Qwen3.5-0.8B-Thinking/lora/train_exp3_qwen35_08b_pt_sft1w_ep3_bs2_ga8_lora64

## exp3_0_1
### Qwen3.5-0.8B-PT
- bricknet-mm sft 1w
- epoch 10
**train**
1. conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --do_train True --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --template qwen3_5_nothink --flash_attn auto --dataset_dir data --dataset BrickNet-MM-SFT --cutoff_len 4096 --learning_rate 5e-05 --num_train_epochs 10 --max_samples 10000 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --lr_scheduler_type cosine --max_grad_norm 1.0 --logging_steps 10 --save_steps 500 --warmup_steps 0 --packing False --enable_thinking False --report_to none --output_dir saves/Qwen3.5-0.8B-Thinking/lora/train_exp3_0_1_qwen35_08b_pt_sft1w_ep10_bs2_ga8_lora64 --bf16 True --plot_loss True --trust_remote_code True --ddp_timeout 180000000 --include_num_input_tokens_seen True --optim adamw_torch --lora_rank 64 --lora_alpha 128 --lora_dropout 0 --lora_target all --freeze_vision_tower True --freeze_multi_modal_projector True --image_max_pixels 589824 --image_min_pixels 1024 --video_max_pixels 65536 --video_min_pixels 256 --tokenized_path .llamafactory_cache/tokenized_dataset/BrickNet-MM-SFT-10000_qwen35-08b_nothink_len4096_img589824 --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64 --create_new_adapter

**eval**
1. conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --quantization_method bnb --template qwen3_5_nothink --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 --per_device_eval_batch_size 1 --predict_with_generate True --report_to none --max_new_tokens 4096 --top_p 0.95 --temperature 1 --output_dir saves/Qwen3.5-0.8B-Thinking/lora/eval_exp3_0_1_in4096_out4096_p95_t1_k20 --trust_remote_code True --ddp_timeout 180000000 --do_predict True --top_k 20 --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64,saves/Qwen3.5-0.8B-Thinking/lora/train_exp3_0_1_qwen35_08b_pt_sft1w_ep10_bs2_ga8_lora64

2. cd ../BrickNet && /home/jiahao/miniconda3/envs/bricknet/bin/python scripts/evaluate_experiment.py --predictions ../LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/eval_exp3_0_1_in4096_out4096_p95_t1_k20/generated_predictions.jsonl --text-metrics ../LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/eval_exp3_0_1_in4096_out4096_p95_t1_k20/predict_results.json --output-dir outputs_val/qwen35_08b/eval_exp3_0_1_in4096_out4096_p95_t1_k20 --prompts-file data/bricknet_datasets/captions_val.jsonl


## exp3_1
### Qwen3.5-0.8B-PT
- bricknet-mm sft 5w
- epoch 3
**train**
1. conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --do_train True --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --template qwen3_5_nothink --flash_attn auto --dataset_dir data --dataset BrickNet-MM-SFT --cutoff_len 4096 --learning_rate 5e-05 --num_train_epochs 3.0 --max_samples 50000 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --lr_scheduler_type cosine --max_grad_norm 1.0 --logging_steps 10 --save_steps 1000 --warmup_steps 0 --packing False --enable_thinking False --report_to none --output_dir saves/Qwen3.5-0.8B-Thinking/lora/train_exp3_1_qwen35_08b_pt_sft5w_ep3_bs2_ga8_lora64 --bf16 True --plot_loss True --trust_remote_code True --ddp_timeout 180000000 --include_num_input_tokens_seen True --optim adamw_torch --lora_rank 64 --lora_alpha 128 --lora_dropout 0 --lora_target all --freeze_vision_tower True --freeze_multi_modal_projector True --image_max_pixels 589824 --image_min_pixels 1024 --video_max_pixels 65536 --video_min_pixels 256 --tokenized_path .llamafactory_cache/tokenized_dataset/BrickNet-MM-SFT-50000_qwen35-08b_nothink_len4096_img589824 --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64 --create_new_adapter

**eval**
1. `conda run -n llamafactory --no-capture-output llamafactory-cli train examples/train_lora/qwen35_08b_bricknet_mm_exp3_1_predict.yaml`
2. `cd ../BrickNet && /home/jiahao/miniconda3/envs/bricknet/bin/python scripts/evaluate_experiment.py --predictions ../LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/eval_exp3_1_in4096_out4096_p95_t1_k20/generated_predictions.jsonl --text-metrics ../LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/eval_exp3_1_in4096_out4096_p95_t1_k20/predict_results.json --output-dir outputs_val/qwen35_08b/eval_exp3_1_in4096_out4096_p95_t1_k20 --prompts-file data/bricknet_datasets/captions_val.jsonl --render-jobs 8 --eval-workers 8 --eval-batch-size 8`
3. `jq -c '{response: .predict, label: .label}' saves/Qwen3.5-0.8B-Thinking/lora/eval_exp3_1_in4096_out4096_p95_t1_k20/generated_predictions.jsonl > ../BrickNet/outputs_val/qwen35_08b/eval_exp3_1_in4096_out4096_p95_t1_k20/alignment_input.jsonl`
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
1. conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --do_train True --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --template qwen3_5_nothink --flash_attn auto --dataset_dir data --dataset BrickNet-MM-SFT --cutoff_len 4096 --learning_rate 5e-05 --num_train_epochs 3.0 --max_samples 1000000 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --lr_scheduler_type cosine --max_grad_norm 1.0 --logging_steps 10 --save_steps 2000 --warmup_steps 0 --packing False --enable_thinking False --report_to none --output_dir saves/Qwen3.5-0.8B-Thinking/lora/train_exp3_2_qwen35_08b_pt_sft_ep3_bs2_ga8_lora64 --bf16 True --plot_loss True --trust_remote_code True --ddp_timeout 180000000 --include_num_input_tokens_seen True --optim adamw_torch --lora_rank 64 --lora_alpha 128 --lora_dropout 0 --lora_target all --freeze_vision_tower True --freeze_multi_modal_projector True --image_max_pixels 589824 --image_min_pixels 1024 --video_max_pixels 65536 --video_min_pixels 256 --tokenized_path .llamafactory_cache/tokenized_dataset/BrickNet-MM-SFT_qwen35-08b_nothink_len4096_img589824 --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64 --create_new_adapter

**eval**



## Stage 2 exp4–exp4_3（四个训练和推理完成，10k全指标完成）
四个实验均以已完成的 mixed PT-exp1 final 为共同初始化。`exp4/exp4_1` overfit 训练与 VAL512 推理完成，
机械 gate 已批准；`exp4_2/exp4_3` 10k 训练与 VAL512 推理均完成。50k/all 继续暂停且不分配版本号。


### exp4 — NonThinking-Control VAL511 overfit
状态：训练完成，train loss `0.1737931`；VAL512 512/512 推理完成。
**train**
1. `python scripts/launch_bricknet_stage2_sft.py --action train --variant nonthinking-control --scale overfit511 --execute --stage0-gate-approved`
**predict VAL512**
1. `python scripts/launch_bricknet_stage2_sft.py --action predict --variant nonthinking-control --scale overfit511 --execute --stage0-gate-approved`
**eval**
1. `python scripts/evaluate_bricknet_stage2.py --experiment exp4 --execute`


### exp4_1 — Thinking-Hard VAL511 overfit
状态：训练完成，train loss `0.0859583`；VAL512 512/512 推理完成。
**train**
1. `python scripts/launch_bricknet_stage2_sft.py --action train --variant thinking-hard --scale overfit511 --execute --stage0-gate-approved`
**predict VAL512**
1. `python scripts/launch_bricknet_stage2_sft.py --action predict --variant thinking-hard --scale overfit511 --execute --stage0-gate-approved`
**eval**
1. `python scripts/evaluate_bricknet_stage2.py --experiment exp4_1 --execute`


### exp4_2 — NonThinking-Control 10k
状态：训练完成，train loss `0.1726632`；VAL512 512/512 推理和全指标完成。parsable
`382/512 (74.61%)`、clean `93/512 (18.16%)`、dense reward `0.58159`、strict success `16/512 (3.12%)`。
**train**
1. `python scripts/launch_bricknet_stage2_sft.py --action train --variant nonthinking-control --scale 10k --execute --stage0-gate-approved --overfit-gate-approved`
**predict VAL512**
1. `python scripts/launch_bricknet_stage2_sft.py --action predict --variant nonthinking-control --scale 10k --execute --stage0-gate-approved --overfit-gate-approved`
**eval**
1. `python scripts/evaluate_bricknet_stage2.py --experiment exp4_2 --execute`


### exp4_3 — Thinking-Hard 10k
状态：训练完成，train loss `0.0433687`；VAL512 512/512 推理完成。strict extractor 得到
`360/512 (70.31%)` 完整合法 trace/path，512 条均有非空 extracted prefix；全指标完成：clean
`101/512 (19.73%)`、dense reward `0.57395`、strict success `13/512 (2.54%)`。相对 exp4_2 没有总体优势，
T1-10k 人工推广 gate 未批准。
**train**
1. `python scripts/launch_bricknet_stage2_sft.py --action train --variant thinking-hard --scale 10k  --execute --stage0-gate-approved --overfit-gate-approved`
**predict VAL512**
1. `python scripts/launch_bricknet_stage2_sft.py --action predict --variant thinking-hard --scale 10k --execute --stage0-gate-approved --overfit-gate-approved`
**eval**
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
cd /home/jiahao/task/LlamaFactory

python scripts/launch_bricknet_stage2_sft.py \
  --action train --variant thinking-hard-v2-lean-state --scale 10k \
  --overfit-gate-approved

python scripts/launch_bricknet_stage2_sft.py \
  --action train --variant thinking-hard-v2-lean-state --scale 10k \
  --execute --stage0-gate-approved --overfit-gate-approved

python scripts/launch_bricknet_stage2_sft.py \
  --action predict --variant thinking-hard-v2-lean-state --scale 10k

python scripts/launch_bricknet_stage2_sft.py \
  --action predict --variant thinking-hard-v2-lean-state --scale 10k \
  --execute --stage0-gate-approved

python scripts/evaluate_bricknet_stage2.py --experiment exp4_3_1
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
python scripts/launch_bricknet_stage3_sft.py --action train
```

正式执行必须同时通过 Stage 0 final、Stage 3 Pilot 人工 approval、exp4_3/T1-10k gate、T2-10k 完整 hard replay、
真实 Qwen processor paired token audit、10k dataset/registry hash 和输出目录 gate。预测还需要 exp5 adapter。
Stage 0 已通过；当前 dry-run 仍应被 Pilot 人工 approval、exp4_3/T1-10k paired gate、T2-10k 数据/replay/token
和 registry 阻断，`training_started=false`。50k/all 不分配版本号或配置。

## PT-exp2 & exp4_4~exp4_6

固定序列为 `PT-exp2-text8m → PT-exp2-mm-e1/e2/e3 → PT-exp2 alias → exp4_4 10k → exp4_5 50k → exp4_6 all`。
不创建 PT-exp2 VAL511 训练或验证。详细数据 hash、配置与 gate 见 [PT-exp2 runbook](bricknet-pt-exp2.md)。

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
已完成的 text8m `tokenized_path` 由所选进程只读复用，不会因为切换单双卡再执行 7,698,261 条 tokenizer；
只有缓存缺失、损坏或路径/config 指纹改变时才会重新预处理。预测与评测仍用单卡。`select-final`、
`approve-scale` 是有意保留的人工决策点，运行前应先阅读上一条评测产生的 metrics：

如只使用 GPU 0，把任一训练命令的 `--gpus 0 1` 改为 `--gpus 0` 即可；launcher 会自动恢复单卡 GA，
无需修改 YAML。例如 text8m 单卡 dry-run：

```bash
python scripts/launch_bricknet_pt_exp2.py --gpus 0 --action train --run text8m
```

```bash
cd /data/jiahao/task/LlamaFactory

python scripts/launch_bricknet_pt_exp2.py --gpus 0 1 --action train --run text8m --execute
python scripts/launch_bricknet_pt_exp2.py --gpus 0 1 --action train --run mm-e1 --execute
python scripts/launch_bricknet_pt_exp2.py --gpus 0 --action predict --run mm-e1 --execute
python scripts/launch_bricknet_pt_exp2.py --gpus 0 --action evaluate --run mm-e1 --execute
python scripts/launch_bricknet_pt_exp2.py --gpus 0 1 --action train --run mm-e2 --execute
python scripts/launch_bricknet_pt_exp2.py --gpus 0 --action predict --run mm-e2 --execute
python scripts/launch_bricknet_pt_exp2.py --gpus 0 --action evaluate --run mm-e2 --execute
python scripts/launch_bricknet_pt_exp2.py --gpus 0 1 --action train --run mm-e3 --execute
python scripts/launch_bricknet_pt_exp2.py --gpus 0 --action predict --run mm-e3 --execute
python scripts/launch_bricknet_pt_exp2.py --gpus 0 --action evaluate --run mm-e3 --execute
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
python scripts/launch_bricknet_pt_exp2.py --gpus 0 1 --action train --run exp4_4 --execute
python scripts/launch_bricknet_pt_exp2.py --gpus 0 --action predict --run exp4_4 --execute
python scripts/launch_bricknet_pt_exp2.py --gpus 0 --action evaluate --run exp4_4 --execute
python scripts/launch_bricknet_pt_exp2.py --action approve-scale --run exp4_4 --execute --approve
python scripts/launch_bricknet_pt_exp2.py --action materialize --run exp4_5 --execute
```

### exp4_5
```bash
python scripts/launch_bricknet_pt_exp2.py --gpus 0 1 --action train --run exp4_5 --execute
python scripts/launch_bricknet_pt_exp2.py --gpus 0 --action predict --run exp4_5 --execute
python scripts/launch_bricknet_pt_exp2.py --gpus 0 --action evaluate --run exp4_5 --execute
python scripts/launch_bricknet_pt_exp2.py --action approve-scale --run exp4_5 --execute --approve
```

### exp4_6
```bash
python scripts/launch_bricknet_pt_exp2.py --gpus 0 1 --action train --run exp4_6 --execute
python scripts/launch_bricknet_pt_exp2.py --gpus 0 --action predict --run exp4_6 --execute
python scripts/launch_bricknet_pt_exp2.py --gpus 0 --action evaluate --run exp4_6 --execute
```
