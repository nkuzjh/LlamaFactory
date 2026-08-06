
# notes

- 统一实验进度与结果：[experiment_results.md](experiment_results.md)。
- 唯一总推进计划：`/home/jiahao/task/BrickNet/BrickNet-MM Agentic LEGO Planner/Constructor Plan.md`。
- 2026-08-05 决策：66,456 条基础处理池不再四分；统一使用 512 VAL；policy-specific full hard mining 暂停，
  条件性恢复时最多处理 2,000 prompts。

    Qwen3-VL-2B-Instruct
    Qwen3.5-0.8B
    Qwen3.5-2B
    hf download Qwen/Qwen3.5-2B-Base



    --tokenized_path .llamafactory_cache/tokenized_dataset/BrickNet-MM-SFT_qwen35-08b_nothink_len4096_img589824 \

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
--output_dir saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64

**eval**
1. `llamafactory-cli train examples/train_lora/qwen35_08b_bricknet_mm_pt_predict.yaml`
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
  llamafactory-cli train examples/train_lora/qwen35_08b_bricknet_mixed_pt.yaml

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
train_exp1_qwen3vl_2b_val_ep10_bs1_ga8_lora16
**eval**
1. llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3-VL-2B-Instruct --preprocessing_num_workers 16 --finetuning_type lora --quantization_method bnb --template qwen3_vl_nothink --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 --per_device_eval_batch_size1 --predict_with_generate True --report_to none --max_new_tokens 512 --top_p 0.95 --temperature 1 --output_dir saves/Qwen3-VL-2B-Instruct/lora/eval_exp1_in4096_out512_p095_k20_t1 --trust_remote_code True --ddp_timeout 180000000 --do_predict True --adapter_name_or_path saves/Qwen3-VL-2B-Instruct/lora/train_exp1_qwen3vl_2b_val_ep10_bs1_ga8_lora16 --top_k 20
2. eval_exp1_in4097_out512_p09_t095

### Qwen3.5-0.8B
**train**
train_exp1_qwen35_08b_val_ep10_bs1_ga8_lora16
**eval**
llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --quantization_method bnb --template qwen3_5_nothink --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 --per_device_eval_batch_size 4 --predict_with_generate True --report_to none --max_new_tokens 512 --top_p 0.95 --temperature 1 --output_dir saves/Qwen3.5-0.8B-Thinking/lora/eval_exp1_in4096_out512_p095_k20_t1 --trust_remote_code True --ddp_timeout 180000000 --do_predict True --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_exp1_qwen35_08b_val_ep10_bs1_ga8_lora16 --top_k 20

### Qwen3.5-2B
**train**
train_exp1_qwen35_2b_val_ep10_bs1_ga8_lora16
**eval**
llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3.5-2B --preprocessing_num_workers 16 --finetuning_type lora --quantization_method bnb --template qwen3_5_nothink --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 --per_device_eval_batch_size 1 --predict_with_generate True --report_to none --max_new_tokens 512 --top_p 0.95 --temperature 1 --output_dir saves/Qwen3.5-2B-Thinking/lora/eval_exp1_in4096_out512_p095_k20_t1 --trust_remote_code True --ddp_timeout 180000000 --do_predict True --top_k 20 --adapter_name_or_path saves/Qwen3.5-2B-Thinking/lora/train_exp1_qwen35_2b_val_ep10_bs1_ga8_lora16

## exp1_1
### Qwen3.5-2B
**train**
train_exp1_1_qwen35_2b_val_ep20_bs4_ga8_lora32_lr1e5_schdlconstanwarm
**eval**
1. llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3.5-2B --preprocessing_num_workers 16 --finetuning_type lora --quantization_method bnb --template qwen3_5_nothink --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 --per_device_eval_batch_size 1 --predict_with_generate True --report_to none --max_new_tokens 512 --top_p 0.95 --temperature 1 --output_dir saves/Qwen3.5-2B-Thinking/lora/eval_exp1_1_in4096_out512_p095_k20_t1 --trust_remote_code True --ddp_timeout 180000000 --do_predict True --top_k 20 --adapter_name_or_path saves/Qwen3.5-2B-Thinking/lora/train_exp1_1_qwen35_2b_val_ep20_bs4_ga8_lora32_lr1e5_schdlconstanwarm

2. llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3.5-2B --preprocessing_num_workers 16 --finetuning_type lora --quantization_method bnb --template qwen3_5_nothink --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 --per_device_eval_batch_size 1 --predict_with_generate True --report_to none --max_new_tokens 512 --top_p 0.9 --temperature 0.95 --output_dir saves/Qwen3.5-2B-Thinking/lora/eval_exp1_1_in4097_out512_p09_t095 --trust_remote_code True --ddp_timeout 180000000 --do_predict True --adapter_name_or_path saves/Qwen3.5-2B-Thinking/lora/train_exp1_1_qwen35_2b_val_ep20_bs4_ga8_lora32_lr1e5_schdlconstanwarm

## exp2
### Qwen3.5-0.8B
- bricknet-mm sft 1w
**train**
1. llamafactory-cli train --stage sft --do_train True --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --template qwen3_5_nothink --flash_attn auto --dataset_dir data --dataset BrickNet-MM-SFT --cutoff_len 4096 --learning_rate 5e-05 --num_train_epochs 3.0 --max_samples 10000 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --lr_scheduler_type cosine --max_grad_norm 1.0 --logging_steps 10 --save_steps 500 --warmup_steps 0 --packing False --enable_thinking False --report_to none --output_dir saves/Qwen3.5-0.8B-Thinking/lora/train_exp2_qwen35_08b_sft1w_ep3_bs2_ga8_lora64 --bf16 True --plot_loss True --trust_remote_code True --ddp_timeout 180000000 --include_num_input_tokens_seen True --optim adamw_torch --lora_rank 64 --lora_alpha 128 --lora_dropout 0 --lora_target all --freeze_vision_tower True --freeze_multi_modal_projector True --image_max_pixels 589824 --image_min_pixels 1024 --video_max_pixels 65536 --video_min_pixels 256 --tokenized_path .llamafactory_cache/tokenized_dataset/BrickNet-MM-SFT-10000_qwen35-08b_nothink_len4096_img589824

**eval**
1. llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --quantization_method bnb --template qwen3_5_nothink --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 --per_device_eval_batch_size 1 --predict_with_generate True --report_to none --max_new_tokens 4096 --top_p 0.95 --temperature 1 --output_dir saves/Qwen3.5-0.8B-Thinking/lora/eval_exp2_in4096_out4096_p95_t1_k20 --trust_remote_code True --ddp_timeout 180000000 --do_predict True --top_k 20 --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_exp2_qwen35_08b_sft1w_ep3_bs2_ga8_lora64

## exp2_1
### Qwen3.5-0.8B
- bricknet-mm sft 5w
**train**
1. llamafactory-cli train --stage sft --do_train True --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --template qwen3_5_nothink --flash_attn auto --dataset_dir data --dataset BrickNet-MM-SFT --cutoff_len 4096 --learning_rate 5e-05 --num_train_epochs 3.0 --max_samples 50000 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --lr_scheduler_type cosine --max_grad_norm 1.0 --logging_steps 10 --save_steps 500 --warmup_steps 0 --packing False --enable_thinking False --report_to none --output_dir saves/Qwen3.5-0.8B-Thinking/lora/train_exp2_1_qwen35_08b_sft5w_ep3_bs2_ga8_lora64 --bf16 True --plot_loss True --trust_remote_code True --ddp_timeout 180000000 --include_num_input_tokens_seen True --optim adamw_torch --lora_rank 64 --lora_alpha 128 --lora_dropout 0 --lora_target all --freeze_vision_tower True --freeze_multi_modal_projector True --image_max_pixels 589824 --image_min_pixels 1024 --video_max_pixels 65536 --video_min_pixels 256 --tokenized_path .llamafactory_cache/tokenized_dataset/BrickNet-MM-SFT-50000_qwen35-08b_nothink_len4096_img589824

**eval**
1. llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --quantization_method bnb --template qwen3_5_nothink --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 --per_device_eval_batch_size 1 --predict_with_generate True --report_to none --max_new_tokens 4096 --top_p 0.95 --temperature 1 --output_dir saves/Qwen3.5-0.8B-Thinking/lora/eval_exp2_1_in4096_out4096_p95_t1_k20 --trust_remote_code True --ddp_timeout 180000000 --do_predict True --top_k 20 --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_exp2_1_qwen35_08b_sft5w_ep3_bs2_ga8_lora64

## exp2_2
### Qwen3.5-0.8B
- bricknet-mm sft all
**train**
1. llamafactory-cli train --stage sft --do_train True --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --template qwen3_5_nothink --flash_attn auto --dataset_dir data --dataset BrickNet-MM-SFT --cutoff_len 4096 --learning_rate 5e-05 --num_train_epochs 3.0 --max_samples 1000000 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --lr_scheduler_type cosine --max_grad_norm 1.0 --logging_steps 10 --save_steps 2000 --warmup_steps 0 --packing False --enable_thinking False --report_to none --output_dir saves/Qwen3.5-0.8B-Thinking/lora/train_exp2_2_qwen35_08b_sft_ep3_bs2_ga8_lora64 --bf16 True --plot_loss True --trust_remote_code True --ddp_timeout 180000000 --include_num_input_tokens_seen True --optim adamw_torch --lora_rank 64 --lora_alpha 128 --lora_dropout 0 --lora_target all --freeze_vision_tower True --freeze_multi_modal_projector True --image_max_pixels 589824 --image_min_pixels 1024 --video_max_pixels 65536 --video_min_pixels 256 --tokenized_path .llamafactory_cache/tokenized_dataset/BrickNet-MM-SFT_qwen35-08b_nothink_len4096_img589824

**eval**


## exp3
### Qwen3.5-0.8B-PT
- bricknet-mm sft 1w
- epoch 3
**train**
1. llamafactory-cli train --stage sft --do_train True --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --template qwen3_5_nothink --flash_attn auto --dataset_dir data --dataset BrickNet-MM-SFT --cutoff_len 4096 --learning_rate 5e-05 --num_train_epochs 3.0 --max_samples 10000 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --lr_scheduler_type cosine --max_grad_norm 1.0 --logging_steps 10 --save_steps 500 --warmup_steps 0 --packing False --enable_thinking False --report_to none --output_dir saves/Qwen3.5-0.8B-Thinking/lora/train_exp3_qwen35_08b_pt_sft1w_ep3_bs2_ga8_lora64 --bf16 True --plot_loss True --trust_remote_code True --ddp_timeout 180000000 --include_num_input_tokens_seen True --optim adamw_torch --lora_rank 64 --lora_alpha 128 --lora_dropout 0 --lora_target all --freeze_vision_tower True --freeze_multi_modal_projector True --image_max_pixels 589824 --image_min_pixels 1024 --video_max_pixels 65536 --video_min_pixels 256 --tokenized_path .llamafactory_cache/tokenized_dataset/BrickNet-MM-SFT-10000_qwen35-08b_nothink_len4096_img589824 --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64 --create_new_adapter

**eval**
1. llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --quantization_method bnb --template qwen3_5_nothink --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 --per_device_eval_batch_size 1 --predict_with_generate True --report_to none --max_new_tokens 4096 --top_p 0.95 --temperature 1 --output_dir saves/Qwen3.5-0.8B-Thinking/lora/eval_exp3_PT_in4096_out4096_p95_t1_k20 --trust_remote_code True --ddp_timeout 180000000 --do_predict True --top_k 20 --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64

2. llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --quantization_method bnb --template qwen3_5_nothink --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 --per_device_eval_batch_size 1 --predict_with_generate True --report_to none --max_new_tokens 4096 --top_p 0.95 --temperature 1 --output_dir saves/Qwen3.5-0.8B-Thinking/lora/eval_exp3_in4096_out4096_p95_t1_k20 --trust_remote_code True --ddp_timeout 180000000 --do_predict True --top_k 20 --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64,saves/Qwen3.5-0.8B-Thinking/lora/train_exp3_qwen35_08b_pt_sft1w_ep3_bs2_ga8_lora64

## exp3_0_1
### Qwen3.5-0.8B-PT
- bricknet-mm sft 1w
- epoch 10
**train**
1. llamafactory-cli train --stage sft --do_train True --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --template qwen3_5_nothink --flash_attn auto --dataset_dir data --dataset BrickNet-MM-SFT --cutoff_len 4096 --learning_rate 5e-05 --num_train_epochs 10 --max_samples 10000 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --lr_scheduler_type cosine --max_grad_norm 1.0 --logging_steps 10 --save_steps 500 --warmup_steps 0 --packing False --enable_thinking False --report_to none --output_dir saves/Qwen3.5-0.8B-Thinking/lora/train_exp3_0_1_qwen35_08b_pt_sft1w_ep10_bs2_ga8_lora64 --bf16 True --plot_loss True --trust_remote_code True --ddp_timeout 180000000 --include_num_input_tokens_seen True --optim adamw_torch --lora_rank 64 --lora_alpha 128 --lora_dropout 0 --lora_target all --freeze_vision_tower True --freeze_multi_modal_projector True --image_max_pixels 589824 --image_min_pixels 1024 --video_max_pixels 65536 --video_min_pixels 256 --tokenized_path .llamafactory_cache/tokenized_dataset/BrickNet-MM-SFT-10000_qwen35-08b_nothink_len4096_img589824 --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64 --create_new_adapter

**eval**
1. llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --quantization_method bnb --template qwen3_5_nothink --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 --per_device_eval_batch_size 1 --predict_with_generate True --report_to none --max_new_tokens 4096 --top_p 0.95 --temperature 1 --output_dir saves/Qwen3.5-0.8B-Thinking/lora/eval_exp3_0_1_in4096_out4096_p95_t1_k20 --trust_remote_code True --ddp_timeout 180000000 --do_predict True --top_k 20 --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64,saves/Qwen3.5-0.8B-Thinking/lora/train_exp3_0_1_qwen35_08b_pt_sft1w_ep10_bs2_ga8_lora64

2. cd ../BrickNet && /home/jiahao/miniconda3/envs/bricknet/bin/python scripts/evaluate_experiment.py --predictions ../LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/eval_exp3_0_1_in4096_out4096_p95_t1_k20/generated_predictions.jsonl --text-metrics ../LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/eval_exp3_0_1_in4096_out4096_p95_t1_k20/predict_results.json --output-dir outputs_val/qwen35_08b/eval_exp3_0_1_in4096_out4096_p95_t1_k20 --prompts-file data/bricknet_datasets/captions_val.jsonl


## exp3_1
### Qwen3.5-0.8B-PT
- bricknet-mm sft 5w
- epoch 3
**train**
1. llamafactory-cli train --stage sft --do_train True --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --template qwen3_5_nothink --flash_attn auto --dataset_dir data --dataset BrickNet-MM-SFT --cutoff_len 4096 --learning_rate 5e-05 --num_train_epochs 3.0 --max_samples 50000 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --lr_scheduler_type cosine --max_grad_norm 1.0 --logging_steps 10 --save_steps 1000 --warmup_steps 0 --packing False --enable_thinking False --report_to none --output_dir saves/Qwen3.5-0.8B-Thinking/lora/train_exp3_1_qwen35_08b_pt_sft5w_ep3_bs2_ga8_lora64 --bf16 True --plot_loss True --trust_remote_code True --ddp_timeout 180000000 --include_num_input_tokens_seen True --optim adamw_torch --lora_rank 64 --lora_alpha 128 --lora_dropout 0 --lora_target all --freeze_vision_tower True --freeze_multi_modal_projector True --image_max_pixels 589824 --image_min_pixels 1024 --video_max_pixels 65536 --video_min_pixels 256 --tokenized_path .llamafactory_cache/tokenized_dataset/BrickNet-MM-SFT-50000_qwen35-08b_nothink_len4096_img589824 --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64 --create_new_adapter

**eval**
1. `llamafactory-cli train examples/train_lora/qwen35_08b_bricknet_mm_exp3_1_predict.yaml`
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
1. llamafactory-cli train --stage sft --do_train True --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora --template qwen3_5_nothink --flash_attn auto --dataset_dir data --dataset BrickNet-MM-SFT --cutoff_len 4096 --learning_rate 5e-05 --num_train_epochs 3.0 --max_samples 1000000 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --lr_scheduler_type cosine --max_grad_norm 1.0 --logging_steps 10 --save_steps 2000 --warmup_steps 0 --packing False --enable_thinking False --report_to none --output_dir saves/Qwen3.5-0.8B-Thinking/lora/train_exp3_2_qwen35_08b_pt_sft_ep3_bs2_ga8_lora64 --bf16 True --plot_loss True --trust_remote_code True --ddp_timeout 180000000 --include_num_input_tokens_seen True --optim adamw_torch --lora_rank 64 --lora_alpha 128 --lora_dropout 0 --lora_target all --freeze_vision_tower True --freeze_multi_modal_projector True --image_max_pixels 589824 --image_min_pixels 1024 --video_max_pixels 65536 --video_min_pixels 256 --tokenized_path .llamafactory_cache/tokenized_dataset/BrickNet-MM-SFT_qwen35-08b_nothink_len4096_img589824 --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64 --create_new_adapter

**eval**
