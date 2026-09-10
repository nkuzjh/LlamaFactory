# Experiment record

前置阅读：[Planner README](../BrickNet/BrickNet-MM%20Agentic%20LEGO%20Planner/README.md) → [Constructor Plan](../BrickNet/BrickNet-MM%20Agentic%20LEGO%20Planner/Constructor%20Plan.md)。结果账本：[experiment_results.md](experiment_results.md)。

除非命令块明确切换仓库，均先从 `/home/jiahao/task/LlamaFactory` 执行；LlamaFactory 命令使用
`conda run -n llamafactory --no-capture-output`。

# PT experiments

## exp0

Qwen3.5-0.8B；PT-exp0 LoRA 版本为
`saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64`。
现有记录没有 exp0 train command，以下只保留已有 prediction/evaluation 命令。

```bash
cd /home/jiahao/task/LlamaFactory

conda run -n llamafactory --no-capture-output llamafactory-cli train \
  examples/train_lora/qwen35_08b_bricknet_mm_pt_predict.yaml

cd /home/jiahao/task/BrickNet
/home/jiahao/miniconda3/envs/bricknet/bin/python scripts/evaluate_experiment.py \
  --predictions /home/jiahao/task/LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20/generated_predictions.jsonl \
  --text-metrics /home/jiahao/task/LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20/predict_results.json \
  --output-dir outputs_val/qwen35_08b/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20 \
  --prompts-file data/bricknet_datasets/captions_val.jsonl

cd /home/jiahao/task/LlamaFactory
jq -c '{response: .predict, label: .label}' \
  saves/Qwen3.5-0.8B-Thinking/lora/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20/generated_predictions.jsonl \
  > /home/jiahao/task/BrickNet/outputs_val/qwen35_08b/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20/alignment_input.jsonl

cd /home/jiahao/task/ms-swift
/home/jiahao/miniconda3/envs/bricknet/bin/python \
  examples/train/grpo/plugin/bricknet/evaluate_experiment.py alignment-worker \
  --results /home/jiahao/task/BrickNet/outputs_val/qwen35_08b/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20/alignment_input.jsonl \
  --dataset /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM/sharegpt/BrickNet-MM_VAL.jsonl \
  --scored /home/jiahao/task/BrickNet/outputs_val/qwen35_08b/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20/scored.jsonl \
  --metrics-json /home/jiahao/task/BrickNet/outputs_val/qwen35_08b/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20/metrics.json \
  --metrics-md /home/jiahao/task/BrickNet/outputs_val/qwen35_08b/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20/metrics.md \
  --output /home/jiahao/task/BrickNet/outputs_val/qwen35_08b/eval_PT_exp0_ptval_in4096_out4096_p95_t1_k20/alignment.jsonl \
  --bricknet-root /home/jiahao/task/BrickNet

cd /home/jiahao/task/LlamaFactory
```

## exp1

Qwen3.5-0.8B mixed PT-exp1；训练配置为
`examples/train_lora/qwen35_08b_bricknet_mixed_pt.yaml`，版本为
`saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp1_qwen35_08b_bricknet_text270k_mmpt135k_ep1_bs2_ga8_lora64`。

```bash
cd /home/jiahao/task/LlamaFactory

conda run -n llamafactory --no-capture-output llamafactory-cli train \
  examples/train_lora/qwen35_08b_bricknet_mixed_pt.yaml
conda run -n llamafactory --no-capture-output llamafactory-cli train \
  examples/train_lora/qwen35_08b_bricknet_mm_pt_exp1_predict.yaml

cd /home/jiahao/task/BrickNet
/home/jiahao/miniconda3/envs/bricknet/bin/python scripts/evaluate_experiment.py \
  --predictions /home/jiahao/task/LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/eval_PT_exp1_ptval_in4096_out4096_p95_t1_k20/generated_predictions.jsonl \
  --text-metrics /home/jiahao/task/LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/eval_PT_exp1_ptval_in4096_out4096_p95_t1_k20/predict_results.json \
  --output-dir outputs_val/qwen35_08b/eval_PT_exp1_ptval_in4096_out4096_p95_t1_k20 \
  --prompts-file data/bricknet_datasets/captions_val.jsonl

cd /home/jiahao/task/LlamaFactory
jq -c '{response: .predict, label: .label}' \
  saves/Qwen3.5-0.8B-Thinking/lora/eval_PT_exp1_ptval_in4096_out4096_p95_t1_k20/generated_predictions.jsonl \
  > /home/jiahao/task/BrickNet/outputs_val/qwen35_08b/eval_PT_exp1_ptval_in4096_out4096_p95_t1_k20/alignment_input.jsonl

cd /home/jiahao/task/ms-swift
/home/jiahao/miniconda3/envs/bricknet/bin/python \
  examples/train/grpo/plugin/bricknet/evaluate_experiment.py alignment-worker \
  --results /home/jiahao/task/BrickNet/outputs_val/qwen35_08b/eval_PT_exp1_ptval_in4096_out4096_p95_t1_k20/alignment_input.jsonl \
  --dataset /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM/sharegpt/BrickNet-MM_VAL.jsonl \
  --scored /home/jiahao/task/BrickNet/outputs_val/qwen35_08b/eval_PT_exp1_ptval_in4096_out4096_p95_t1_k20/scored.jsonl \
  --metrics-json /home/jiahao/task/BrickNet/outputs_val/qwen35_08b/eval_PT_exp1_ptval_in4096_out4096_p95_t1_k20/metrics.json \
  --metrics-md /home/jiahao/task/BrickNet/outputs_val/qwen35_08b/eval_PT_exp1_ptval_in4096_out4096_p95_t1_k20/metrics.md \
  --output /home/jiahao/task/BrickNet/outputs_val/qwen35_08b/eval_PT_exp1_ptval_in4096_out4096_p95_t1_k20/alignment.jsonl \
  --bricknet-root /home/jiahao/task/BrickNet

cd /home/jiahao/task/LlamaFactory
```

# BrickNet-Paper SFT experiments

GT 使用已完成并验证的 `outputs_gt/val_8view_renders_v2_official_rowids`；以下入口仅对五个 paper SFT 模型执行官方 BrickNet-Render 八视图渲染，并对 GT/五模型统一计算指标。任一指标在最低批量/低显存配置下仍失败时保留其他指标，并让对应单元为空。

```bash
set -euo pipefail
cd /home/jiahao/task/BrickNet

PY=/home/jiahao/miniconda3/envs/bricknet/bin/python
ROOT=outputs_val/bricknet_paper_official
CAPTIONS=data/bricknet_datasets/captions_val.jsonl
RENDER_BIN=/home/jiahao/task/.conda/bricknet-render-official-b49e873/bin/bricknet-render

score_target() {
  local target="$1" renders="$2"
  local out="$ROOT/$target"
  "$PY" scripts/package_eval_renders.py --renders-dir "$renders" \
    --captions-jsonl "$CAPTIONS" --out-dir "$out/render_8view_shards"
  run_metric() {
    local metric="$1"
    local pending="$out/${metric}.pending.jsonl"
    shift
    if "$@" --out "$pending" && mv -f "$pending" "$out/${metric}.jsonl"; then
      :
    else
      rm -f "$pending"
      echo "$metric skipped after failure; leave metric cell blank: $target" >&2
    fi
  }
  run_metric pe "$PY" eval/pe.py --images-dir "$out/render_8view_shards" \
    --captions-jsonl "$out/render_8view_shards/captions_rendered.jsonl" \
    --num-workers 2 --batch-size 1
  run_metric siglip2 "$PY" eval/siglip2.py --images-dir "$out/render_8view_shards" \
    --captions-jsonl "$out/render_8view_shards/captions_rendered.jsonl" \
    --num-workers 2 --batch-size 1 --view-batch-size 1
  run_metric vqa env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" eval/vqascore.py --images-dir "$out/render_8view_shards" \
    --captions-jsonl "$out/render_8view_shards/captions_rendered.jsonl" \
    --num-workers 2 --attn-implementation sdpa --max-gpu-memory 2GiB
}

score_target gt_val outputs_gt/val_8view_renders_v2_official_rowids
for spec in \
  'qwen3_06b_sft|outputs_val/qwen3_06b/models1' \
  'qwen3_1_7b_sft|outputs_val/qwen3_1_7b/models' \
  'qwen3_4b_sft|outputs_val/qwen3_4b/models' \
  'qwen3_8b_sft|outputs_val/qwen3_8b/models' \
  'qwen3_14b_sft|outputs_val/qwen3_14b/models'; do
  target="${spec%%|*}"; models="${spec#*|}"
  "$PY" scripts/render_bricknet_render_ldr_8views.py \
    --models-dir "$models" --output-dir "$ROOT/$target/renders_8view" \
    --renderer-bin "$RENDER_BIN" --gpu-ids 0 --workers 4
  "$PY" scripts/render_bricknet_render_ldr_8views.py \
    --models-dir "$models" --output-dir "$ROOT/$target/renders_8view" \
    --renderer-bin "$RENDER_BIN" --gpu-ids 0 --workers 4 --verify-only
  score_target "$target" "$ROOT/$target/renders_8view"
done

cd /home/jiahao/task/LlamaFactory
```

# SFT experiments

## exp0

Qwen3-VL-2B-Instruct：LlamaFactory debug 与 BrickNet-MM-VAL overfit；现有记录没有可复用的 train command，不补写。

## exp1

BrickNet-MM-VAL overfit；保留三个已有模型版本的 VAL prediction 命令。Qwen3-VL-2B-Instruct 版本为
`saves/Qwen3-VL-2B-Instruct/lora/train_exp1_qwen3vl_2b_val_ep10_bs1_ga8_lora16`；Qwen3.5-0.8B 版本为
`saves/Qwen3.5-0.8B-Thinking/lora/train_exp1_qwen35_08b_val_ep10_bs1_ga8_lora16`；Qwen3.5-2B 版本为
`saves/Qwen3.5-2B-Thinking/lora/train_exp1_qwen35_2b_val_ep10_bs1_ga8_lora16`。

```bash
cd /home/jiahao/task/LlamaFactory

conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft \
  --model_name_or_path Qwen/Qwen3-VL-2B-Instruct --preprocessing_num_workers 16 \
  --finetuning_type lora --quantization_method bnb --template qwen3_vl_nothink \
  --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 \
  --max_samples 100000 --per_device_eval_batch_size 1 --predict_with_generate True \
  --report_to none --max_new_tokens 512 --top_p 0.95 --temperature 1 \
  --output_dir saves/Qwen3-VL-2B-Instruct/lora/eval_exp1_in4096_out512_p095_k20_t1 \
  --trust_remote_code True --ddp_timeout 180000000 --do_predict True \
  --adapter_name_or_path saves/Qwen3-VL-2B-Instruct/lora/train_exp1_qwen3vl_2b_val_ep10_bs1_ga8_lora16 \
  --top_k 20

conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft \
  --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 \
  --finetuning_type lora --quantization_method bnb --template qwen3_5_nothink \
  --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 \
  --max_samples 100000 --per_device_eval_batch_size 4 --predict_with_generate True \
  --report_to none --max_new_tokens 512 --top_p 0.95 --temperature 1 \
  --output_dir saves/Qwen3.5-0.8B-Thinking/lora/eval_exp1_in4096_out512_p095_k20_t1 \
  --trust_remote_code True --ddp_timeout 180000000 --do_predict True \
  --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_exp1_qwen35_08b_val_ep10_bs1_ga8_lora16 \
  --top_k 20

conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft \
  --model_name_or_path Qwen/Qwen3.5-2B --preprocessing_num_workers 16 \
  --finetuning_type lora --quantization_method bnb --template qwen3_5_nothink \
  --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 \
  --max_samples 100000 --per_device_eval_batch_size 1 --predict_with_generate True \
  --report_to none --max_new_tokens 512 --top_p 0.95 --temperature 1 \
  --output_dir saves/Qwen3.5-2B-Thinking/lora/eval_exp1_in4096_out512_p095_k20_t1 \
  --trust_remote_code True --ddp_timeout 180000000 --do_predict True --top_k 20 \
  --adapter_name_or_path saves/Qwen3.5-2B-Thinking/lora/train_exp1_qwen35_2b_val_ep10_bs1_ga8_lora16
```

## exp1_1

Qwen3.5-2B；VAL overfit 20 epochs、LoRA rank 32，版本为
`saves/Qwen3.5-2B-Thinking/lora/train_exp1_1_qwen35_2b_val_ep20_bs4_ga8_lora32_lr1e5_schdlconstanwarm`。
现有记录只有以下两条 prediction command。

```bash
cd /home/jiahao/task/LlamaFactory

conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft \
  --model_name_or_path Qwen/Qwen3.5-2B --preprocessing_num_workers 16 \
  --finetuning_type lora --quantization_method bnb --template qwen3_5_nothink \
  --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 \
  --max_samples 100000 --per_device_eval_batch_size 1 --predict_with_generate True \
  --report_to none --max_new_tokens 512 --top_p 0.95 --temperature 1 \
  --output_dir saves/Qwen3.5-2B-Thinking/lora/eval_exp1_1_in4096_out512_p095_k20_t1 \
  --trust_remote_code True --ddp_timeout 180000000 --do_predict True --top_k 20 \
  --adapter_name_or_path saves/Qwen3.5-2B-Thinking/lora/train_exp1_1_qwen35_2b_val_ep20_bs4_ga8_lora32_lr1e5_schdlconstanwarm

conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft \
  --model_name_or_path Qwen/Qwen3.5-2B --preprocessing_num_workers 16 \
  --finetuning_type lora --quantization_method bnb --template qwen3_5_nothink \
  --flash_attn auto --dataset_dir data --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 \
  --max_samples 100000 --per_device_eval_batch_size 1 --predict_with_generate True \
  --report_to none --max_new_tokens 512 --top_p 0.9 --temperature 0.95 \
  --output_dir saves/Qwen3.5-2B-Thinking/lora/eval_exp1_1_in4097_out512_p09_t095 \
  --trust_remote_code True --ddp_timeout 180000000 --do_predict True \
  --adapter_name_or_path saves/Qwen3.5-2B-Thinking/lora/train_exp1_1_qwen35_2b_val_ep20_bs4_ga8_lora32_lr1e5_schdlconstanwarm
```

## exp2

Qwen3.5-0.8B；BrickNet-MM-SFT 10k，3 epochs，LoRA 64。

```bash
cd /home/jiahao/task/LlamaFactory

conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --do_train True \
  --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora \
  --template qwen3_5_nothink --flash_attn auto --dataset_dir data --dataset BrickNet-MM-SFT \
  --cutoff_len 4096 --learning_rate 5e-05 --num_train_epochs 3.0 --max_samples 10000 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --lr_scheduler_type cosine \
  --max_grad_norm 1.0 --logging_steps 10 --save_steps 500 --warmup_steps 0 --packing False \
  --enable_thinking False --report_to none \
  --output_dir saves/Qwen3.5-0.8B-Thinking/lora/train_exp2_qwen35_08b_sft1w_ep3_bs2_ga8_lora64 \
  --bf16 True --plot_loss True --trust_remote_code True --ddp_timeout 180000000 \
  --include_num_input_tokens_seen True --optim adamw_torch --lora_rank 64 --lora_alpha 128 \
  --lora_dropout 0 --lora_target all --freeze_vision_tower True --freeze_multi_modal_projector True \
  --image_max_pixels 589824 --image_min_pixels 1024 --video_max_pixels 65536 --video_min_pixels 256 \
  --tokenized_path .llamafactory_cache/tokenized_dataset/BrickNet-MM-SFT-10000_qwen35-08b_nothink_len4096_img589824

conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft \
  --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora \
  --quantization_method bnb --template qwen3_5_nothink --flash_attn auto --dataset_dir data \
  --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 \
  --per_device_eval_batch_size 1 --predict_with_generate True --report_to none --max_new_tokens 4096 \
  --top_p 0.95 --temperature 1 \
  --output_dir saves/Qwen3.5-0.8B-Thinking/lora/eval_exp2_in4096_out4096_p95_t1_k20 \
  --trust_remote_code True --ddp_timeout 180000000 --do_predict True --top_k 20 \
  --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_exp2_qwen35_08b_sft1w_ep3_bs2_ga8_lora64
```

## exp2_1

Qwen3.5-0.8B；BrickNet-MM-SFT 50k，3 epochs，LoRA 64。

```bash
cd /home/jiahao/task/LlamaFactory

conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --do_train True \
  --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora \
  --template qwen3_5_nothink --flash_attn auto --dataset_dir data --dataset BrickNet-MM-SFT \
  --cutoff_len 4096 --learning_rate 5e-05 --num_train_epochs 3.0 --max_samples 50000 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --lr_scheduler_type cosine \
  --max_grad_norm 1.0 --logging_steps 10 --save_steps 500 --warmup_steps 0 --packing False \
  --enable_thinking False --report_to none \
  --output_dir saves/Qwen3.5-0.8B-Thinking/lora/train_exp2_1_qwen35_08b_sft5w_ep3_bs2_ga8_lora64 \
  --bf16 True --plot_loss True --trust_remote_code True --ddp_timeout 180000000 \
  --include_num_input_tokens_seen True --optim adamw_torch --lora_rank 64 --lora_alpha 128 \
  --lora_dropout 0 --lora_target all --freeze_vision_tower True --freeze_multi_modal_projector True \
  --image_max_pixels 589824 --image_min_pixels 1024 --video_max_pixels 65536 --video_min_pixels 256 \
  --tokenized_path .llamafactory_cache/tokenized_dataset/BrickNet-MM-SFT-50000_qwen35-08b_nothink_len4096_img589824

conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft \
  --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora \
  --quantization_method bnb --template qwen3_5_nothink --flash_attn auto --dataset_dir data \
  --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 \
  --per_device_eval_batch_size 1 --predict_with_generate True --report_to none --max_new_tokens 4096 \
  --top_p 0.95 --temperature 1 \
  --output_dir saves/Qwen3.5-0.8B-Thinking/lora/eval_exp2_1_in4096_out4096_p95_t1_k20 \
  --trust_remote_code True --ddp_timeout 180000000 --do_predict True --top_k 20 \
  --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_exp2_1_qwen35_08b_sft5w_ep3_bs2_ga8_lora64
```

## exp2_2

Qwen3.5-0.8B；BrickNet-MM-SFT 全量，3 epochs，LoRA 64，版本为
`saves/Qwen3.5-0.8B-Thinking/lora/train_exp2_2_qwen35_08b_sft_ep3_bs2_ga8_lora64`。
现有记录没有 exp2_2 eval command，不补写。

```bash
cd /home/jiahao/task/LlamaFactory

conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --do_train True \
  --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora \
  --template qwen3_5_nothink --flash_attn auto --dataset_dir data --dataset BrickNet-MM-SFT \
  --cutoff_len 4096 --learning_rate 5e-05 --num_train_epochs 3.0 --max_samples 1000000 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --lr_scheduler_type cosine \
  --max_grad_norm 1.0 --logging_steps 10 --save_steps 2000 --warmup_steps 0 --packing False \
  --enable_thinking False --report_to none \
  --output_dir saves/Qwen3.5-0.8B-Thinking/lora/train_exp2_2_qwen35_08b_sft_ep3_bs2_ga8_lora64 \
  --bf16 True --plot_loss True --trust_remote_code True --ddp_timeout 180000000 \
  --include_num_input_tokens_seen True --optim adamw_torch --lora_rank 64 --lora_alpha 128 \
  --lora_dropout 0 --lora_target all --freeze_vision_tower True --freeze_multi_modal_projector True \
  --image_max_pixels 589824 --image_min_pixels 1024 --video_max_pixels 65536 --video_min_pixels 256 \
  --tokenized_path .llamafactory_cache/tokenized_dataset/BrickNet-MM-SFT_qwen35-08b_nothink_len4096_img589824
```

## exp3

Qwen3.5-0.8B-PT；从 PT-exp0 初始化，BrickNet-MM-SFT 10k，3 epochs，LoRA 64。

```bash
cd /home/jiahao/task/LlamaFactory

conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --do_train True \
  --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora \
  --template qwen3_5_nothink --flash_attn auto --dataset_dir data --dataset BrickNet-MM-SFT \
  --cutoff_len 4096 --learning_rate 5e-05 --num_train_epochs 3.0 --max_samples 10000 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --lr_scheduler_type cosine \
  --max_grad_norm 1.0 --logging_steps 10 --save_steps 500 --warmup_steps 0 --packing False \
  --enable_thinking False --report_to none \
  --output_dir saves/Qwen3.5-0.8B-Thinking/lora/train_exp3_qwen35_08b_pt_sft1w_ep3_bs2_ga8_lora64 \
  --bf16 True --plot_loss True --trust_remote_code True --ddp_timeout 180000000 \
  --include_num_input_tokens_seen True --optim adamw_torch --lora_rank 64 --lora_alpha 128 \
  --lora_dropout 0 --lora_target all --freeze_vision_tower True --freeze_multi_modal_projector True \
  --image_max_pixels 589824 --image_min_pixels 1024 --video_max_pixels 65536 --video_min_pixels 256 \
  --tokenized_path .llamafactory_cache/tokenized_dataset/BrickNet-MM-SFT-10000_qwen35-08b_nothink_len4096_img589824 \
  --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64 \
  --create_new_adapter

conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft \
  --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora \
  --quantization_method bnb --template qwen3_5_nothink --flash_attn auto --dataset_dir data \
  --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 \
  --per_device_eval_batch_size 1 --predict_with_generate True --report_to none --max_new_tokens 4096 \
  --top_p 0.95 --temperature 1 \
  --output_dir saves/Qwen3.5-0.8B-Thinking/lora/eval_exp3_PT_in4096_out4096_p95_t1_k20 \
  --trust_remote_code True --ddp_timeout 180000000 --do_predict True --top_k 20 \
  --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64

conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft \
  --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora \
  --quantization_method bnb --template qwen3_5_nothink --flash_attn auto --dataset_dir data \
  --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 \
  --per_device_eval_batch_size 1 --predict_with_generate True --report_to none --max_new_tokens 4096 \
  --top_p 0.95 --temperature 1 \
  --output_dir saves/Qwen3.5-0.8B-Thinking/lora/eval_exp3_in4096_out4096_p95_t1_k20 \
  --trust_remote_code True --ddp_timeout 180000000 --do_predict True --top_k 20 \
  --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64,saves/Qwen3.5-0.8B-Thinking/lora/train_exp3_qwen35_08b_pt_sft1w_ep3_bs2_ga8_lora64
```

## exp3_0_1

Qwen3.5-0.8B-PT；从 PT-exp0 初始化，BrickNet-MM-SFT 10k，10 epochs，LoRA 64。

```bash
cd /home/jiahao/task/LlamaFactory

conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --do_train True \
  --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora \
  --template qwen3_5_nothink --flash_attn auto --dataset_dir data --dataset BrickNet-MM-SFT \
  --cutoff_len 4096 --learning_rate 5e-05 --num_train_epochs 10 --max_samples 10000 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --lr_scheduler_type cosine \
  --max_grad_norm 1.0 --logging_steps 10 --save_steps 500 --warmup_steps 0 --packing False \
  --enable_thinking False --report_to none \
  --output_dir saves/Qwen3.5-0.8B-Thinking/lora/train_exp3_0_1_qwen35_08b_pt_sft1w_ep10_bs2_ga8_lora64 \
  --bf16 True --plot_loss True --trust_remote_code True --ddp_timeout 180000000 \
  --include_num_input_tokens_seen True --optim adamw_torch --lora_rank 64 --lora_alpha 128 \
  --lora_dropout 0 --lora_target all --freeze_vision_tower True --freeze_multi_modal_projector True \
  --image_max_pixels 589824 --image_min_pixels 1024 --video_max_pixels 65536 --video_min_pixels 256 \
  --tokenized_path .llamafactory_cache/tokenized_dataset/BrickNet-MM-SFT-10000_qwen35-08b_nothink_len4096_img589824 \
  --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64 \
  --create_new_adapter

conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft \
  --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora \
  --quantization_method bnb --template qwen3_5_nothink --flash_attn auto --dataset_dir data \
  --eval_dataset BrickNet-MM-VAL --cutoff_len 4096 --max_samples 100000 \
  --per_device_eval_batch_size 1 --predict_with_generate True --report_to none --max_new_tokens 4096 \
  --top_p 0.95 --temperature 1 \
  --output_dir saves/Qwen3.5-0.8B-Thinking/lora/eval_exp3_0_1_in4096_out4096_p95_t1_k20 \
  --trust_remote_code True --ddp_timeout 180000000 --do_predict True --top_k 20 \
  --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64,saves/Qwen3.5-0.8B-Thinking/lora/train_exp3_0_1_qwen35_08b_pt_sft1w_ep10_bs2_ga8_lora64

cd /home/jiahao/task/BrickNet
/home/jiahao/miniconda3/envs/bricknet/bin/python scripts/evaluate_experiment.py \
  --predictions /home/jiahao/task/LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/eval_exp3_0_1_in4096_out4096_p95_t1_k20/generated_predictions.jsonl \
  --text-metrics /home/jiahao/task/LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/eval_exp3_0_1_in4096_out4096_p95_t1_k20/predict_results.json \
  --output-dir outputs_val/qwen35_08b/eval_exp3_0_1_in4096_out4096_p95_t1_k20 \
  --prompts-file data/bricknet_datasets/captions_val.jsonl

cd /home/jiahao/task/LlamaFactory
```

## exp3_1

Qwen3.5-0.8B-PT；从 PT-exp0 初始化，BrickNet-MM-SFT 50k，3 epochs，LoRA 64；训练版本为
`saves/Qwen3.5-0.8B-Thinking/lora/train_exp3_1_qwen35_08b_pt_sft5w_ep3_bs2_ga8_lora64`。

```bash
cd /home/jiahao/task/LlamaFactory

conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --do_train True \
  --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora \
  --template qwen3_5_nothink --flash_attn auto --dataset_dir data --dataset BrickNet-MM-SFT \
  --cutoff_len 4096 --learning_rate 5e-05 --num_train_epochs 3.0 --max_samples 50000 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --lr_scheduler_type cosine \
  --max_grad_norm 1.0 --logging_steps 10 --save_steps 1000 --warmup_steps 0 --packing False \
  --enable_thinking False --report_to none \
  --output_dir saves/Qwen3.5-0.8B-Thinking/lora/train_exp3_1_qwen35_08b_pt_sft5w_ep3_bs2_ga8_lora64 \
  --bf16 True --plot_loss True --trust_remote_code True --ddp_timeout 180000000 \
  --include_num_input_tokens_seen True --optim adamw_torch --lora_rank 64 --lora_alpha 128 \
  --lora_dropout 0 --lora_target all --freeze_vision_tower True --freeze_multi_modal_projector True \
  --image_max_pixels 589824 --image_min_pixels 1024 --video_max_pixels 65536 --video_min_pixels 256 \
  --tokenized_path .llamafactory_cache/tokenized_dataset/BrickNet-MM-SFT-50000_qwen35-08b_nothink_len4096_img589824 \
  --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64 \
  --create_new_adapter

conda run -n llamafactory --no-capture-output llamafactory-cli train \
  examples/train_lora/qwen35_08b_bricknet_mm_exp3_1_predict.yaml

cd /home/jiahao/task/BrickNet
/home/jiahao/miniconda3/envs/bricknet/bin/python scripts/evaluate_experiment.py \
  --predictions /home/jiahao/task/LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/eval_exp3_1_in4096_out4096_p95_t1_k20/generated_predictions.jsonl \
  --text-metrics /home/jiahao/task/LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/eval_exp3_1_in4096_out4096_p95_t1_k20/predict_results.json \
  --output-dir outputs_val/qwen35_08b/eval_exp3_1_in4096_out4096_p95_t1_k20 \
  --prompts-file data/bricknet_datasets/captions_val.jsonl --render-jobs 8 --eval-workers 8 --eval-batch-size 8

cd /home/jiahao/task/LlamaFactory
jq -c '{response: .predict, label: .label}' \
  saves/Qwen3.5-0.8B-Thinking/lora/eval_exp3_1_in4096_out4096_p95_t1_k20/generated_predictions.jsonl \
  > /home/jiahao/task/BrickNet/outputs_val/qwen35_08b/eval_exp3_1_in4096_out4096_p95_t1_k20/alignment_input.jsonl

cd /home/jiahao/task/ms-swift
/home/jiahao/miniconda3/envs/bricknet/bin/python \
  examples/train/grpo/plugin/bricknet/evaluate_experiment.py alignment-worker \
  --results /home/jiahao/task/BrickNet/outputs_val/qwen35_08b/eval_exp3_1_in4096_out4096_p95_t1_k20/alignment_input.jsonl \
  --dataset /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM/sharegpt/BrickNet-MM_VAL.jsonl \
  --scored /home/jiahao/task/BrickNet/outputs_val/qwen35_08b/eval_exp3_1_in4096_out4096_p95_t1_k20/scored.jsonl \
  --metrics-json /home/jiahao/task/BrickNet/outputs_val/qwen35_08b/eval_exp3_1_in4096_out4096_p95_t1_k20/metrics.json \
  --metrics-md /home/jiahao/task/BrickNet/outputs_val/qwen35_08b/eval_exp3_1_in4096_out4096_p95_t1_k20/metrics.md \
  --output /home/jiahao/task/BrickNet/outputs_val/qwen35_08b/eval_exp3_1_in4096_out4096_p95_t1_k20/alignment.jsonl \
  --bricknet-root /home/jiahao/task/BrickNet

cd /home/jiahao/task/LlamaFactory
```

## exp3_2

Qwen3.5-0.8B-PT；从 PT-exp0 初始化，BrickNet-MM-SFT 全量，3 epochs，LoRA 64，版本为
`saves/Qwen3.5-0.8B-Thinking/lora/train_exp3_2_qwen35_08b_pt_sft_ep3_bs2_ga8_lora64`。
现有记录没有 exp3_2 eval command，不补写。

```bash
cd /home/jiahao/task/LlamaFactory

conda run -n llamafactory --no-capture-output llamafactory-cli train --stage sft --do_train True \
  --model_name_or_path Qwen/Qwen3.5-0.8B --preprocessing_num_workers 16 --finetuning_type lora \
  --template qwen3_5_nothink --flash_attn auto --dataset_dir data --dataset BrickNet-MM-SFT \
  --cutoff_len 4096 --learning_rate 5e-05 --num_train_epochs 3.0 --max_samples 1000000 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --lr_scheduler_type cosine \
  --max_grad_norm 1.0 --logging_steps 10 --save_steps 2000 --warmup_steps 0 --packing False \
  --enable_thinking False --report_to none \
  --output_dir saves/Qwen3.5-0.8B-Thinking/lora/train_exp3_2_qwen35_08b_pt_sft_ep3_bs2_ga8_lora64 \
  --bf16 True --plot_loss True --trust_remote_code True --ddp_timeout 180000000 \
  --include_num_input_tokens_seen True --optim adamw_torch --lora_rank 64 --lora_alpha 128 \
  --lora_dropout 0 --lora_target all --freeze_vision_tower True --freeze_multi_modal_projector True \
  --image_max_pixels 589824 --image_min_pixels 1024 --video_max_pixels 65536 --video_min_pixels 256 \
  --tokenized_path .llamafactory_cache/tokenized_dataset/BrickNet-MM-SFT_qwen35-08b_nothink_len4096_img589824 \
  --adapter_name_or_path saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64 \
  --create_new_adapter
```

# Stage 2 experiments

PT-exp1 mixed text+mm 初始化

## exp4

NonThinking-Control VAL511 overfit；配置为
`examples/train_lora/qwen35_08b_bricknet_stage2_exp4_nonthinking_control_val511.yaml`。

```bash
cd /home/jiahao/task/LlamaFactory
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_stage2_sft.py \
  --action train --variant nonthinking-control --scale overfit511 \
  --execute --stage0-gate-approved
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_stage2_sft.py \
  --action predict --variant nonthinking-control --scale overfit511 \
  --execute --stage0-gate-approved
conda run -n llamafactory --no-capture-output python scripts/evaluate_bricknet_stage2.py \
  --experiment exp4 --execute
```

## exp4_1

Thinking-Hard VAL511 overfit；配置为
`examples/train_lora/qwen35_08b_bricknet_stage2_exp4_1_thinking_hard_val511.yaml`。

```bash
cd /home/jiahao/task/LlamaFactory
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_stage2_sft.py \
  --action train --variant thinking-hard --scale overfit511 \
  --execute --stage0-gate-approved
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_stage2_sft.py \
  --action predict --variant thinking-hard --scale overfit511 \
  --execute --stage0-gate-approved
conda run -n llamafactory --no-capture-output python scripts/evaluate_bricknet_stage2.py \
  --experiment exp4_1 --execute
```

## exp4_2

NonThinking-Control 10k；action-only 主线配置为
`examples/train_lora/qwen35_08b_bricknet_stage2_exp4_2_nonthinking_control_10k.yaml`。

```bash
cd /home/jiahao/task/LlamaFactory
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_stage2_sft.py \
  --action train --variant nonthinking-control --scale 10k \
  --execute --stage0-gate-approved --overfit-gate-approved
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_stage2_sft.py \
  --action predict --variant nonthinking-control --scale 10k \
  --execute --stage0-gate-approved --overfit-gate-approved
conda run -n llamafactory --no-capture-output python scripts/evaluate_bricknet_stage2.py \
  --experiment exp4_2 --execute
```

## exp4_3

Thinking-Hard 10k；配置为
`examples/train_lora/qwen35_08b_bricknet_stage2_exp4_3_thinking_hard_10k.yaml`。

```bash
cd /home/jiahao/task/LlamaFactory
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_stage2_sft.py \
  --action train --variant thinking-hard --scale 10k \
  --execute --stage0-gate-approved --overfit-gate-approved
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_stage2_sft.py \
  --action predict --variant thinking-hard --scale 10k \
  --execute --stage0-gate-approved --overfit-gate-approved
conda run -n llamafactory --no-capture-output python scripts/evaluate_bricknet_stage2.py \
  --experiment exp4_3 --execute
```

## exp4_3_1

Thinking-Hard-V2 Lean-State 10k；配置为
`examples/train_lora/qwen35_08b_bricknet_stage2_exp4_3_1_thinking_hard_v2_lean_state_10k.yaml`，保持普通监督文本且关闭原生 thinking。

```bash
cd /home/jiahao/task/LlamaFactory
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_stage2_sft.py \
  --action train --variant thinking-hard-v2-lean-state --scale 10k \
  --overfit-gate-approved
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_stage2_sft.py \
  --action train --variant thinking-hard-v2-lean-state --scale 10k \
  --execute --stage0-gate-approved --overfit-gate-approved
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_stage2_sft.py \
  --action predict --variant thinking-hard-v2-lean-state --scale 10k
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_stage2_sft.py \
  --action predict --variant thinking-hard-v2-lean-state --scale 10k \
  --execute --stage0-gate-approved
conda run -n llamafactory --no-capture-output python scripts/evaluate_bricknet_stage2.py \
  --experiment exp4_3_1 --execute
```

# Stage 3 experiments

人工标注semantic/cue语义

## exp5

Thinking-Semantic 10k dormant配置；train/predict YAML 分别为
`examples/train_lora/qwen35_08b_bricknet_stage3_exp5_thinking_semantic_10k.yaml` 与
`examples/train_lora/qwen35_08b_bricknet_stage3_exp5_thinking_semantic_predict.yaml`。
只保留现有 train dry-run，不将其写成已训练。

```bash
cd /home/jiahao/task/LlamaFactory
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_stage3_sft.py \
  --action train
```

# PT-exp2 experiments

## text8m

Qwen3.5-0.8B text-only PT 250k；使用现有 gate-protected launcher。

```bash
cd /home/jiahao/task/LlamaFactory
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2.py \
  --action train --run text8m --gpus 1 --execute
```

## mm e1/e2/e3

MM consolidation 使用 v2-no-meta YAML；e1 从 text8m，e2 从 e1，e3 从 e2 继续。

```bash
cd /home/jiahao/task/LlamaFactory

conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_mm_v2.py \
  --gpus 1 --action train --run mm-e1 --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_mm_v2.py \
  --gpus 1 --action predict --run mm-e1 --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_mm_v2.py \
  --gpus 1 --action evaluate --run mm-e1 --execute

conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_mm_v2.py \
  --gpus 1 --action train --run mm-e2 --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_mm_v2.py \
  --gpus 1 --action predict --run mm-e2 --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_mm_v2.py \
  --gpus 1 --action evaluate --run mm-e2 --execute

conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_mm_v2.py \
  --gpus 1 --action train --run mm-e3 --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_mm_v2.py \
  --gpus 1 --action predict --run mm-e3 --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_mm_v2.py \
  --gpus 1 --action evaluate --run mm-e3 --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_mm_v2.py \
  --action select-final --run mm-e3 --execute --approve
```

## rowbal-cont3

从 text8m 250k endpoint 连续训练三 epoch；prediction/evaluation 按 ep1、ep2、ep3 使用现有 launcher。

```bash
cd /home/jiahao/task/LlamaFactory

conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3.py \
  --action prepare-audits --processor-workers 4 --processor-chunksize 8 --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3.py \
  --action prepare-cache --cache-num-proc 1 --cache-batch-size 10000 --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3.py \
  --action train --gpus 1 --execute

conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3.py \
  --action predict --run ep1 --gpus 1 --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3.py \
  --action evaluate --run ep1 --gpus 1 --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3.py \
  --action predict --run ep2 --gpus 1 --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3.py \
  --action evaluate --run ep2 --gpus 1 --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3.py \
  --action predict --run ep3 --gpus 1 --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3.py \
  --action evaluate --run ep3 --gpus 1 --execute
```

## exp4_4 / exp4_7

PT-exp2-v2 下游：两个 10k Stage-2 变体分别为 NonThinking-Control 与 Lean-State。

```bash
cd /home/jiahao/task/LlamaFactory

conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_v2_downstream.py \
  --run exp4_4 --action train --gpus 1 --pt-exp2-v2-approved --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_v2_downstream.py \
  --run exp4_4 --action predict --gpus 1 --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_v2_downstream.py \
  --run exp4_4 --action evaluate --gpus 1 --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_v2_downstream.py \
  --run exp4_7 --action train --gpus 1 --pt-exp2-v2-approved --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_v2_downstream.py \
  --run exp4_7 --action predict --gpus 1 --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_v2_downstream.py \
  --run exp4_7 --action evaluate --gpus 1 --execute
```

## exp4_4_1 / exp4_7_1 (text-100k)

Text-only 下游从 text8m checkpoint-100000 开始；训练与 prediction 只能使用这两个现有 YAML，不能改绑 MM alias。

```bash
cd /home/jiahao/task/LlamaFactory

CUDA_VISIBLE_DEVICES=0 conda run -n llamafactory --no-capture-output llamafactory-cli train \
  examples/train_lora/qwen35_08b_bricknet_stage2_exp4_4_1_nonthinking_control_10k_pt_exp2_100k.yaml \
  gradient_accumulation_steps=16
CUDA_VISIBLE_DEVICES=0 conda run -n llamafactory --no-capture-output llamafactory-cli train \
  examples/train_lora/qwen35_08b_bricknet_stage2_exp4_4_1_nonthinking_control_predict_pt_exp2_100k.yaml
conda run -n llamafactory --no-capture-output python scripts/evaluate_bricknet_stage2.py \
  --experiment exp4_4_1 --execute

CUDA_VISIBLE_DEVICES=1 conda run -n llamafactory --no-capture-output llamafactory-cli train \
  examples/train_lora/qwen35_08b_bricknet_stage2_exp4_7_1_thinking_hard_10k_pt_exp2_100k.yaml \
  gradient_accumulation_steps=16
CUDA_VISIBLE_DEVICES=1 conda run -n llamafactory --no-capture-output llamafactory-cli train \
  examples/train_lora/qwen35_08b_bricknet_stage2_exp4_7_1_thinking_hard_predict_pt_exp2_100k.yaml
conda run -n llamafactory --no-capture-output python scripts/evaluate_bricknet_stage2.py \
  --experiment exp4_7_1 --execute
```

## exp4_4_2 / exp4_7_2 (text-250k)

Text-only 下游直接从 text8m 250k adapter 开始；train 受 `--text250k-approved` gate 保护。

```bash
cd /home/jiahao/task/LlamaFactory

conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_text250k_downstream.py \
  --run exp4_4_2 --action train --gpus 1 --text250k-approved --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_text250k_downstream.py \
  --run exp4_4_2 --action predict --gpus 1 --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_text250k_downstream.py \
  --run exp4_4_2 --action evaluate --gpus 1 --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_text250k_downstream.py \
  --run exp4_7_2 --action train --gpus 1 --text250k-approved --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_text250k_downstream.py \
  --run exp4_7_2 --action predict --gpus 1 --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_text250k_downstream.py \
  --run exp4_7_2 --action evaluate --gpus 1 --execute
```

## exp4_4_3 / exp4_7_3 (rowbal downstream)

下游直接绑定 rowbal-cont3 ep3 endpoint；train 受 `--rowbal-ep3-approved` gate 保护。

```bash
cd /home/jiahao/task/LlamaFactory

conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3_downstream.py \
  --run exp4_4_3 --action train --gpus 0 --rowbal-ep3-approved --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3_downstream.py \
  --run exp4_4_3 --action predict --gpus 0 --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3_downstream.py \
  --run exp4_4_3 --action evaluate --gpus 0 --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3_downstream.py \
  --run exp4_7_3 --action train --gpus 0 --rowbal-ep3-approved --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3_downstream.py \
  --run exp4_7_3 --action predict --gpus 0 --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3_downstream.py \
  --run exp4_7_3 --action evaluate --gpus 0 --execute
```

## dormant exp4_5 / exp4_6

PT-exp2 downstream 扩容仍 dormant；exp4_5 为 50k，exp4_6 为 all。以下是已有 launcher 入口，不能视为已执行。

```bash
cd /home/jiahao/task/LlamaFactory

conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2.py \
  --gpus 0 1 --action train --run exp4_5 --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2.py \
  --gpus 0 --action predict --run exp4_5 --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2.py \
  --gpus 0 --action evaluate --run exp4_5 --execute

conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2.py \
  --gpus 0 1 --action train --run exp4_6 --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2.py \
  --gpus 0 --action predict --run exp4_6 --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2.py \
  --gpus 0 --action evaluate --run exp4_6 --execute
```

# Stage 5–8 active entries

Stage 5 的现有 verifier 结果由后续 controller launcher 读取。Stage 6–7 当前只允许 compact A1；B1/V1/V2/A0 不再提供新的正式启动入口，A1 也不能写成已完成。

## A1-only controller

以下是当前 compact A1 的唯一 controller 入口；失败 direct child 只回传紧凑摘要，prompt 上限按 processor 实际 input tokens 处理。

```bash
cd /home/jiahao/task/LlamaFactory
cd /home/jiahao/task/BrickNet

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  /home/jiahao/miniconda3/envs/bricknet/bin/python scripts/run_bricknet_agentic_a1_compact_inference.py \
  --input /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Reasoning/validation/datasets/BrickNet-Stage2-NonThinking-Control-VAL512-Eval.jsonl \
  --output /home/jiahao/task/BrickNet/outputs_val/qwen35_08b/agentic_exp4_2_a1/controller_audit.jsonl \
  --mode a1-feedback-search --backend hf --prompt-protocol exp4_2-stepwise --seed 42 \
  --model Qwen/Qwen3.5-0.8B \
  --model-revision "$(jq -r '.inference_contract.model.revision' /home/jiahao/task/BrickNet/configs/agentic_stage67_exp4_2.json)" \
  --adapter /home/jiahao/task/LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp1_qwen35_08b_bricknet_text270k_mmpt135k_ep1_bs2_ga8_lora64 \
  --adapter /home/jiahao/task/LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/train_exp4_2_qwen35_08b_mixedpt_stage2_nonthinking_control_10k_ep3_bs1_ga16_lora64_len16384 \
  --contract-config /home/jiahao/task/BrickNet/configs/agentic_stage67_exp4_2.json \
  --candidates-per-round 8 --max-rounds-per-state 4 --max-backtrack-depth 3 \
  --stage5-report /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT/stage5/Stage5-full-replay-report.json
```

## A1 evaluation

先做只读 preflight，再只评测 A1 的 final/raw 层；不重跑旧 controller 组。

```bash
cd /home/jiahao/task/LlamaFactory
cd /home/jiahao/task/BrickNet

PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  scripts/evaluate_bricknet_agentic_stage67.py --action preflight
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  scripts/evaluate_bricknet_agentic_stage67.py --action all --runs a1
BRICKNET_DATA=/home/jiahao/task/BrickNet/data/bricknet_datasets \
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  scripts/evaluate_bricknet_agentic_stage67.py \
  --action evaluate --runs a1 --layers final raw --execute --force
```

## Stage 8 R1-S

R1-S 是 success-only Act SFT；先走 smoke，再走 10k。比较器使用独立的 `S8-ZS-Greedy` 入口。

```bash
cd /home/jiahao/task/LlamaFactory

conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_stage8_act_sft.py \
  --run R1-S --scale 64 --refresh-initialization-audit
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_stage8_act_sft.py \
  --run R1-S --scale 64 --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_stage8_act_sft.py \
  --run R1-S --scale 10k --refresh-initialization-audit
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_stage8_act_sft.py \
  --run R1-S --scale 10k --execute

conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_stage8_controller_eval.py \
  --run S8-ZS-Greedy
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_stage8_controller_eval.py \
  --run S8-ZS-Greedy --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_stage8_controller_eval.py \
  --run R1-S
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_stage8_controller_eval.py \
  --run R1-S --execute

cd /home/jiahao/task/BrickNet
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  scripts/evaluate_bricknet_agentic_stage8.py --treatment R1-S --action all
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  scripts/evaluate_bricknet_agentic_stage8.py --treatment R1-S --action all --execute
cd /home/jiahao/task/LlamaFactory
```

## Stage 8 R1-C

R1-C 只在真实 rejection/correction 数据与 R1-S matched-token gate 通过后开放。

```bash
cd /home/jiahao/task/LlamaFactory

conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_stage8_act_sft.py \
  --run R1-C --scale 10k --refresh-initialization-audit
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_stage8_act_sft.py \
  --run R1-C --scale 10k --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_stage8_controller_eval.py \
  --run R1-C
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_stage8_controller_eval.py \
  --run R1-C --execute

cd /home/jiahao/task/BrickNet
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  scripts/evaluate_bricknet_agentic_stage8.py --treatment R1-C --action all
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  scripts/evaluate_bricknet_agentic_stage8.py --treatment R1-C --action all --execute
cd /home/jiahao/task/LlamaFactory
```

## Stage 8 R1-B

R1-B 只在真实 rollback transitions、R1-C gate 与两遍 token gate 通过后开放；不把它写成已完成。

```bash
cd /home/jiahao/task/LlamaFactory

conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_stage8_act_sft.py \
  --run R1-B --scale 10k --refresh-initialization-audit
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_stage8_act_sft.py \
  --run R1-B --scale 10k --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_stage8_controller_eval.py \
  --run S8-ZS-DFS
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_stage8_controller_eval.py \
  --run S8-ZS-DFS --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_stage8_controller_eval.py \
  --run R1-B
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_stage8_controller_eval.py \
  --run R1-B --execute

cd /home/jiahao/task/BrickNet
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  scripts/evaluate_bricknet_agentic_stage8.py --treatment R1-B --action all
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  scripts/evaluate_bricknet_agentic_stage8.py --treatment R1-B --action all --execute
cd /home/jiahao/task/LlamaFactory
```

# Official data v2 selection

官方 data v2 只在训练或 prediction 命令中显式选择；数据构造与检查见 [data_preprocess 文档](../BrickNet/data_preprocess/README_zh.md)。

```bash
cd /home/jiahao/task/LlamaFactory

# v2 SFT training example
conda run -n llamafactory --no-capture-output llamafactory-cli train \
  examples/train_lora/qwen35_08b_bricknet_stage2_exp4_2_nonthinking_control_10k.yaml \
  bricknet_dataset_version=v2

# v2 VAL prediction example
conda run -n llamafactory --no-capture-output llamafactory-cli train \
  examples/train_lora/qwen35_08b_bricknet_stage2_exp4_2_nonthinking_control_predict.yaml \
  bricknet_dataset_version=v2
```
