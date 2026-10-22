# Experiment record

前置阅读：[Planner README](../BrickNet/BrickNet-MM%20Agentic%20LEGO%20Planner/README.md) → [Constructor Plan](../BrickNet/BrickNet-MM%20Agentic%20LEGO%20Planner/Constructor%20Plan.md)。结果账本：[experiment_results.md](experiment_results.md)。

除非命令块明确切换仓库，均先从 `/home/jiahao/task/LlamaFactory` 执行；LlamaFactory 命令使用
`conda run -n llamafactory --no-capture-output`。

# PT experiments

## BrickNet PT official media render（已验证完成）

本节登记 PT 官方媒体层的完成证据和历史运行入口。正式 run
`pt-elastic-cuda1-20260914T142134Z-003` 已完成 135,051/135,051 rows：rendered 58,101、
skipped 76,950、failed 0；135,051 个 raw row 目录中的 1,080,408 张 PNG 和 135,051 张 collage 已
通过 strict verify，pass marker 计数为 1。旧双卡 run
`pt-elastic-20260913T200555Z-633995` 已于 2026-09-14 22:05+08 安全停止，迁移前为
`76900/135051`（rendered `13601`、skipped `63299`、failed `0`）。

completion gate 于 `2026-09-19T13:49:08Z` 生成且为 `PASS`：

```text
path:            /data/jiahao/task/BrickNet/outputs_gt/.bricknet_render_v2_official_pt/pt_completion_gate.json
gate file SHA:   9a0796acf72ea3c183bc7850dceaf3e6fb90152b56dde46ab5dc1432f7f88f84
gate self SHA:   8441671258afa2027cd996b521456ddcfd38c44cc19b5b321c3c4ce2fa41cd60
verify log SHA:  bdf3e5d85a8a430db7e52d695c5719d9ec7cca73d295bef8fff1b54dee293c73
```

`downstream_guard=PASS`，5 个受保护文件的 hash 均已验证；
`projection_activated=false`、`dataset_mapping_modified=false`、
`new_dataset_version_created=false`。因此当前只完成独立媒体层，尚未接入训练。

已验收运行的冻结入口为：

- 配置：`/data/jiahao/task/BrickNet/configs/bricknet_mm_image_v2_official_pt_elastic_runtime_cuda1.json`
  （SHA-256 `ac4dca232592b4e2d99d83f4218e691cc1f2873084fba7a09b2c0f9d020c6cb0`）。
- Identity 记录必须区分 adoption 与当前文件：run003 的 adoption-commit event artifact SHA-256 为
  `35dea999f3518dd31b1ae659105064b9cf8ccf7f40e645b1b6861c5a28f2d594`；启动后的 `ensure_identity`
  已规范化重写 `pt_identity.json`，当前 normalized identity raw SHA-256 为
  `7f84ed4e9ed807cb6dd985ddd6749b6dc0dd320dda70da0b07f7ff13a0e2e26d`，`identity_mismatch_fields` 已移除。
- 拓扑：物理 CUDA `1` 映射为进程逻辑 `cuda:0`；物理 CUDA `0` 不使用；identity cap `7`，运行时
  active admission 动态为 `1–7`。
- 运行语义：官方 `CYCLES + OPTIX`；保留 idle balloon、heavy/extreme row 保护（阈值 `90/100`、
  hold expiry `900s`）、CUDA/OptiX 显存压力最多一次重试、磁盘低水位暂停/恢复、原子 row 输出和
  `skip_existing` 断点恢复。
- 进程边界：supervisor/governor 只归因、停止自己启动且身份校验通过的进程组及 balloon；不得向其他
  用户进程发信号。PT 媒体仍是未激活层，图片路径映射保持不变。

以下为历史正式启动记录；任务已完成，不得重复启动该 `run_id`：

```bash
cd /data/jiahao/task/BrickNet

PT_ELASTIC_RUN_ID="pt-elastic-cuda1-20260914T142134Z-003"
PT_ELASTIC_LOG="/data/jiahao/task/LlamaFactory/tmp_bash/pt_official_elastic_${PT_ELASTIC_RUN_ID}.nohup.log"
nohup setsid /home/jiahao/miniconda3/envs/bricknet/bin/python \
  scripts/supervise_bricknet_pt_elastic.py \
  --run-id "$PT_ELASTIC_RUN_ID" \
  --config configs/bricknet_mm_image_v2_official_pt_elastic_runtime_cuda1.json \
  >"$PT_ELASTIC_LOG" 2>&1 </dev/null &
printf 'run_id=%s supervisor_pid=%s log=%s\n' "$PT_ELASTIC_RUN_ID" "$!" "$PT_ELASTIC_LOG"
```

历史运行的增量监控以 `current_run.json` 为权威入口；完成结论以 strict verify 和 gate 为准：

```bash
cd /data/jiahao/task/BrickNet
PT_ELASTIC_BASE=outputs_gt/.bricknet_render_v2_official_pt/elastic_runtime
PT_ELASTIC_RUN_ID="$(jq -er '.run_id' "$PT_ELASTIC_BASE/current_run.json")"
PT_ELASTIC_RUNTIME="$PT_ELASTIC_BASE/$PT_ELASTIC_RUN_ID"

jq '{run_id,state,result,stage,owned_processes,updated_utc}' \
  "$PT_ELASTIC_RUNTIME/supervisor.status.json"
jq '{state,control_epoch,measurements_by_gpu,updated_utc}' \
  "$PT_ELASTIC_RUNTIME/governor.status.json"
jq '{accepted_epoch,desired_active_by_gpu,inflight_by_gpu,terminal,updated_utc}' \
  "$PT_ELASTIC_RUNTIME/pt_runtime_control.ack.json"
jq '{split,rows_requested,stats,timing,workers_per_gpu,gpu_ids}' \
  outputs_gt/.bricknet_render_v2_official_pt/pt_progress.json
tail -n 100 "$PT_ELASTIC_RUNTIME/logs/renderer.log"
```

受控停止必须只对 `current_run.json` 中、且经 `ps` 核对命令行包含同一 run ID 的 supervisor PID 发送
`SIGTERM`；不要直接终止 renderer、governor 或 balloon：

```bash
cd /data/jiahao/task/BrickNet
PT_ELASTIC_BASE=outputs_gt/.bricknet_render_v2_official_pt/elastic_runtime
PT_ELASTIC_RUN_ID="$(jq -er '.run_id' "$PT_ELASTIC_BASE/current_run.json")"
PT_ELASTIC_RUNTIME="$PT_ELASTIC_BASE/$PT_ELASTIC_RUN_ID"
SUPERVISOR_PID="$(jq -er '.pid' "$PT_ELASTIC_BASE/current_run.json")"
ps -p "$SUPERVISOR_PID" -o pid=,args=
# 仅在上一行确认是 supervise_bricknet_pt_elastic.py 且包含 "$PT_ELASTIC_RUN_ID" 后执行：
kill -TERM "$SUPERVISOR_PID"
while kill -0 "$SUPERVISOR_PID" 2>/dev/null; do sleep 2; done
jq '{run_id,state,result,stage,updated_utc}' "$PT_ELASTIC_RUNTIME/supervisor.status.json"
```

以下断点续跑命令仅用于未来确认产物损坏、需要重新物化且重新批准之后；不得复用旧 run ID：

```bash
cd /data/jiahao/task/BrickNet
PT_ELASTIC_RUN_ID="pt-elastic-cuda1-$(date -u +%Y%m%dT%H%M%SZ)-$$"
PT_ELASTIC_LOG="/data/jiahao/task/LlamaFactory/tmp_bash/pt_official_elastic_${PT_ELASTIC_RUN_ID}.nohup.log"
nohup setsid /home/jiahao/miniconda3/envs/bricknet/bin/python \
  scripts/supervise_bricknet_pt_elastic.py \
  --run-id "$PT_ELASTIC_RUN_ID" \
  --config configs/bricknet_mm_image_v2_official_pt_elastic_runtime_cuda1.json \
  >"$PT_ELASTIC_LOG" 2>&1 </dev/null &
printf 'run_id=%s supervisor_pid=%s log=%s\n' "$PT_ELASTIC_RUN_ID" "$!" "$PT_ELASTIC_LOG"
```

双卡 `pt-elastic-20260913T200555Z-633995` 与
`configs/bricknet_mm_image_v2_official_pt_elastic_runtime.json` 仅为迁移历史；旧双卡入口不得重新启动，
也不得把迁移前的局部计数写成当前全量结果。本次 renderer 已正常进入
`RENDER_COMPLETE_PENDING_VERIFY`/`result=OK`，随后 strict verify 与 completion gate 均为 `PASS`；
该结论不激活训练图片路径，也不创建数据集版本。

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

论文条件生成的严格复现契约见 [BrickNet Paper Reproduction Protocol](../BrickNet/BrickNet-MM%20Agentic%20LEGO%20Planner/BrickNet%20Paper%20Reproduction%20Protocol.md)（`planned`）：五规模 VAL512、每题首次生成加最多九次仅因完整 parse 失败触发的重采样，共最多十次尝试。以下是既有五模型官方渲染及评测入口，不是上述重采样协议的实现或执行记录；原始 prompt/generated token IDs 的逐 attempt 保存也尚未由当前生成入口满足。

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

# Official data v2 selection

官方渲染完成，并依次通过 media completion、projection、loader/token gate 后，训练或 prediction 才可按命令显式切换到 data v2；v2 的构造、渲染和各 gate 只记录在 [BRICKNET_RENDER_IMAGES_V2_zh.md](../BrickNet/data_preprocess/BRICKNET_RENDER_IMAGES_V2_zh.md) 与 [BRICKNET_MM_IMAGES_V2_PROJECTION_zh.md](../BrickNet/data_preprocess/BRICKNET_MM_IMAGES_V2_PROJECTION_zh.md)。运行时默认仍为 v1。

只有存在对应 `_v2` registry entry 且该 entry 的 gate 已通过的 SFT、VAL 或派生数据，才允许选择 data v2。PT 没有 official data v2；`PT-exp2` 的 `v2-no-meta` 等命名表示内部实验/派生投影，不等于 official data v2，也不能仅凭名称切换。

后续 record 中的直接命令按以下规则设置参数：

| 命令形式 | v2 选择参数 | 必须同时满足 |
| --- | --- | --- |
| YAML positional config | `bricknet_dataset_version=v2` | 在 config positional argument 后追加 `bricknet_dataset_version=v2 output_dir=..._data_v2`；prediction 同时追加/覆盖 `adapter_name_or_path=<v2-train-artifact>` |
| 纯 flag CLI | `--bricknet_dataset_version v2` | 添加 `--bricknet_dataset_version v2`，把现有 `--output_dir` 改为独立的 `..._data_v2`；prediction 同时修改 `--adapter_name_or_path <v2-train-artifact>` |

非空 `tokenized_path` 会由代码按 v2 自动隔离为 `_v2` 路径；不得复用 v1 tokenized cache，也不要手工把 dataset 名改成 `_v2`。当前通用 `python scripts/launch_*.py` gate launcher 不会透传 `bricknet_dataset_version`，不能在其命令末尾直接追加该参数；只有 launcher 明确支持该参数，并同时校验 v2 data、cache、output 的 identity 后才可使用，禁止绕过 launcher gate。

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

PT-exp2 text8m 当前登记为同一冻结协议的 fresh rerun。此前本机的 final/checkpoints 不可作为 resume；当前状态为 `ready`、尚未启动，fresh run 要求预期 output 目录不存在。使用现有 gate-protected launcher。

固定实验身份与配置：Qwen3.5-0.8B，固定 revision `2fc06364715b967f1860aea9cf38778875588b17`；冻结的 7,698,261 条 canonical paths；持久化 with-length tokenized cache；non-packing、`cutoff_len=6401`、250k steps；双卡每卡 batch size 16、GA1、global batch size 32。

```bash
cd /home/jiahao/task/LlamaFactory

# launcher 不会把 GPU 占用自动判为 blocker；正式启动前确认两卡没有其他计算进程。
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader

# dry-run
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2.py \
  --action train --run text8m --gpus 0 1

# 正式 fresh rerun：仅在两卡无其他计算进程时执行。
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2.py \
  --action train --run text8m --gpus 0 1 --execute
```

## text8m unconditional generation/evaluation

PT-exp2-text8m 的纯文本 final adapter 使用与 BrickNet 官方 PT 表相同的无条件生成/结构评分协议。以下命令登记为待执行入口；本轮未运行推理或评测。执行前必须人工确认 `ADAPTER_PATH` 指向本次 fresh rerun 完成且独立验收后的 final adapter 保存位置；此前 final adapter 的绑定不可作为当前产物。确认后才可使用 `--adapter-approved`。所有 action 默认是 dry-run；smoke/generate 的真实生成同时需要 `--adapter-approved` 与 `--execute` 两道门。

```bash
cd /home/jiahao/task/LlamaFactory
ADAPTER_PATH=/absolute/path/to/reviewed/text8m-250k-final-adapter
GPU_ID=0

conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_unconditional.py \
  --action preflight --adapter-path "$ADAPTER_PATH"
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_unconditional.py \
  --action smoke --adapter-path "$ADAPTER_PATH" --gpus "$GPU_ID" --adapter-approved --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_unconditional.py \
  --action generate --adapter-path "$ADAPTER_PATH" --gpus "$GPU_ID" --adapter-approved --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_unconditional.py \
  --action evaluate --adapter-path "$ADAPTER_PATH" --execute
conda run -n llamafactory --no-capture-output python scripts/launch_bricknet_pt_exp2_unconditional.py \
  --action verify --adapter-path "$ADAPTER_PATH" --execute
```

正式生成先写入 `outputs_pt/qwen35_08b/pt_exp2_text8m_250k_unconditional_v1.building/`，smoke 使用同级 `.smoke/`；`verify` 通过后才原子提升到最终目录 `/home/jiahao/task/BrickNet/outputs_pt/qwen35_08b/pt_exp2_text8m_250k_unconditional_v1/`。最终 artifact 为 `manifest.json`、`run_spec.json`、`out.jsonl`、`scored.jsonl`、`metrics.json`、`metrics.md` 和 `evaluation_manifest.json`。

## BrickNet PT unconditional NS（四个固定 arm）

NS 指 nucleus sampling。严格论文复现的五规模（0.6/1.7/4/8/14B）Graph/NS、每规模 65,536 条正式样本及原始 token 证据契约见 [BrickNet Paper Reproduction Protocol](../BrickNet/BrickNet-MM%20Agentic%20LEGO%20Planner/BrickNet%20Paper%20Reproduction%20Protocol.md)（`planned`）；本节四个固定 ID、N 和命令只代表既有 launcher 范围，不构成新五规模协议已经实现或运行的证据。

本节是四个 BrickNet PT 无条件 NS 实验的当前 launcher 入口，实际接口为 `scripts/launch_bricknet_paper_pt_unconditional.py`：0.6B/1.7B 使用 Paper Graph/NS 的 65,536 样本协议，8B/14B 使用现有 NS 代码的 2,048 样本协议。此前已完成该 launcher 和文档登记，四个 arm 均未运行；没有生成、评测或 verify 结果可回填。本轮仅修订新版方案文档，不授权执行下列命令；AS/Pose 不在本轮范围。

固定实验 ID、正式样本数和输出 slug 如下；不得自行增加 sweep arm 或修改 N：

| experiment ID | Model / Adapter | N | output slug |
| --- | --- | ---: | --- |
| `bricknet-pt-qwen3-06b-uncond-paper-ns-n65536-v1` | Qwen3-0.6B + BrickNet-0.6B-PT | 65,536 | `bricknet-pt-qwen3-06b-uncond-paper-ns-n65536-v1` |
| `bricknet-pt-qwen3-17b-uncond-paper-ns-n65536-v1` | Qwen3-1.7B + BrickNet-1.7B-PT | 65,536 | `bricknet-pt-qwen3-17b-uncond-paper-ns-n65536-v1` |
| `bricknet-pt-qwen3-8b-uncond-ns-n2048-v1` | Qwen3-8B + BrickNet-8B-PT | 2,048 | `bricknet-pt-qwen3-8b-uncond-ns-n2048-v1` |
| `bricknet-pt-qwen3-14b-uncond-ns-n2048-v1` | Qwen3-14B + BrickNet-14B-PT | 2,048 | `bricknet-pt-qwen3-14b-uncond-ns-n2048-v1` |

所有 action 默认是只读 dry-run。`preflight` 不需要 `--execute`；`smoke`、`pilot`、`generate` 的真实执行必须同时通过 `--artifacts-approved` 和 `--execute`；`evaluate`、`verify` 必须带 `--execute`，且只能在前一阶段的正式 staged artifact 通过 gate 后执行。GPU 选择器是物理 CUDA ID，必须先确认不抢占当前 PT render；launcher 不会替用户判断其它进程是否占用 GPU。

```bash
cd /data/jiahao/task/LlamaFactory
PAPER_NS_GPU_IDS=0  # 仅示例；替换为当前空闲的物理 CUDA ID（可写成 0,1）
PAPER_NS_LAUNCHER=scripts/launch_bricknet_paper_pt_unconditional.py

# 四个固定 arm 的只读 preflight；代码验收通过后先执行到这里。
conda run -n llamafactory --no-capture-output python "$PAPER_NS_LAUNCHER" \
  --experiment bricknet-pt-qwen3-06b-uncond-paper-ns-n65536-v1 --action preflight --gpus "$PAPER_NS_GPU_IDS"
conda run -n llamafactory --no-capture-output python "$PAPER_NS_LAUNCHER" \
  --experiment bricknet-pt-qwen3-17b-uncond-paper-ns-n65536-v1 --action preflight --gpus "$PAPER_NS_GPU_IDS"
conda run -n llamafactory --no-capture-output python "$PAPER_NS_LAUNCHER" \
  --experiment bricknet-pt-qwen3-8b-uncond-ns-n2048-v1 --action preflight --gpus "$PAPER_NS_GPU_IDS"
conda run -n llamafactory --no-capture-output python "$PAPER_NS_LAUNCHER" \
  --experiment bricknet-pt-qwen3-14b-uncond-ns-n2048-v1 --action preflight --gpus "$PAPER_NS_GPU_IDS"

# 通过人工资源/产物审批后，对一个已通过 preflight 的 ID 按顺序执行；
# 将 EXPERIMENT 分别设为上表四个 ID，不能跨 ID 复用 staged artifact。
EXPERIMENT=bricknet-pt-qwen3-06b-uncond-paper-ns-n65536-v1
conda run -n llamafactory --no-capture-output python "$PAPER_NS_LAUNCHER" \
  --experiment "$EXPERIMENT" --action smoke --gpus "$PAPER_NS_GPU_IDS" \
  --artifacts-approved --execute
conda run -n llamafactory --no-capture-output python "$PAPER_NS_LAUNCHER" \
  --experiment "$EXPERIMENT" --action pilot --gpus "$PAPER_NS_GPU_IDS" \
  --artifacts-approved --execute
conda run -n llamafactory --no-capture-output python "$PAPER_NS_LAUNCHER" \
  --experiment "$EXPERIMENT" --action generate --gpus "$PAPER_NS_GPU_IDS" \
  --artifacts-approved --execute
conda run -n llamafactory --no-capture-output python "$PAPER_NS_LAUNCHER" \
  --experiment "$EXPERIMENT" --action evaluate --gpus "$PAPER_NS_GPU_IDS" --execute
conda run -n llamafactory --no-capture-output python "$PAPER_NS_LAUNCHER" \
  --experiment "$EXPERIMENT" --action verify --gpus "$PAPER_NS_GPU_IDS" --execute
```

正式输出根目录为 `/data/jiahao/task/BrickNet/outputs_pt/paper_pt_unconditional/`。每个 slug 的正式生成先进入 `<slug>.building/`，诊断产物进入同级 `<slug>.smoke/` 和 `<slug>.pilot/`；staging/final bundle 包含 `run_spec.json`、`out.jsonl`、`scored.jsonl`、`metrics.json`、`metrics.md`、`evaluation_manifest.json`，`verify` 通过后才把正式 `.building/` 原子提升为 `<slug>/`，并新增 `manifest.json`。当前四个目录均不应被视为已存在或已完成的结果。

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
