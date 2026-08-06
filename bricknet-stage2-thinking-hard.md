# BrickNet Stage 2 Thinking-Hard SFT Runbook

状态：训练前数据、token gate、配置和安全启动器已准备；阶段 2 训练未启动。当前只等待阶段 0 gate
通过和 mixed PT-exp1 final adapter 冻结。

本阶段比较严格同 ID 的两组数据：

- `nonthinking`：assistant 为原始 BrickNet path；
- `thinking-hard`：assistant 为显式 `<think>/<action>` 完整 trace。

两组都使用 `qwen3_5_nothink`、`enable_thinking=false`、`cutoff_len=16384`、
`train_on_prompt=false`、`packing=false`。显式 `<think>` 是普通 assistant 监督文本，不启用 Qwen 原生 thinking。

## 1. 已准备入口

训练基础配置：

```text
examples/train_lora/qwen35_08b_bricknet_stage2_nonthinking.yaml
examples/train_lora/qwen35_08b_bricknet_stage2_thinking_hard.yaml
```

预测基础配置：

```text
examples/train_lora/qwen35_08b_bricknet_stage2_nonthinking_predict.yaml
examples/train_lora/qwen35_08b_bricknet_stage2_thinking_hard_predict.yaml
```

统一安全启动器：

```text
scripts/launch_bricknet_stage2_sft.py
```

启动器默认只执行 dry-run。实际执行必须同时传入：

```text
--execute --stage0-gate-approved
```

且必须通过 final Stage-0 adapter、Stage-1 token gate、VAL annotation/token gate、dataset、manifest 和输出目录检查。

## 2. 数据准备

### 2.1 生成 nested manifests

只生成 ID manifest，不提前物化 10k/50k 训练 JSONL：

```bash
cd /home/jiahao/task/BrickNet
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  data_preprocess/prepare_bricknet_stage2_sft.py manifests
```

选择规则固定为：

```text
ascending sha256("42\0" + sample_id), then sample_id
```

因此 10k 是 50k 的严格前缀/子集，50k 是 all 的严格前缀/子集。manifest 保存 ID、selection rank、source line、
reference hash 和必要分组字段。

2026-08-06 已完成该命令：66,456 条 paired 样本的同 ID、同顺序、同 reference、同 image/user prompt
校验全部通过，且 `10k ⊂ 50k ⊂ all`。报告为
`/home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Reasoning/stage2/reports/stage2_nested_manifests_report.json`；
`datasets_materialized=false`。

### 2.2 构造 VAL511 overfit 训练数据和 VAL512 推理数据

```bash
cd /home/jiahao/task/BrickNet
BRICKNET_DATA=/home/jiahao/task/BrickNet/data/bricknet_datasets \
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  data_preprocess/prepare_bricknet_stage2_sft.py build-val
```

该命令复用官方 VAL 的原始 reference path，不运行训练。它同时生成：

- `VAL511-Train`：对 Thinking-Hard 执行真实 parse、inventory、collision 和 byte-exact extractor 检查；
- `VAL512-Eval`：保留原始全部 512 条 prompt/reference，Thinking-Hard 数据只替换 system prompt，assistant label
  仍是 pure reference path；预测 trace 由后续 strict extractor 抽取。

2026-08-06 实际 gate：

```text
processed=512
eval_accepted=512
train_accepted=511
train_excluded=1
evaluation_gate_passed=true
train_annotation_gate_passed=true
```

训练排除记录：

```text
source line: 115
id: BrickNet-MM__VAL__row-000114__caption-000__pathline-000010079
collision_indices: [46]
disposition: excluded_from_VAL511_overfit_only; retained_in_VAL512_inference
```

该 reference 可解析，共 65 parts/129 lines，真实 mesh checker 在 action 46 返回碰撞。它没有被删除、替换或
修改，只不参与 overfit 训练。如果以后出现第二条排除，或该 ID 的 collision indices 不再是 `[46]`，
`build-val` 仍会失败。报告：

```text
/home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Reasoning/validation/reports/BrickNet-Stage2-VAL511-Train-VAL512-Eval_report.json
```

随后执行真实多模态 token audit：

```bash
cd /home/jiahao/task/LlamaFactory
conda run -n llamafactory --no-capture-output python \
  scripts/audit_bricknet_reasoning_tokens.py \
  --stage 2 --audit-purpose val511_overfit_train \
  --dataset NonThinking=/home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Reasoning/validation/datasets/BrickNet-Stage2-NonThinking-VAL511-Train.jsonl \
  --dataset Thinking-Hard=/home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Reasoning/validation/datasets/BrickNet-Stage2-Thinking-Hard-VAL511-Train.jsonl \
  --bricknet-root /home/jiahao/task/BrickNet \
  --output-dir /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Reasoning/validation/reports/token_audit/train_val511 \
  --cutoff-len 16384 --workers 2 --chunksize 8
```

VAL512 推理数据审计：

```bash
conda run -n llamafactory --no-capture-output python \
  scripts/audit_bricknet_reasoning_tokens.py \
  --stage 2 --audit-purpose val512_inference_eval \
  --dataset NonThinking=/home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Reasoning/validation/datasets/BrickNet-Stage2-NonThinking-VAL512-Eval.jsonl \
  --dataset Thinking-Hard=/home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Reasoning/validation/datasets/BrickNet-Stage2-Thinking-Hard-VAL512-Eval.jsonl \
  --bricknet-root /home/jiahao/task/BrickNet \
  --output-dir /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Reasoning/validation/reports/token_audit/eval_val512 \
  --cutoff-len 16384 --workers 2 --chunksize 8
```

两次审计已完成：

- VAL511 Train：NonThinking/Thinking-Hard 各 511 条，0 error、0 truncation，paired ID/order 相同；
- VAL512 Eval：NonThinking/Thinking-Hard 各 512 条，0 error、0 truncation，paired ID/order 相同。

两个 token audit 完成后，必须把外部 gate 合并回统一 VAL report：

```bash
cd /home/jiahao/task/BrickNet
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  data_preprocess/prepare_bricknet_stage2_sft.py finalize-val
```

该命令会复核两个 token report 的 stage/purpose、样本数、dataset SHA、paired ID/order、error 和
truncation，然后写入 `readiness_gate_passed=true`、`training_eligible=true`、
`prediction_eligible=true`。它不调用 trainer。

### 2.3 在某个实验启动前物化 10k/50k

10k：

```bash
cd /home/jiahao/task/BrickNet
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  data_preprocess/prepare_bricknet_stage2_sft.py materialize --scale 10k
```

50k 只在 10k gate 通过后执行：

```bash
cd /home/jiahao/task/BrickNet
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  data_preprocess/prepare_bricknet_stage2_sft.py materialize --scale 50k
```

`all` 通过 `data/bricknet_stage2/all/BrickNet-Stage2-{NonThinking,ThinkingHard}.jsonl` 语义链接直接读取
阶段 1 不可变的 66,456 条源文件，不复制 1.4 GB 数据；启动器仍对解引后内容做 Stage-1 token-report
SHA-256 校验。

## 3. Dry-run 和正式训练命令

### 3.1 511 VAL overfit smoke

先检查 NonThinking：

```bash
cd /home/jiahao/task/LlamaFactory
python scripts/launch_bricknet_stage2_sft.py \
  --action train --variant nonthinking --scale overfit511
```

再检查 Thinking-Hard：

```bash
python scripts/launch_bricknet_stage2_sft.py \
  --action train --variant thinking-hard --scale overfit511
```

阶段 0 gate 通过、dry-run `ready=true` 且用户确认后，才可在相同命令末尾增加：

```text
--execute --stage0-gate-approved
```

### 3.2 10k paired 主实验

先物化 10k，然后分别 dry-run：

```bash
python scripts/launch_bricknet_stage2_sft.py \
  --action train --variant nonthinking --scale 10k

python scripts/launch_bricknet_stage2_sft.py \
  --action train --variant thinking-hard --scale 10k
```

50k/all 只需要把 `--scale` 改为 `50k` 或 `all`；必须遵守 `10k → 50k → all` gate，不允许跳级扩容。

## 4. 预测与 path extractor

训练完成后先 dry-run 预测：

```bash
python scripts/launch_bricknet_stage2_sft.py \
  --action predict --variant thinking-hard --scale 10k
```

实际生成完成后，Thinking trace 必须先提取为 BrickNet path：

```bash
cd /home/jiahao/task/BrickNet
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  scripts/extract_reasoning_predictions.py \
  --variant thinking-hard \
  --label-format path \
  --input /home/jiahao/task/LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/eval_stage2_thinking_hard_10k_in16384_out16384_p95_t1_k20/generated_predictions.jsonl \
  --output outputs_val/qwen35_08b/stage2_thinking_hard_10k/path_predictions.jsonl \
  --report outputs_val/qwen35_08b/stage2_thinking_hard_10k/trace_extraction_report.json
```

之后把 `path_predictions.jsonl` 交给现有 `scripts/evaluate_experiment.py` 和 ms-swift alignment worker。
LlamaFactory 对原始 trace 文本计算的 BLEU/ROUGE 不是阶段 2 的最终结构指标；最终报告必须包含 trace-format、extractor、
parse、inventory、collision、pose、语义指标、token、延迟和 GPU time。

## 5. 当前明确等待项

1. **等待阶段 0：** mixed PT-exp1 仍在运行；其输出根目录尚无 final `adapter_config.json` 和
   `adapter_model.safetensors`，安全启动器会返回 `WAIT_STAGE0_FINAL`。
2. **等待用户 gate：** 即使 final adapter 已存在，实际启动仍要求用户确认阶段 0 gate，并显式传入
   `--stage0-gate-approved`。
3. **等待真实显存 smoke：** 16,384 token 将 micro-batch 降为 1、gradient accumulation 提至 16；实际吞吐、峰值显存和
   是否需要进一步调整只能在阶段 0 完成后的 511 smoke 中确定。NonThinking/Thinking-Hard 必须保持相同有效 batch。
4. **等待逐级扩容：** 50k 只在 10k paired gate 通过后物化；all 只在 50k gate 通过后启动。

准备代码和 dry-run 不代表阶段 2 已开始；只有安全启动器实际调用 LlamaFactory trainer 才算进入训练。
