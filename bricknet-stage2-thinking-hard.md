# BrickNet Stage 2 Thinking-Hard SFT Runbook

状态：`exp4`–`exp4_3` 的数据、配置、注册项、启动命令和 dry-run gate 已准备；阶段 2 训练及推理均未启动。
当前训练入口的唯一外部等待项是阶段 0 gate 与 mixed PT-exp1 final adapter；预测还必须等待对应阶段 2 adapter。

本阶段比较严格同 ID 的两组数据：

- `nonthinking-control` / `NonThinking-Control`：assistant 为原始 BrickNet path；
- `thinking-hard` / `Thinking-Hard`：assistant 为显式 `<think>/<action>` 完整 trace。

两组都使用 `qwen3_5_nothink`、`enable_thinking=false`、`cutoff_len=16384`、
`train_on_prompt=false`、`packing=false`。显式 `<think>` 是普通 assistant 监督文本，不启用 Qwen 原生 thinking。

## 1. 实验版本与配置

阶段 2 沿用 LlamaFactory 既有主序列 `exp3_2` 之后的版本号：

| Exp | 实验 | 训练配置 | 预测配置 |
| --- | --- | --- | --- |
| `exp4` | NonThinking-Control VAL511 overfit | `examples/train_lora/qwen35_08b_bricknet_stage2_exp4_nonthinking_control_val511.yaml` | `examples/train_lora/qwen35_08b_bricknet_stage2_exp4_nonthinking_control_predict.yaml` |
| `exp4_1` | Thinking-Hard VAL511 overfit | `examples/train_lora/qwen35_08b_bricknet_stage2_exp4_1_thinking_hard_val511.yaml` | `examples/train_lora/qwen35_08b_bricknet_stage2_exp4_1_thinking_hard_predict.yaml` |
| `exp4_2` | NonThinking-Control 10k | `examples/train_lora/qwen35_08b_bricknet_stage2_exp4_2_nonthinking_control_10k.yaml` | `examples/train_lora/qwen35_08b_bricknet_stage2_exp4_2_nonthinking_control_predict.yaml` |
| `exp4_3` | Thinking-Hard 10k | `examples/train_lora/qwen35_08b_bricknet_stage2_exp4_3_thinking_hard_10k.yaml` | `examples/train_lora/qwen35_08b_bricknet_stage2_exp4_3_thinking_hard_predict.yaml` |

统一安全启动器为 `scripts/launch_bricknet_stage2_sft.py`。默认只执行 dry-run；实际执行必须同时传入：

```text
--execute --stage0-gate-approved
```

启动器只接受 `overfit511` 和 `10k`，并检查 final Stage-0 adapter、Stage-1 token gate、VAL
annotation/token gate、dataset registry、数据与 manifest 行数、10k 物化报告/hash、对应 Stage-2 adapter 和输出目录。
10k train 还要求在 paired VAL511 结果经审核后显式传入 `--overfit-gate-approved`，防止跳级。

50k/all 当前暂停：不分配实验版本号，不提供训练/预测配置、dataset registry 或 launcher scale；此前用于 all 的两个
临时软链接已移除，Stage-1 的 66,456 条不可变源数据未删除。已有 50k/all nested manifest 仅作未来可复现依据；
只有用户明确恢复该路线后才能继续准备。

## 2. 数据准备与证据

### 2.1 Nested manifest

确定性选择规则为：

```text
ascending sha256("42\0" + sample_id), then sample_id
```

2026-08-06 已验证 66,456 条 paired 样本同 ID、同顺序、同 reference、同 image/user prompt，且
`10k ⊂ 50k ⊂ all`。证据：

```text
/home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Reasoning/stage2/reports/stage2_nested_manifests_report.json
```

当前只消费 `stage2_train_10k_seed42.jsonl`；50k/all manifest 不会触发数据或实验准备。

### 2.2 VAL511 训练与 VAL512 推理

构造命令不运行训练：

```bash
cd /home/jiahao/task/BrickNet
BRICKNET_DATA=/home/jiahao/task/BrickNet/data/bricknet_datasets \
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  data_preprocess/prepare_bricknet_stage2_sft.py build-val
```

实际结果：`processed=512`、`eval_accepted=512`、`train_accepted=511`、`train_excluded=1`。
唯一排除 ID `BrickNet-MM__VAL__row-000114__caption-000__pathline-000010079` 的 reference 在 action 46
发生真实 mesh collision；它只不参加 overfit 训练，原 reference 未修改并保留在 512 条推理/评测集。

两次真实多模态 token audit 的命令为：

```bash
cd /home/jiahao/task/LlamaFactory
conda run -n llamafactory --no-capture-output python \
  scripts/audit_bricknet_reasoning_tokens.py \
  --stage 2 --audit-purpose val511_overfit_train \
  --dataset NonThinking-Control=/home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Reasoning/validation/datasets/BrickNet-Stage2-NonThinking-Control-VAL511-Train.jsonl \
  --dataset Thinking-Hard=/home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Reasoning/validation/datasets/BrickNet-Stage2-Thinking-Hard-VAL511-Train.jsonl \
  --bricknet-root /home/jiahao/task/BrickNet \
  --output-dir /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Reasoning/validation/reports/token_audit/train_val511 \
  --cutoff-len 16384 --workers 2 --chunksize 8

conda run -n llamafactory --no-capture-output python \
  scripts/audit_bricknet_reasoning_tokens.py \
  --stage 2 --audit-purpose val512_inference_eval \
  --dataset NonThinking-Control=/home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Reasoning/validation/datasets/BrickNet-Stage2-NonThinking-Control-VAL512-Eval.jsonl \
  --dataset Thinking-Hard=/home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Reasoning/validation/datasets/BrickNet-Stage2-Thinking-Hard-VAL512-Eval.jsonl \
  --bricknet-root /home/jiahao/task/BrickNet \
  --output-dir /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Reasoning/validation/reports/token_audit/eval_val512 \
  --cutoff-len 16384 --workers 2 --chunksize 8
```

当前结果：VAL511 两组各 511 条，VAL512 两组各 512 条；均为 paired ID/order 相同、0 error、0 truncation。
统一报告已通过 `finalize-val` 合并，`readiness_gate_passed=true`、`training_eligible=true`、
`prediction_eligible=true`：

```text
/home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Reasoning/validation/reports/BrickNet-Stage2-VAL511-Train-VAL512-Eval_report.json
```

短名 `NonThinking` 的上一版 VAL 产物保存在：

```text
/home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Reasoning/validation/superseded/2026-08-06-nonthinking-short-name/
```

### 2.3 10k 物化

已执行且不运行训练：

```bash
cd /home/jiahao/task/BrickNet
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  data_preprocess/prepare_bricknet_stage2_sft.py materialize --scale 10k
```

两组各 10,000 条，使用同一 manifest，`paired_order_and_ids=true`、`training_eligible=true`。报告：

```text
/home/jiahao/task/LlamaFactory/data/bricknet_stage2/10k/materialization_report.json
```

NonThinking-Control SHA-256 为 `4be6a7fb711ba24a658fc3096a8f1e2a9aa3e630c76adb8f04313bf9bff02a15`；
Thinking-Hard 为 `f1842b43772eeabedc2094a09da4a5814f84608eafd2956390d8b0121e16f100`；两者 ordered ID
SHA-256 均为 `2d87ff4c3b918f748dde48721cbec66595ccc17317cf728f77e30efc04230dea`。

## 3. Dry-run 与正式训练命令

以下命令当前只做检查，不会训练：

```bash
cd /home/jiahao/task/LlamaFactory

# exp4: NonThinking-Control VAL511 overfit
python scripts/launch_bricknet_stage2_sft.py \
  --action train --variant nonthinking-control --scale overfit511

# exp4_1: Thinking-Hard VAL511 overfit
python scripts/launch_bricknet_stage2_sft.py \
  --action train --variant thinking-hard --scale overfit511

# exp4_2: NonThinking-Control 10k
python scripts/launch_bricknet_stage2_sft.py \
  --action train --variant nonthinking-control --scale 10k

# exp4_3: Thinking-Hard 10k
python scripts/launch_bricknet_stage2_sft.py \
  --action train --variant thinking-hard --scale 10k
```

阶段 0 完成、dry-run 仅剩的 `WAIT_STAGE0_FINAL` 消失且用户确认后，才可在对应命令末尾增加：

```text
--execute --stage0-gate-approved
```

上式只适用于两个 overfit 实验。10k 当前 dry-run 还会返回 `WAIT_STAGE2_OVERFIT_GATE`；两组 overfit 完成并
经用户审核后，10k 正式执行必须同时增加：

```text
--overfit-gate-approved --execute --stage0-gate-approved
```

不得把 `--scale` 改成 50k/all；启动器不会接受这两个暂停规模。

## 4. 预测与 path extractor

每个 checkpoint 都对完整 512 条 VAL 推理。对应 dry-run 命令为：

```bash
python scripts/launch_bricknet_stage2_sft.py \
  --action predict --variant nonthinking-control --scale overfit511
python scripts/launch_bricknet_stage2_sft.py \
  --action predict --variant thinking-hard --scale overfit511
python scripts/launch_bricknet_stage2_sft.py \
  --action predict --variant nonthinking-control --scale 10k
python scripts/launch_bricknet_stage2_sft.py \
  --action predict --variant thinking-hard --scale 10k
```

预测 dry-run 在训练前会额外返回 `WAIT_STAGE2_TRAIN`。Thinking-Hard 生成完成后，使用 strict extractor：

```bash
cd /home/jiahao/task/BrickNet
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  scripts/extract_reasoning_predictions.py \
  --variant thinking-hard \
  --label-format path \
  --input /home/jiahao/task/LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/eval_exp4_3_stage2_thinking_hard_10k_val512_in16384_out16384_p95_t1_k20/generated_predictions.jsonl \
  --output outputs_val/qwen35_08b/eval_exp4_3_stage2_thinking_hard_10k_val512/path_predictions.jsonl \
  --report outputs_val/qwen35_08b/eval_exp4_3_stage2_thinking_hard_10k_val512/trace_extraction_report.json
```

LlamaFactory 对 trace 文本计算的 BLEU/ROUGE 不是最终结构指标；最终报告必须包含 trace-format、extractor、
parse、inventory、collision、pose、语义指标、token、延迟和 GPU time。

## 5. 准备完成口径与等待项

四个实验的本地准备已完成：配置与 registry 存在、VAL/10k 数据和 manifest 行数正确、10k hash 与物化报告一致、
VAL 和全量 Stage-1 token gate 通过、八个 train/predict dry-run 均只出现预期等待项，且所有输出目录均不存在。

这不等于可立即训练：

1. mixed PT-exp1 输出根目录尚无 final `adapter_config.json` 和 adapter 权重，训练 dry-run 返回
   `WAIT_STAGE0_FINAL`；
2. 即使 final adapter 存在，仍须用户确认阶段 0 gate 并显式传入 `--stage0-gate-approved --execute`；
3. 10k train 还必须等待 paired VAL511 审核并显式传入 `--overfit-gate-approved`；当前返回
   `WAIT_STAGE2_OVERFIT_GATE`；
4. 16,384 token 的真实吞吐与峰值显存只能在获准后的 511 overfit 中确认；两组必须保持相同有效 batch；
5. 预测必须等待对应 Stage-2 adapter，因此当前额外返回 `WAIT_STAGE2_TRAIN`；
6. 50k/all 准备暂停，当前不属于可执行实验范围。

准备代码和 dry-run 不代表阶段 2 已开始；只有安全启动器实际调用 LlamaFactory trainer 才算进入训练。
