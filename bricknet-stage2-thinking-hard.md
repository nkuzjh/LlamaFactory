# BrickNet Stage 2 Thinking-Hard SFT Runbook

状态（2026-08-13 +08:00）：Stage 0 mixed PT-exp1 final 已完成；`exp4`–`exp4_3` 四个训练和 VAL512
512/512 推理和全指标均完成。paired 结果显示 Thinking-Hard 只改善 clean/collision-prefix，parsable、dense、
strict success 和图文指标均低于 Control；T1-10k 人工推广 gate 未批准。新增的 `exp4_3_1` 是封顶 10k、独立数据链的
Lean-State 诊断；正式数据和两份真实 processor audit 已通过，训练 dry-run ready；训练、推理和评测尚未执行。

原 Stage 2 比较严格同 ID 的两组数据，新增 V2 仍复用相同 10k ID：

- `nonthinking-control` / `NonThinking-Control`：assistant 为原始 BrickNet path；
- `thinking-hard` / `Thinking-Hard`：assistant 为显式 `<think>/<action>` 完整 trace。
- `thinking-hard-v2-lean-state` / `Thinking-Hard-V2-Lean-State`：assistant 为删除答案复述和文字 checks 的短
  state-before `<think>/<action>` trace，只用于 `exp4_3_1` 10k。

三组都使用 `qwen3_5_nothink`、`enable_thinking=false`、`cutoff_len=16384`、
`train_on_prompt=false`、`packing=false`。显式 `<think>` 是普通 assistant 监督文本，不启用 Qwen 原生 thinking。

## 1. 实验版本与配置

阶段 2 沿用 LlamaFactory 既有主序列 `exp3_2` 之后的版本号：

| Exp | 实验 | 训练配置 | 预测配置 |
| --- | --- | --- | --- |
| `exp4` | NonThinking-Control VAL511 overfit | `examples/train_lora/qwen35_08b_bricknet_stage2_exp4_nonthinking_control_val511.yaml` | `examples/train_lora/qwen35_08b_bricknet_stage2_exp4_nonthinking_control_predict.yaml` |
| `exp4_1` | Thinking-Hard VAL511 overfit | `examples/train_lora/qwen35_08b_bricknet_stage2_exp4_1_thinking_hard_val511.yaml` | `examples/train_lora/qwen35_08b_bricknet_stage2_exp4_1_thinking_hard_predict.yaml` |
| `exp4_2` | NonThinking-Control 10k | `examples/train_lora/qwen35_08b_bricknet_stage2_exp4_2_nonthinking_control_10k.yaml` | `examples/train_lora/qwen35_08b_bricknet_stage2_exp4_2_nonthinking_control_predict.yaml` |
| `exp4_3` | Thinking-Hard 10k | `examples/train_lora/qwen35_08b_bricknet_stage2_exp4_3_thinking_hard_10k.yaml` | `examples/train_lora/qwen35_08b_bricknet_stage2_exp4_3_thinking_hard_predict.yaml` |
| `exp4_3_1` | Thinking-Hard-V2-Lean-State 10k | `examples/train_lora/qwen35_08b_bricknet_stage2_exp4_3_1_thinking_hard_v2_lean_state_10k.yaml` | `examples/train_lora/qwen35_08b_bricknet_stage2_exp4_3_1_thinking_hard_v2_lean_state_predict.yaml` |

统一安全启动器为 `scripts/launch_bricknet_stage2_sft.py`。默认只执行 dry-run；实际执行必须同时传入：

```text
--execute --stage0-gate-approved
```

启动器只接受 `overfit511` 和 `10k`，并检查 final Stage-0 adapter、Stage-1 token gate、VAL
annotation/token gate、dataset registry、数据与 manifest 行数、10k 物化报告/hash、对应 Stage-2 adapter 和输出目录。
10k train 还要求在 paired VAL511 结果经审核后显式传入 `--overfit-gate-approved`，防止跳级。

原 Stage 2 路线的 50k/all 当前暂停：`launch_bricknet_stage2_sft.py` 不接受这两个 scale；此前用于 all 的两个
临时软链接已移除，Stage-1 的 66,456 条不可变源数据未删除。新 PT-exp2 分支另外预留 `exp4_5/exp4_6` 的 dormant
配置和 registry，但仍分别受 exp4_4/exp4_5 人工收益 gate 保护，不能据此绕过本路线的暂停决定。

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

## 4. 预测与统一评测

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

完整评测统一使用以下入口；默认 dry-run，正式执行追加 `--execute`：

```bash
python scripts/evaluate_bricknet_stage2.py --experiment exp4
python scripts/evaluate_bricknet_stage2.py --experiment exp4_1
python scripts/evaluate_bricknet_stage2.py --experiment exp4_2
python scripts/evaluate_bricknet_stage2.py --experiment exp4_3
```

该入口依次完成 strict path 提取、path BLEU/ROUGE、BrickNet 结构/渲染/图文指标和 alignment，并自动选择
`nonthinking-control` 或 `thinking-hard`。已有完整结果会安全复用；只有显式传入 `--force` 才会重算。

预测 dry-run 在训练前会额外返回 `WAIT_STAGE2_TRAIN`。launcher 在预测成功后自动运行 strict extractor，输出与
`generated_predictions.jsonl` 同目录的 `path_predictions.jsonl` 和 `trace_extraction_report.json`。需要对已有预测
补做或重做提取时，执行：

```bash
cd /home/jiahao/task/BrickNet
PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python \
  scripts/extract_reasoning_predictions.py \
  --variant thinking-hard \
  --label-format path \
  --input /home/jiahao/task/LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/eval_exp4_3_stage2_thinking_hard_10k_val512_in16384_out16384_p95_t1_k20/generated_predictions.jsonl \
  --output /home/jiahao/task/LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/eval_exp4_3_stage2_thinking_hard_10k_val512_in16384_out16384_p95_t1_k20/path_predictions.jsonl \
  --report /home/jiahao/task/LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/eval_exp4_3_stage2_thinking_hard_10k_val512_in16384_out16384_p95_t1_k20/trace_extraction_report.json \
  --overwrite
```

Thinking-Hard 的 `generated_predictions.jsonl` 是 reasoning trace，BrickNet 结构和 alignment 评测必须读取提取后的
`path_predictions.jsonl`。文本指标同样必须基于 extracted path 重新计算，并通过 `--text-metrics` 显式传给 evaluator；
原始 LlamaFactory `predict_results.json` 比较的是 trace 和 path label，数值无效。最终报告必须包含 trace-format、extractor、
parse、inventory、collision、pose、语义指标、token、延迟和 GPU time。

两种 variant 使用同一个 canonical evaluation contract：

| Variant | `generated_predictions.jsonl` | `path_predictions.jsonl` | 后续结构/图文/text/alignment 输入 |
| --- | --- | --- | --- |
| `nonthinking-control` | 原生 path | 只规范化末尾换行，path 内容不变 | canonical path |
| `thinking-hard` | `<think>/<action>` trace | strict extractor 提取的 path 或合法前缀 | canonical path |
| `thinking-hard-v2-lean-state` | Lean-State `<think>/<action>` trace | state-machine 提取 path，并逐行重算 V2 state 检查内部一致性 | canonical path |

launcher 会根据 experiment 自动传入正确的 `--variant`。手工调用 extractor 时必须显式选择 variant。non-thinking
报告中的 `trace_format_rate=100%` 只表示无需解析 trace，不代表结构 100% 合法；两类实验的 Connectivity/Clean
仍统一由 BrickNet path scorer 计算。

## 5. 当前完成度与等待项

四个实验均已通过既有数据/manifest/token/launcher gate并完成训练：

1. `exp4`：train loss `0.1737931`，VAL512 512/512 推理完成；
2. `exp4_1`：train loss `0.0859583`，VAL512 512/512 推理完成；
3. `exp4_2`：train loss `0.1726632`，VAL512 512/512 推理和全指标完成；parsable
   `382/512 (74.61%)`、clean `93/512 (18.16%)`、dense reward `0.58159`、strict success `16/512 (3.12%)`；
4. `exp4_3`：train loss `0.0433687`，VAL512 512/512 推理完成；strict extractor 得到
   `360/512 (70.31%)` 完整合法 trace/path，512 条均有非空 extracted prefix；全指标为 clean
   `101/512 (19.73%)`、Collision `6.1738`、PE/SigLIP2/VQAScore `0.2799/0.7818/0.7486`、Inventory F1
   `0.8812`、dense reward `0.57395`、strict success `13/512 (2.54%)`。

相对 exp4_2，exp4_3 的 clean `+1.56 pp`、collision-prefix `+0.0133`，但 parsable `-4.30 pp`、dense reward
`-0.00765`、strict success `-0.59 pp`，PE/SigLIP2/VQA 也较低，因此没有总体优势。Stage 2 原路线的 50k/all
继续暂停；新 PT-exp2 分支另从 `exp4_4` 10k 开始，见 [PT-exp2 runbook](bricknet-pt-exp2.md)。

## 6. exp4_3_1 / Stage2 V2 Lean-State

`exp4_3_1` 不重写 T1 v1。每步 `<think>` 只有一行由输入 inventory 和已执行 prefix 确定的 state-before；
`<action>` 保留原 reference。完整 schema、构造路径和研究边界见
[BrickNet Stage 2 V2 说明](../BrickNet/BrickNet-MM%20Agentic%20LEGO%20Planner/Stage%202%20V2%20Lean-State%20Auto-Annotation.md)。

LlamaFactory registry 使用：

```text
BrickNet-Stage2-ThinkingHard-V2-LeanState-10k
BrickNet-Stage2-ThinkingHard-V2-LeanState-VAL512-Eval
```

启动器对 V2 使用独立构造报告和 train/VAL512 两份真实 processor token report；同时核对 path、count 和 SHA-256。
它不接受仅凭文件存在放行。V2 strict extractor 沿用同一 `<think>/<action>` state-machine，并从完整生成 path
重算每步 inventory、node range 和 connector candidates；只有逐行一致才计入 trace-format valid。统一 evaluator
的实验 ID 为 `exp4_3_1`。

数据构造和两次 token audit 完成后，按以下顺序运行：

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

当前 V2 data/report 和两份 token report 已通过，train dry-run 为 `ready=true, blockers=[]`；predict 在训练前只应
报告 `WAIT_STAGE2_TRAIN`。不得删除该 gate 或用旧 T1 token report 代替。该实验只运行 10k，不添加
overfit511、50k 或 all 配置。
