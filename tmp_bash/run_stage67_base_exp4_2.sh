 #!/usr/bin/env bash
  set -euo pipefail

  cd /home/jiahao/task/BrickNet
  export BRICKNET_PY=/home/jiahao/miniconda3/envs/bricknet/bin/python
  export BRICKNET_DATA=/home/jiahao/task/BrickNet/data/bricknet_datasets

  # 可选：GPU 0 空闲（利用率 < 20%）后再开始；不需要就保持注释。
  # while [ "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0)" -ge 20 ]; do
  #   sleep 60
  # done

  # 1) B1
  PYTHONPATH=src "$BRICKNET_PY" scripts/run_bricknet_agentic_inference.py \
    --input outputs_preprocess/BrickNet-MM-Reasoning/validation/datasets/BrickNet-Stage2-NonThinking-Control-VAL512-Eval.jsonl \
    --output outputs_val/qwen35_08b/agentic_exp4_2_b1/controller_audit.jsonl \
    --mode b1-post-hoc --backend hf --prompt-protocol exp4_2-stepwise --seed 42 \
    --stage5-report outputs_preprocess/BrickNet-MM-Act-SFT/stage5/Stage5-full-replay-report.json

  # 2) V1
  PYTHONPATH=src "$BRICKNET_PY" scripts/run_bricknet_agentic_inference.py \
    --input outputs_preprocess/BrickNet-MM-Reasoning/validation/datasets/BrickNet-Stage2-NonThinking-Control-VAL512-Eval.jsonl \
    --output outputs_val/qwen35_08b/agentic_exp4_2_v1/controller_audit.jsonl \
    --mode v1-silent-retry --backend hf --prompt-protocol exp4_2-stepwise --seed 42 \
    --stage5-report outputs_preprocess/BrickNet-MM-Act-SFT/stage5/Stage5-full-replay-report.json

  # 3) V2
  PYTHONPATH=src "$BRICKNET_PY" scripts/run_bricknet_agentic_inference.py \
    --input outputs_preprocess/BrickNet-MM-Reasoning/validation/datasets/BrickNet-Stage2-NonThinking-Control-VAL512-Eval.jsonl \
    --output outputs_val/qwen35_08b/agentic_exp4_2_v2/controller_audit.jsonl \
    --mode v2-silent-dfs --backend hf --prompt-protocol exp4_2-stepwise --seed 42 \
    --candidates-per-round 8 --max-rounds-per-state 4 --max-backtrack-depth 3 \
    --stage5-report outputs_preprocess/BrickNet-MM-Act-SFT/stage5/Stage5-full-replay-report.json

  # 4) A0
  PYTHONPATH=src "$BRICKNET_PY" scripts/run_bricknet_agentic_inference.py \
    --input outputs_preprocess/BrickNet-MM-Reasoning/validation/datasets/BrickNet-Stage2-NonThinking-Control-VAL512-Eval.jsonl \
    --output outputs_val/qwen35_08b/agentic_exp4_2_a0/controller_audit.jsonl \
    --mode a0-act-feedback --backend hf --prompt-protocol stage8-act --seed 42 \
    --stage5-report outputs_preprocess/BrickNet-MM-Act-SFT/stage5/Stage5-full-replay-report.json

  # 5) A1
  PYTHONPATH=src "$BRICKNET_PY" scripts/run_bricknet_agentic_inference.py \
    --input outputs_preprocess/BrickNet-MM-Reasoning/validation/datasets/BrickNet-Stage2-NonThinking-Control-VAL512-Eval.jsonl \
    --output outputs_val/qwen35_08b/agentic_exp4_2_a1/controller_audit.jsonl \
    --mode a1-feedback-search --backend hf --prompt-protocol stage8-act --seed 42 \
    --candidates-per-round 8 --max-rounds-per-state 4 --max-backtrack-depth 3 \
    --stage5-report outputs_preprocess/BrickNet-MM-Act-SFT/stage5/Stage5-full-replay-report.json

  # 6) 后处理：preflight → 10 套 final/raw 评测 → 冻结 manifest → paired bootstrap。
  PYTHONPATH=src "$BRICKNET_PY" \
    scripts/evaluate_bricknet_agentic_stage67.py --action all --execute --force

  # 7) 打印最终分数。
  STATS=outputs_val/qwen35_08b/agentic_exp4_2_stage67_statistics.json
  jq -r '.runs | to_entries[] |
    "\(.key)\tfinal_strict=\(.value.final_system.task_strict_success*100|round/100)%\t" +
    "dense=\(.value.final_system.dense_reward)\thard_valid=\(.value.controller.controller_hard_valid_success)/512"' "$STATS"
  echo '---'
  jq -r '.comparisons | to_entries[] |
    "\(.key)\tfinal_strict_delta=\(.value.final_system.task_strict_success.candidate_minus_baseline*100|round/100)%\t" +
    "CI=[\(.value.final_system.task_strict_success.ci95_low*100|round/100)%, " +
    "\(.value.final_system.task_strict_success.ci95_high*100|round/100)%]"' "$STATS"
  echo '---'
  jq -r '.external_comparisons | to_entries[] |
    "\(.key)\tfinal_strict_delta=\(.value.task_strict_success.candidate_minus_baseline*100|round/100)%\t" +
    "CI=[\(.value.task_strict_success.ci95_low*100|round/100)%, " +
    "\(.value.task_strict_success.ci95_high*100|round/100)%]"' "$STATS"
  echo '---'
  echo '完整实验表：outputs_val/qwen35_08b/agentic_exp4_2_stage67_results.md'


#     运行方式建议：nohup bash stage67_run.sh > stage67_run.log 2>&1 &。按上一轮耗时，五组推理合计约 40 小时（V2/A1 最慢），统一评测还需数小时；set -e 会在任一步失败时立即停止，避免拿半成品继续评测。