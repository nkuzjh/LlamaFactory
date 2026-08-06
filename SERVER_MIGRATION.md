# BrickNet / LlamaFactory / ms-swift 服务器迁移手册

更新时间：2026-07-31

本文档用于把当前三个项目迁移到新服务器，并恢复 PT → SFT → GRPO 主实验链路。命令默认：

- 在旧服务器执行第 1～5 节；
- 在新服务器执行第 6～11 节；
- 新服务器仍使用 `$HOME/task/{BrickNet,LlamaFactory,ms-swift}` 的相对布局；
- 不删除旧服务器上的任何内容，所有 `rsync` 命令都可以重复执行。

开始最终迁移前，必须等待或优雅停止正在运行的训练/数据构建任务。不要复制正在写入的
checkpoint、日志或 `.tmp` 文件。编写本文档时，旧服务器仍有 exp2_2、exp3_1 训练和
`prepare_bricknet_text_pt.py` 在运行，因此实际迁移时应以停止后的最新完整 checkpoint
为准。

## 0. 迁移范围

### 必须迁移

| 内容 | 当前来源 | 约占空间 |
| --- | --- | ---: |
| 三个 Git worktree，包含 `.git`、未提交修改和未跟踪文件 | `/home/jiahao/task/{BrickNet,LlamaFactory,ms-swift}` | 158 MiB |
| 已处理的 PT/SFT/VAL JSON、评测 JSONL 和 RL-2k JSONL | `BrickNet/outputs_preprocess` | 1.61 GiB |
| 已处理数据集引用的 PT/SFT/VAL 图片 | `BrickNet/outputs_preprocess/BrickNet-MM/images` | 24.77 GiB |
| BrickNet collision meshes | `/home/jiahao/.local/share/bricknet/inset` | 1.50 GiB |
| LDraw library 和 `captions_val.jsonl` | 当前 `/data` 挂载 | 0.49 GiB |
| PT-exp0、SFT-exp3、GRPO-exp0 三个 LoRA adapter | LlamaFactory/ms-swift 输出 | 0.40 GiB |

实际网络传输约 29 GiB。重建三个 Conda 环境和基础模型后约占 56 GiB；考虑训练输出和缓存，
新服务器至少预留 100 GiB，建议预留 200 GiB。

### 默认不迁移

- `/data/home/jiahao/data/bricknet_datasets` 的 182 GiB 原始数据；
- 176 GiB shuffled 原始数据；
- `BrickNet/outputs_gt`、`outputs_val`、`outputs_pt`、`tmp`；
- `outputs_preprocess` 中的 manifests、reports、mining shards、构建阶段重复 JSONL 和其他中间产物；
- LlamaFactory `.llamafactory_cache`、重复 checkpoint、生成预测和渲染产物；
- ms-swift TensorBoard、completions、渲染结果、optimizer state；
- 1.7 GiB 的 `Qwen3.5-0.8B-PT-exp0-merged`，它将在新服务器重新生成；
- `/data/home/jiahao/hf_checkpoints` 的完整 101 GiB 模型库。

当前代码基准：

| 项目 | Branch | Commit |
| --- | --- | --- |
| BrickNet | `main` | `9b858e920655f2924068d93270ac5791a73be74c` |
| LlamaFactory | `main` | `b455cb5559858e9881c9c30a97fa04c380da8cac` |
| ms-swift | `main` | `77f7f6d1596f96618df1c5aa480c14b44f38ea8a` |

由于三个 worktree 都有未提交或未跟踪内容，不能只在新服务器重新 `git clone`；必须复制当前
worktree。

## 1. 设置迁移变量

在旧服务器执行。必须把 `MIG_NEW_HOST` 和 `MIG_NEW_HOME` 改成新服务器的实际值：

```bash
set -euo pipefail

export MIG_OLD_HOME=/home/jiahao
export MIG_OLD_TASK_ROOT="${MIG_OLD_HOME}/task"

export MIG_NEW_HOST="new_user@new_server"
export MIG_NEW_HOME="/home/new_user"
export MIG_NEW_TASK_ROOT="${MIG_NEW_HOME}/task"

export MIG_META_DIR="${MIG_OLD_TASK_ROOT}/migration_meta"
```

确认源文件均存在：

```bash
test -d "${MIG_OLD_TASK_ROOT}/BrickNet/.git"
test -d "${MIG_OLD_TASK_ROOT}/LlamaFactory/.git"
test -d "${MIG_OLD_TASK_ROOT}/ms-swift/.git"
test -d /data/home/jiahao/data/bricknet_datasets/ldraw
test -d "${MIG_OLD_HOME}/.local/share/bricknet/inset"
```

## 2. 保存环境和 Git 清单

这些清单只用于重建与核对，不替代 worktree 复制。

先检查后台任务：

```bash
pgrep -af '[l]lamafactory-cli train|[s]wift rlhf|[p]repare_bricknet_text_pt.py' || true
```

如果有输出，等待任务完成，或从原启动终端用 `Ctrl-C` 优雅停止。再次执行上面的命令，
确认没有相关进程后再继续。若 text-PT 构建已经完成，
`LlamaFactory/data/BrickNet-PT_text_270102_seed42.jsonl` 及其 report 属于最终数据，
会随 LlamaFactory worktree 一起复制；其 170 GiB 原始 `paths_*.jsonl` 不需要迁移。

```bash
mkdir -p "${MIG_META_DIR}/env" "${MIG_META_DIR}/git"

for MIG_ENV_NAME in bricknet llamafactory swift; do
    "${MIG_OLD_HOME}/miniconda3/bin/conda" env export \
        -n "${MIG_ENV_NAME}" --no-builds |
        sed '/^prefix:/d' \
        > "${MIG_META_DIR}/env/${MIG_ENV_NAME}.yml"

    "${MIG_OLD_HOME}/miniconda3/envs/${MIG_ENV_NAME}/bin/python" \
        -m pip freeze \
        > "${MIG_META_DIR}/env/${MIG_ENV_NAME}.pip-freeze.txt"
done

for MIG_REPO_NAME in BrickNet LlamaFactory ms-swift; do
    git -C "${MIG_OLD_TASK_ROOT}/${MIG_REPO_NAME}" rev-parse HEAD \
        > "${MIG_META_DIR}/git/${MIG_REPO_NAME}.commit"
    git -C "${MIG_OLD_TASK_ROOT}/${MIG_REPO_NAME}" status --short \
        > "${MIG_META_DIR}/git/${MIG_REPO_NAME}.status"
    git -C "${MIG_OLD_TASK_ROOT}/${MIG_REPO_NAME}" diff --binary \
        > "${MIG_META_DIR}/git/${MIG_REPO_NAME}.patch"
done
```

当前环境的关键版本为：

| 环境 | Python | PyTorch | Transformers | PEFT | TRL | 其他 |
| --- | --- | --- | --- | --- | --- | --- |
| `bricknet` | 3.14.4 | 2.12.1 | 5.12.0 | 0.19.1 | - | meshlib 3.1.2.192 |
| `llamafactory` | 3.13.14 | 2.12.1 | 5.8.0 | 0.18.1 | 0.24.0 | LlamaFactory 0.9.6.dev0 |
| `swift` | 3.12.13 | 2.11.0 | 5.12.1 | 0.19.1 | 0.29.1 | vLLM 0.26.0、FLA 0.5.2 |

## 3. 在新服务器创建目标目录

仍在旧服务器执行：

```bash
ssh "${MIG_NEW_HOST}" "
set -e
mkdir -p \
  '${MIG_NEW_TASK_ROOT}/BrickNet' \
  '${MIG_NEW_TASK_ROOT}/LlamaFactory' \
  '${MIG_NEW_TASK_ROOT}/ms-swift' \
  '${MIG_NEW_TASK_ROOT}/migration_meta' \
  '${MIG_NEW_HOME}/.local/share/bricknet/inset'
"
```

## 4. 迁移三个项目的代码

这里保留 `.git`、本地修改和未跟踪代码，只排除数据、模型、cache 和生成输出。不要为这些
命令增加 `-L/--copy-links`，否则会沿绝对软链接复制整个 `/data`。

```bash
rsync -aH --partial --info=progress2 \
  --exclude='/outputs_preprocess/' \
  --exclude='/outputs_gt/' \
  --exclude='/outputs_val/' \
  --exclude='/outputs_pt/' \
  --exclude='/tmp/' \
  --exclude='/data/bricknet_datasets' \
  --exclude='/data/bricknet_datasets_shuffled' \
  --exclude='/hf_checkpoints' \
  --exclude='__pycache__/' \
  --exclude='/.ruff_cache/' \
  "${MIG_OLD_TASK_ROOT}/BrickNet/" \
  "${MIG_NEW_HOST}:${MIG_NEW_TASK_ROOT}/BrickNet/"

rsync -aH --partial --info=progress2 \
  --exclude='/saves/' \
  --exclude='/.llamafactory_cache/' \
  --exclude='/llamaboard_cache/' \
  --exclude='/data/BrickNet-MM_PT.json' \
  --exclude='/data/BrickNet-MM_SFT.json' \
  --exclude='/data/BrickNet-MM_VAL.json' \
  --exclude='/data/images' \
  --exclude='__pycache__/' \
  --exclude='/.ruff_cache/' \
  "${MIG_OLD_TASK_ROOT}/LlamaFactory/" \
  "${MIG_NEW_HOST}:${MIG_NEW_TASK_ROOT}/LlamaFactory/"

rsync -aH --partial --info=progress2 \
  --exclude='/output/' \
  --exclude='/models/' \
  --exclude='/data/BrickNet-MM-RL_n2000_seed42.jsonl' \
  --exclude='/ms_swift.egg-info/' \
  --exclude='__pycache__/' \
  --exclude='/.ruff_cache/' \
  "${MIG_OLD_TASK_ROOT}/ms-swift/" \
  "${MIG_NEW_HOST}:${MIG_NEW_TASK_ROOT}/ms-swift/"

rsync -aH --partial --info=progress2 \
  "${MIG_META_DIR}/" \
  "${MIG_NEW_HOST}:${MIG_NEW_TASK_ROOT}/migration_meta/"
```

## 5. 迁移最终数据、必要资产和最小权重

### 5.1 最终数据集

先创建只包含最终数据的目录：

```bash
ssh "${MIG_NEW_HOST}" "
set -e
mkdir -p \
  '${MIG_NEW_TASK_ROOT}/BrickNet/outputs_preprocess/BrickNet-MM/sharegpt' \
  '${MIG_NEW_TASK_ROOT}/BrickNet/outputs_preprocess/BrickNet-MM/images' \
  '${MIG_NEW_TASK_ROOT}/BrickNet/outputs_preprocess/BrickNet-MM-RL/samples' \
  '${MIG_NEW_TASK_ROOT}/BrickNet/data/bricknet_datasets/ldraw'
"
```

传输 PT/SFT/VAL JSON、评测脚本直接读取的 VAL JSONL、RL-2k JSONL 和已经整理好的图片：

```bash
rsync -aH --partial --info=progress2 \
  "${MIG_OLD_TASK_ROOT}/BrickNet/outputs_preprocess/BrickNet-MM/images/" \
  "${MIG_NEW_HOST}:${MIG_NEW_TASK_ROOT}/BrickNet/outputs_preprocess/BrickNet-MM/images/"

for MIG_DATASET_FILE in \
  BrickNet-MM_PT.json \
  BrickNet-MM_SFT.json \
  BrickNet-MM_VAL.json \
  BrickNet-MM_VAL.jsonl
do
  rsync -a --partial --info=progress2 \
    "${MIG_OLD_TASK_ROOT}/BrickNet/outputs_preprocess/BrickNet-MM/sharegpt/${MIG_DATASET_FILE}" \
    "${MIG_NEW_HOST}:${MIG_NEW_TASK_ROOT}/BrickNet/outputs_preprocess/BrickNet-MM/sharegpt/"
done

rsync -a --partial --info=progress2 \
  "${MIG_OLD_TASK_ROOT}/BrickNet/outputs_preprocess/BrickNet-MM-RL/samples/BrickNet-MM-RL_n2000_seed42.jsonl" \
  "${MIG_NEW_HOST}:${MIG_NEW_TASK_ROOT}/BrickNet/outputs_preprocess/BrickNet-MM-RL/samples/"
```

这些文件分别包含：

| 文件/目录 | 数量 |
| --- | ---: |
| `BrickNet-MM_PT.json` | 135,051 rows |
| `BrickNet-MM_SFT.json` | 334,355 rows |
| `BrickNet-MM_VAL.json` | 512 rows |
| `BrickNet-MM_VAL.jsonl` | 512 rows |
| `BrickNet-MM-RL_n2000_seed42.jsonl` | 2,000 rows |
| `images/PT` | 135,051 images |
| `images/SFT` | 67,178 images |
| `images/VAL` | 512 images |

### 5.2 从 `/data` 挂载中只迁移必要文件

不迁移完整的 182 GiB `bricknet_datasets`。评测只额外需要 108 KiB captions 和约
0.49 GiB LDraw library：

```bash
rsync -aH --partial --info=progress2 \
  /data/home/jiahao/data/bricknet_datasets/ldraw/ \
  "${MIG_NEW_HOST}:${MIG_NEW_TASK_ROOT}/BrickNet/data/bricknet_datasets/ldraw/"

rsync -a --partial --info=progress2 \
  /data/home/jiahao/data/bricknet_datasets/captions_val.jsonl \
  "${MIG_NEW_HOST}:${MIG_NEW_TASK_ROOT}/BrickNet/data/bricknet_datasets/"
```

### 5.3 碰撞检测 meshes

```bash
rsync -aH --partial --info=progress2 \
  "${MIG_OLD_HOME}/.local/share/bricknet/inset/" \
  "${MIG_NEW_HOST}:${MIG_NEW_HOME}/.local/share/bricknet/inset/"
```

目标目录应包含 21,084 个 `.ply` 文件。GRPO 正式训练不能省略该目录。

### 5.4 主实验最小 adapter

只迁移每个 PEFT adapter 的 `adapter_config.json` 和 `adapter_model.safetensors`。

```bash
export MIG_PT_ADAPTER_REL="saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64"
export MIG_SFT_ADAPTER_REL="saves/Qwen3.5-0.8B-Thinking/lora/train_exp3_qwen35_08b_pt_sft1w_ep3_bs2_ga8_lora64"
export MIG_GRPO_REL="output/bricknet_grpo/exp0_qwen35_08b_exp3_rl_n2000_g8"

ssh "${MIG_NEW_HOST}" "
set -e
mkdir -p \
  '${MIG_NEW_TASK_ROOT}/LlamaFactory/${MIG_PT_ADAPTER_REL}' \
  '${MIG_NEW_TASK_ROOT}/LlamaFactory/${MIG_SFT_ADAPTER_REL}' \
  '${MIG_NEW_TASK_ROOT}/ms-swift/${MIG_GRPO_REL}/checkpoint-1000'
"

for MIG_ADAPTER_REL in "${MIG_PT_ADAPTER_REL}" "${MIG_SFT_ADAPTER_REL}"; do
  rsync -a --info=progress2 \
    "${MIG_OLD_TASK_ROOT}/LlamaFactory/${MIG_ADAPTER_REL}/adapter_config.json" \
    "${MIG_OLD_TASK_ROOT}/LlamaFactory/${MIG_ADAPTER_REL}/adapter_model.safetensors" \
    "${MIG_NEW_HOST}:${MIG_NEW_TASK_ROOT}/LlamaFactory/${MIG_ADAPTER_REL}/"
done

rsync -a --info=progress2 \
  "${MIG_OLD_TASK_ROOT}/ms-swift/${MIG_GRPO_REL}/checkpoint-1000/adapter_config.json" \
  "${MIG_OLD_TASK_ROOT}/ms-swift/${MIG_GRPO_REL}/checkpoint-1000/adapter_model.safetensors" \
  "${MIG_OLD_TASK_ROOT}/ms-swift/${MIG_GRPO_REL}/checkpoint-1000/additional_config.json" \
  "${MIG_OLD_TASK_ROOT}/ms-swift/${MIG_GRPO_REL}/checkpoint-1000/README.md" \
  "${MIG_NEW_HOST}:${MIG_NEW_TASK_ROOT}/ms-swift/${MIG_GRPO_REL}/checkpoint-1000/"

rsync -a --info=progress2 \
  "${MIG_OLD_TASK_ROOT}/ms-swift/${MIG_GRPO_REL}/args.json" \
  "${MIG_OLD_TASK_ROOT}/ms-swift/${MIG_GRPO_REL}/logging.jsonl" \
  "${MIG_OLD_TASK_ROOT}/ms-swift/${MIG_GRPO_REL}/README.md" \
  "${MIG_NEW_HOST}:${MIG_NEW_TASK_ROOT}/ms-swift/${MIG_GRPO_REL}/"
```

不迁移 GRPO `optimizer.pt`、`scheduler.pt` 和 `rng_state.pth`，因为 exp0 已完成，后续实验
应从 adapter 权重新建实验，而不是继续沿用 exp0 的 optimizer state。

### 5.5 可选：保留其他 ablation 或恢复未完成训练

以下内容不属于主链路最小集合：

- 若要保留 exp2、exp2_1、exp3_0_1，只复制各目录根部的
  `adapter_config.json` 和 `adapter_model.safetensors`；
- 若要恢复未完成的 exp2_2 或 exp3_1，应在进程停止后完整复制最新 checkpoint；
- 不需要复制这些实验目录下重复的 final adapter + checkpoint 两份权重。

以下命令会自动选择两个实验中最新的完整 checkpoint。必须在第 2 节进程检查无输出后
执行：

```bash
for MIG_ACTIVE_EXP in \
  train_exp2_2_qwen35_08b_sft_ep3_bs2_ga8_lora64 \
  train_exp3_1_qwen35_08b_pt_sft5w_ep3_bs2_ga8_lora64
do
  MIG_ACTIVE_ROOT="${MIG_OLD_TASK_ROOT}/LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/${MIG_ACTIVE_EXP}"
  MIG_LATEST_CHECKPOINT=""

  while IFS= read -r MIG_CHECKPOINT_CANDIDATE; do
    if [[ -s "${MIG_CHECKPOINT_CANDIDATE}/adapter_model.safetensors" ]] &&
       [[ -s "${MIG_CHECKPOINT_CANDIDATE}/optimizer.pt" ]] &&
       [[ -s "${MIG_CHECKPOINT_CANDIDATE}/scheduler.pt" ]] &&
       [[ -s "${MIG_CHECKPOINT_CANDIDATE}/rng_state.pth" ]] &&
       [[ -s "${MIG_CHECKPOINT_CANDIDATE}/trainer_state.json" ]]; then
      MIG_LATEST_CHECKPOINT="${MIG_CHECKPOINT_CANDIDATE}"
      break
    fi
  done < <(
    find "${MIG_ACTIVE_ROOT}" -maxdepth 1 -type d -name 'checkpoint-*' -print |
      sort -Vr
  )

  if [[ -z "${MIG_LATEST_CHECKPOINT}" ]]; then
    echo "No complete checkpoint found for ${MIG_ACTIVE_EXP}; skipping"
    continue
  fi

  MIG_CHECKPOINT_NAME="$(basename "${MIG_LATEST_CHECKPOINT}")"
  ssh "${MIG_NEW_HOST}" \
    "mkdir -p '${MIG_NEW_TASK_ROOT}/LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/${MIG_ACTIVE_EXP}'"
  rsync -aH --partial --ignore-missing-args --info=progress2 \
    "${MIG_LATEST_CHECKPOINT}" \
    "${MIG_ACTIVE_ROOT}/trainer_log.jsonl" \
    "${MIG_NEW_HOST}:${MIG_NEW_TASK_ROOT}/LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/${MIG_ACTIVE_EXP}/"
  echo "Copied ${MIG_ACTIVE_EXP}/${MIG_CHECKPOINT_NAME}"
done
```

完整 checkpoint 包含 adapter、optimizer、scheduler、RNG 和 trainer state，可在新服务器
重建 tokenized cache 后用 LlamaFactory 的 `resume_from_checkpoint` 继续训练。

## 6. 新服务器：检查硬件和安装 Miniconda

以下命令开始在新服务器执行：

```bash
set -euo pipefail

export MIG_NEW_HOME="${HOME}"
export MIG_NEW_TASK_ROOT="${MIG_NEW_HOME}/task"

nvidia-smi
df -h "${MIG_NEW_HOME}"
```

当前 GRPO-exp0 在单卡上记录约 75.85 GiB 显存，原配置建议使用 80 GiB 以上 GPU。当前源
服务器是 RTX PRO 6000 Blackwell 96 GiB、NVIDIA driver 580.173.02。显存更小时必须降低
batch、completion 长度或 vLLM 显存占比，不能直接复用原配置。

若新服务器尚未安装 Conda：

```bash
cd "${MIG_NEW_HOME}"
curl -fsSLo miniconda.sh \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash miniconda.sh -b -p "${MIG_NEW_HOME}/miniconda3"
source "${MIG_NEW_HOME}/miniconda3/etc/profile.d/conda.sh"
conda config --set auto_activate_base false
```

安装常用系统工具；没有 sudo 权限时请让管理员安装：

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs rsync curl ripgrep
```

## 7. 新服务器：恢复软链接并修改绝对路径

### 7.1 数据软链接

```bash
replace_migration_link() {
    local MIG_LINK_PATH="$1"
    local MIG_LINK_TARGET="$2"

    if [[ -L "${MIG_LINK_PATH}" ]]; then
        unlink "${MIG_LINK_PATH}"
    elif [[ -e "${MIG_LINK_PATH}" ]]; then
        echo "Refusing to replace non-symlink: ${MIG_LINK_PATH}" >&2
        return 1
    fi
    ln -s "${MIG_LINK_TARGET}" "${MIG_LINK_PATH}"
}

replace_migration_link \
  "${MIG_NEW_TASK_ROOT}/LlamaFactory/data/BrickNet-MM_PT.json" \
  "../../BrickNet/outputs_preprocess/BrickNet-MM/sharegpt/BrickNet-MM_PT.json"
replace_migration_link \
  "${MIG_NEW_TASK_ROOT}/LlamaFactory/data/BrickNet-MM_SFT.json" \
  "../../BrickNet/outputs_preprocess/BrickNet-MM/sharegpt/BrickNet-MM_SFT.json"
replace_migration_link \
  "${MIG_NEW_TASK_ROOT}/LlamaFactory/data/BrickNet-MM_VAL.json" \
  "../../BrickNet/outputs_preprocess/BrickNet-MM/sharegpt/BrickNet-MM_VAL.json"
replace_migration_link \
  "${MIG_NEW_TASK_ROOT}/LlamaFactory/data/images" \
  "../../BrickNet/outputs_preprocess/BrickNet-MM/images"
replace_migration_link \
  "${MIG_NEW_TASK_ROOT}/ms-swift/data/BrickNet-MM-RL_n2000_seed42.jsonl" \
  "../../BrickNet/outputs_preprocess/BrickNet-MM-RL/samples/BrickNet-MM-RL_n2000_seed42.jsonl"
```

统一结果文档使用相对链接，复制后应自动有效：

```bash
test "$(readlink "${MIG_NEW_TASK_ROOT}/ms-swift/experiment_results.md")" \
  = "../LlamaFactory/experiment_results.md"
test -f "${MIG_NEW_TASK_ROOT}/ms-swift/experiment_results.md"
```

### 7.2 处理代码中的旧服务器绝对路径

推荐保持 `$HOME/task` 布局。若新用户名不是 `jiahao`，执行：

```bash
grep -rlZ '/home/jiahao' \
  "${MIG_NEW_TASK_ROOT}/BrickNet/scripts" \
  "${MIG_NEW_TASK_ROOT}/BrickNet/data_preprocess" \
  "${MIG_NEW_TASK_ROOT}/ms-swift/examples/train/grpo/plugin/bricknet" \
  "${MIG_NEW_TASK_ROOT}/LlamaFactory/record.md" \
  "${MIG_NEW_TASK_ROOT}/ms-swift/record.md" |
  xargs -0 -r sed -i "s|/home/jiahao|${MIG_NEW_HOME}|g"
```

检查仍会影响执行的旧路径：

```bash
grep -RIn '/home/jiahao' \
  "${MIG_NEW_TASK_ROOT}/BrickNet/scripts" \
  "${MIG_NEW_TASK_ROOT}/BrickNet/data_preprocess" \
  "${MIG_NEW_TASK_ROOT}/ms-swift/examples/train/grpo/plugin/bricknet" \
  || true
```

## 8. 新服务器：重建三个 Conda 环境

优先使用从旧服务器导出的精确环境：

```bash
source "${MIG_NEW_HOME}/miniconda3/etc/profile.d/conda.sh"

for MIG_ENV_FILE in \
  "${MIG_NEW_TASK_ROOT}/migration_meta/env/bricknet.yml" \
  "${MIG_NEW_TASK_ROOT}/migration_meta/env/llamafactory.yml" \
  "${MIG_NEW_TASK_ROOT}/migration_meta/env/swift.yml"
do
  sed -i "s|/home/jiahao|${MIG_NEW_HOME}|g" "${MIG_ENV_FILE}"
  sed -i '/^prefix:/d' "${MIG_ENV_FILE}"
  conda env create -f "${MIG_ENV_FILE}"
done
```

然后重新安装本地 editable 项目，避免使用旧路径或同名 PyPI 包：

```bash
conda run -n bricknet python -m pip install -e \
  "${MIG_NEW_TASK_ROOT}/BrickNet"

conda run -n llamafactory python -m pip install -e \
  "${MIG_NEW_TASK_ROOT}/LlamaFactory"
conda run -n llamafactory python -m pip install -r \
  "${MIG_NEW_TASK_ROOT}/LlamaFactory/requirements/metrics.txt"

conda run -n swift python -m pip install -e \
  "${MIG_NEW_TASK_ROOT}/ms-swift"
conda run -n swift python -m pip install -e \
  "${MIG_NEW_TASK_ROOT}/BrickNet"
conda run -n swift python -m pip install -r \
  "${MIG_NEW_TASK_ROOT}/ms-swift/examples/train/grpo/plugin/bricknet/requirements.txt"
```

如果精确 YAML 因目标 GPU、CUDA 或 package channel 不兼容而失败，使用仓库依赖干净重建：

```bash
conda create -y -n bricknet python=3.12
conda create -y -n llamafactory python=3.13
conda create -y -n swift python=3.12

conda run -n bricknet python -m pip install \
  torch torchvision transformers==5.12.0 open_clip_torch \
  qwen-vl-utils==0.0.14
conda run -n bricknet python -m pip install -e \
  "${MIG_NEW_TASK_ROOT}/BrickNet"

conda run -n llamafactory python -m pip install -e \
  "${MIG_NEW_TASK_ROOT}/LlamaFactory"
conda run -n llamafactory python -m pip install -r \
  "${MIG_NEW_TASK_ROOT}/LlamaFactory/requirements/metrics.txt"

conda run -n swift python -m pip install vllm==0.26.0
conda run -n swift python -m pip install \
  flash-linear-attention==0.5.2 qwen-vl-utils==0.0.14
conda run -n swift python -m pip install -e \
  "${MIG_NEW_TASK_ROOT}/ms-swift"
conda run -n swift python -m pip install -e \
  "${MIG_NEW_TASK_ROOT}/BrickNet"
conda run -n swift python -m pip install -r \
  "${MIG_NEW_TASK_ROOT}/ms-swift/examples/train/grpo/plugin/bricknet/requirements.txt"
```

验证关键版本和导入：

```bash
conda run -n bricknet python -c \
  "import bricknet, meshlib, torch; print(torch.__version__)"
conda run -n llamafactory llamafactory-cli version
conda run -n swift python -c \
  "import swift, vllm, meshlib, torch; print(torch.__version__, vllm.__version__)"
```

## 9. 新服务器：安装 LDView

完整图文评测需要 LDView；仅训练或 `--skip-image-metrics` 时可以暂不安装。

```bash
bash "${MIG_NEW_TASK_ROOT}/BrickNet/eval/ldview_install.sh"
test -x "${MIG_NEW_HOME}/.local/bin/ldview"
test -d "${MIG_NEW_TASK_ROOT}/BrickNet/data/bricknet_datasets/ldraw"
```

## 10. 新服务器：最小预训练模型方案

### 10.1 主训练链路只下载一个基础模型

不要迁移完整 Hugging Face cache，也不要迁移 PT-merged 模型。只下载固定 revision：

```bash
conda run -n swift hf download \
  Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17
```

网络不可用时，可以改为从旧服务器复制唯一的 1.7 GiB cache：

```bash
# 在旧服务器执行，和在线下载二选一。
ssh "${MIG_NEW_HOST}" \
  "mkdir -p '${MIG_NEW_HOME}/.cache/huggingface/hub'"

rsync -aH --partial --info=progress2 \
  /home/jiahao/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/ \
  "${MIG_NEW_HOST}:${MIG_NEW_HOME}/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/"
```

不需要迁移 Qwen3.5-2B/4B/9B、Qwen3-0.6B/1.7B/4B/8B/14B 等历史基线权重。

### 10.2 可选：完整图文评测模型

训练、推理、BLEU/ROUGE、parse、collision 和 GRPO alignment 指标都不需要以下模型。
只有重新计算 PE、SigLIP2、VQAScore 时才执行：

```bash
mkdir -p "${MIG_NEW_TASK_ROOT}/BrickNet/hf_checkpoints/timm"
mkdir -p "${MIG_NEW_TASK_ROOT}/BrickNet/hf_checkpoints/google"
mkdir -p "${MIG_NEW_TASK_ROOT}/BrickNet/hf_checkpoints/Qwen"

conda run -n bricknet hf download \
  timm/PE-Core-bigG-14-448 \
  --exclude "*.bin" \
  --local-dir "${MIG_NEW_TASK_ROOT}/BrickNet/hf_checkpoints/timm/PE-Core-bigG-14-448"

conda run -n bricknet hf download \
  google/siglip2-giant-opt-patch16-384 \
  --local-dir "${MIG_NEW_TASK_ROOT}/BrickNet/hf_checkpoints/google/siglip2-giant-opt-patch16-384"

conda run -n bricknet hf download \
  Qwen/Qwen2-VL-7B-Instruct \
  --local-dir "${MIG_NEW_TASK_ROOT}/BrickNet/hf_checkpoints/Qwen/Qwen2-VL-7B-Instruct"
```

PE 只保留 safetensors，排除重复的 9 GiB `.bin`。三项可选权重合计约 32 GiB，而不是迁移
当前 101 GiB `hf_checkpoints`。

## 11. 新服务器：重建派生产物并验收

### 11.1 重新生成 PT-merged base 和 adapter view

```bash
source "${MIG_NEW_HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate swift
cd "${MIG_NEW_TASK_ROOT}/ms-swift"

export MIG_BASE_REVISION=2fc06364715b967f1860aea9cf38778875588b17
export MIG_BASE_SNAPSHOT="${MIG_NEW_HOME}/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/${MIG_BASE_REVISION}"
test -f "${MIG_BASE_SNAPSHOT}/config.json"

BASE_MODEL="${MIG_BASE_SNAPSHOT}" \
  bash examples/train/grpo/plugin/bricknet/prepare_exp3_base.sh
```

该命令应生成：

```text
ms-swift/models/Qwen3.5-0.8B-PT-exp0-merged/
ms-swift/models/Qwen3.5-0.8B-PT-exp0-adapter/
ms-swift/models/Qwen3.5-0.8B-exp3-adapter/
```

这里显式传入 snapshot 路径，避免以后 Hugging Face 仓库的 `main` 更新时静默换用其他
revision。

### 11.2 检查数据与软链接

```bash
test -f "${MIG_NEW_TASK_ROOT}/LlamaFactory/data/BrickNet-MM_PT.json"
test -f "${MIG_NEW_TASK_ROOT}/LlamaFactory/data/BrickNet-MM_SFT.json"
test -f "${MIG_NEW_TASK_ROOT}/LlamaFactory/data/BrickNet-MM_VAL.json"
test -f "${MIG_NEW_TASK_ROOT}/BrickNet/outputs_preprocess/BrickNet-MM/sharegpt/BrickNet-MM_VAL.jsonl"
test -d "${MIG_NEW_TASK_ROOT}/LlamaFactory/data/images"
test -f "${MIG_NEW_TASK_ROOT}/ms-swift/data/BrickNet-MM-RL_n2000_seed42.jsonl"

test "$(find "${MIG_NEW_TASK_ROOT}/BrickNet/outputs_preprocess/BrickNet-MM/images/PT" -type f | wc -l)" -eq 135051
test "$(find "${MIG_NEW_TASK_ROOT}/BrickNet/outputs_preprocess/BrickNet-MM/images/SFT" -type f | wc -l)" -eq 67178
test "$(find "${MIG_NEW_TASK_ROOT}/BrickNet/outputs_preprocess/BrickNet-MM/images/VAL" -type f | wc -l)" -eq 512
test "$(wc -l < "${MIG_NEW_TASK_ROOT}/ms-swift/data/BrickNet-MM-RL_n2000_seed42.jsonl")" -eq 2000
test "$(find "${MIG_NEW_HOME}/.local/share/bricknet/inset" -type f | wc -l)" -eq 21084
```

### 11.3 校验关键文件 SHA-256

如果是新登录 shell，先定义：

```bash
export MIG_PT_ADAPTER_REL="saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64"
export MIG_SFT_ADAPTER_REL="saves/Qwen3.5-0.8B-Thinking/lora/train_exp3_qwen35_08b_pt_sft1w_ep3_bs2_ga8_lora64"
export MIG_GRPO_REL="output/bricknet_grpo/exp0_qwen35_08b_exp3_rl_n2000_g8"
```

```bash
printf '%s  %s\n' \
  9daa4703ae8e56afec862ce4fbe6cf9344422542b614c6327fcc09690eb4c055 \
  "${MIG_NEW_TASK_ROOT}/BrickNet/outputs_preprocess/BrickNet-MM/sharegpt/BrickNet-MM_PT.json" \
  8c3425d355392684f3c7bbc5c275feae82459b57575b7cfa7cb9cd94dcf4d3b2 \
  "${MIG_NEW_TASK_ROOT}/BrickNet/outputs_preprocess/BrickNet-MM/sharegpt/BrickNet-MM_SFT.json" \
  69e70080a8ccfd243d653df823e9eb6596f9ffc6b6e1b2a72e2fb884ffc21735 \
  "${MIG_NEW_TASK_ROOT}/BrickNet/outputs_preprocess/BrickNet-MM/sharegpt/BrickNet-MM_VAL.json" \
  30e6a61be0b866e251ce848b2676ec82cc704748a6dd700a79a5931ef9b9165b \
  "${MIG_NEW_TASK_ROOT}/BrickNet/outputs_preprocess/BrickNet-MM/sharegpt/BrickNet-MM_VAL.jsonl" \
  58a0bec06338cf0052c026d35f0968fc00a4871b7399c6c67acfee4a98cb6dfd \
  "${MIG_NEW_TASK_ROOT}/BrickNet/outputs_preprocess/BrickNet-MM-RL/samples/BrickNet-MM-RL_n2000_seed42.jsonl" \
  b9edaabbf19a61ac5fc313353504e5842537011453ae265b064f16a81edac716 \
  "${MIG_NEW_TASK_ROOT}/BrickNet/data/bricknet_datasets/captions_val.jsonl" \
  9a7f48d24bde51906bb990c4109ff4fd9d161984dfe6b912b3c18712821b3670 \
  "${MIG_NEW_TASK_ROOT}/LlamaFactory/${MIG_PT_ADAPTER_REL}/adapter_model.safetensors" \
  c849e6984bc6d3dd7836787c5532125b1e505303557f2ee46f73eb7050b37fbd \
  "${MIG_NEW_TASK_ROOT}/LlamaFactory/${MIG_SFT_ADAPTER_REL}/adapter_model.safetensors" \
  92611d12ef3c45474f4a236f1df44e92542dd8c086f3845625a2fef085ceb8be \
  "${MIG_NEW_TASK_ROOT}/ms-swift/${MIG_GRPO_REL}/checkpoint-1000/adapter_model.safetensors" |
  sha256sum -c -
```

图片数量很多，不逐个在文档中保存 hash。传输完成后可以在旧服务器把第 5.1 节的图片
`rsync` 命令改为 `rsync -aHnci` 再执行一次；没有输出即表示目标图片内容一致。

### 11.4 Reward 和评测 dry-run

```bash
cd "${MIG_NEW_TASK_ROOT}/ms-swift"
export BRICKNET_DATA="${MIG_NEW_HOME}/.local/share/bricknet"
export BRICKNET_ROOT="${MIG_NEW_TASK_ROOT}/BrickNet"
export ROOT_IMAGE_DIR="${MIG_NEW_TASK_ROOT}/BrickNet"

conda run -n swift python \
  examples/train/grpo/plugin/bricknet/verify_reward.py

bash examples/train/grpo/plugin/bricknet/evaluate_exp0_qwen35_08b_exp3.sh \
  --skip-image-metrics \
  --dry-run
```

`verify_reward.py` 的七项输出应全部为 `1.0`。dry-run 通过后，迁移已满足训练、GRPO 推理
和非图文评测要求。

### 11.5 运行推理或重新评测

GRPO 交互推理：

```bash
cd "${MIG_NEW_TASK_ROOT}/ms-swift"
EXP0_CHECKPOINT="${MIG_NEW_TASK_ROOT}/ms-swift/${MIG_GRPO_REL}/checkpoint-1000" \
bash examples/train/grpo/plugin/bricknet/infer_exp0_qwen35_08b_exp3.sh
```

安装第 10.2 节可选模型后，重新执行完整评测：

```bash
bash examples/train/grpo/plugin/bricknet/evaluate_exp0_qwen35_08b_exp3.sh
```

需要重新训练 GRPO 时：

```bash
bash examples/train/grpo/plugin/bricknet/grpo_exp0_qwen35_08b_exp3.sh
```

请为新训练修改 `output_dir` 和实验编号，避免覆盖已迁移的 exp0。
