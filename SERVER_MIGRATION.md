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

export MIG_NEW_HOST="jiahao@10.119.146.65"
export MIG_NEW_HOME="/data/jiahao"
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

## 2. 保存环境和 Git 清单 (根据实际情况自行使用conda和git迁移)

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

### 2.1. 在新服务器创建目标目录

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

### 2.2. 迁移三个项目的代码

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

## 3. 迁移最终数据、必要资产和最小权重

### 3.1 最终数据集

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

### 3.2 从 `/data` 挂载中只迁移必要文件

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

### 3.3 碰撞检测 meshes

```bash
rsync -aH --partial --info=progress2 \
  "${MIG_OLD_HOME}/.local/share/bricknet/inset/" \
  "${MIG_NEW_HOST}:${MIG_NEW_HOME}/.local/share/bricknet/inset/"
```

目标目录应包含 21,084 个 `.ply` 文件。GRPO 正式训练不能省略该目录。

### 3.4 主实验最小 adapter (根据实际情况使用rsync自行迁移)

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

### 3.5 可选：保留其他 ablation 或恢复未完成训练

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

## 4. 新服务器：检查硬件和安装 Miniconda

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

## 5. 新服务器：恢复软链接并修改绝对路径

### 5.1 数据软链接

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

### 5.2 处理代码中的旧服务器绝对路径

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

## 6. 新服务器：重建三个 Conda 环境

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

## 7. 新服务器：安装 LDView

完整图文评测需要 LDView；仅训练或 `--skip-image-metrics` 时可以暂不安装。

```bash
bash "${MIG_NEW_TASK_ROOT}/BrickNet/eval/ldview_install.sh"
test -x "${MIG_NEW_HOME}/.local/bin/ldview"
test -d "${MIG_NEW_TASK_ROOT}/BrickNet/data/bricknet_datasets/ldraw"
```

## 8. 新服务器：最小预训练模型方案

### 8.1 主训练链路只下载一个基础模型

不要迁移完整 Hugging Face cache，也不要迁移 PT-merged 模型。只下载固定 revision：

```bash
# 下载hf cli
curl -LsSf https://hf.co/cli/install.sh | bash

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

### 8.2 可选：完整图文评测模型

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

## 9. 新服务器：重建派生产物并验收

### 9.1 重新生成 PT-merged base 和 adapter view (根据实验进度自行选择权重使用rsync迁移)

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

### 9.2 检查数据与软链接

```bash
test -f "/data/jiahao/task/LlamaFactory/data/BrickNet-MM_PT.json"
test -f "/data/jiahao/task/LlamaFactory/data/BrickNet-MM_SFT.json"
test -f "/data/jiahao/task/LlamaFactory/data/BrickNet-MM_VAL.json"
test -f "/data/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM/sharegpt/BrickNet-MM_VAL.jsonl"
test -d "/data/jiahao/task/LlamaFactory/data/images"
test -f "/data/jiahao/task/ms-swift/data/BrickNet-MM-RL_n2000_seed42.jsonl"

test "$(find "/data/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM/images/PT" -type f | wc -l)" -eq 135051
test "$(find "/data/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM/images/SFT" -type f | wc -l)" -eq 67178
test "$(find "/data/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM/images/VAL" -type f | wc -l)" -eq 512
test "$(wc -l < "/data/jiahao/task/ms-swift/data/BrickNet-MM-RL_n2000_seed42.jsonl")" -eq 2000
test "$(find "/home/jiahao/.local/share/bricknet/inset" -type f | wc -l)" -eq 21084
```

### 9.3 校验关键文件 SHA-256

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

### 9.4 Reward 和评测 dry-run

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

### 9.5 运行推理或重新评测 (根据实验进度自行决定是否需要重跑实验)

GRPO 交互推理：

```bash
cd "${MIG_NEW_TASK_ROOT}/ms-swift"
EXP0_CHECKPOINT="${MIG_NEW_TASK_ROOT}/ms-swift/${MIG_GRPO_REL}/checkpoint-1000" \
bash examples/train/grpo/plugin/bricknet/infer_exp0_qwen35_08b_exp3.sh
```

安装第 8.2 节可选模型后，重新执行完整评测：

```bash
bash examples/train/grpo/plugin/bricknet/evaluate_exp0_qwen35_08b_exp3.sh
```

需要重新训练 GRPO 时：

```bash
bash examples/train/grpo/plugin/bricknet/grpo_exp0_qwen35_08b_exp3.sh
```

请为新训练修改 `output_dir` 和实验编号，避免覆盖已迁移的 exp0。

## 10. 2026-09-06 官方 SFT 媒体渲染交接补充

本节是对上文旧迁移范围的 dated supplement。上文把 `outputs_gt` 和渲染产物列为默认不迁移；
本轮用户明确要求迁移下面这组已经完成的官方 SFT 媒体，因此本节只对这组新产物作例外说明，
不改变 PT/SFT/GRPO 的旧迁移策略。本轮 official SFT 收尾的选择性同步不要求默认修改或同步
`ms-swift`；只有用户另行要求完整 PT/SFT/GRPO 迁移时，才执行上文三仓库迁移方案。上文
第 9.2/9.3 节的检查仍是 legacy v1 images/和旧迁移的检查，不替代本节的 official SFT
raw/collage strict verify 与 completion gate。

### 10.1 已完成内容和边界

正式监督器状态为 `COMPLETE/OK`，strict verify 与 completion gate 均为 `PASS`：

```text
/data/jiahao/task/LlamaFactory/tmp_bash/supervise_official_sft_render.status
run_id=20260902T194605Z-381805
rows=67178
raw_views=537424 (67178 x 8)
collages=67178
failed=0, skipped=0, stitched=0
```

本轮只覆盖 `SFT`。`PT` 和 `VAL` 不在本轮渲染范围内；本轮仅完成 validated 媒体层及其 provenance，
projection、dataset registry、token cache 和训练接线仍为 pending。现有 PT/SFT/评测仍使用 v1 images
路径和注册，不应因为这些 v2 official 文件已经存在就自动改绑数据集。

主要产物：

```text
/data/jiahao/task/BrickNet/outputs_gt/sft_8view_renders_v2_rowids
/data/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM/image_v2_official/SFT
```

第一条是每个 row 的 8 张独立视角图，命名为 `<row_id>_0000.png` 至
`<row_id>_0007.png`；第二条是每个 row 的 4x2 拼接图 `<row_id>.png`。正式 metadata namespace
及摘要为：

```text
/data/jiahao/task/BrickNet/outputs_gt/.bricknet_render_v2_official/sft_identity.json
/data/jiahao/task/BrickNet/outputs_gt/.bricknet_render_v2_official/sft_progress.json
/data/jiahao/task/BrickNet/outputs_gt/.bricknet_render_v2_official/sft_summary.json
/data/jiahao/task/BrickNet/outputs_gt/.bricknet_render_v2_official/sft_timings.jsonl
```

`sft.lock` 是可变的 render/verify sentinel，不是迁移身份文件；迁移时显式列出上面四个 metadata
文件，排除 `sft.lock`。当前已生成并验证通过的
`/data/jiahao/task/BrickNet/outputs_gt/.bricknet_render_v2_official/sft_completion_gate.json` 必须
显式加入同步清单；不能以缺失 gate 的媒体副本声称 validated。

### 10.2 渲染身份、配置和证据入口

本轮使用 CYCLES/OPTIX、物理 GPU `0,1`、8 views、`512x512`、256 samples、seed `0`。
配置文件是：

```text
/data/jiahao/task/BrickNet/configs/bricknet_mm_image_v2_official_render.json
```

配置中的 `workers_per_gpu` 默认值为 `8`；正式 supervisor 通过 CLI 覆盖为实际
`16 workers/GPU`。迁移或复现时必须同时记录这两个值，不能把 config 默认值误写成实际运行值。

代码和 gate 证据入口：

```text
/data/jiahao/task/BrickNet/scripts/render_bricknet_render_8views.py
/data/jiahao/task/BrickNet/scripts/render_bricknet_render_8views_official.sh
/data/jiahao/task/BrickNet/scripts/generate_bricknet_sft_completion_gate.py
/data/jiahao/task/LlamaFactory/tmp_bash/supervise_official_sft_render.sh
/data/jiahao/task/LlamaFactory/tmp_bash/supervise_official_sft_render-20260902T194605Z-381805.log
/data/jiahao/task/LlamaFactory/tmp_bash/supervise_official_sft_render-20260902T194605Z-381805.gpu.tsv
/data/jiahao/task/LlamaFactory/tmp_bash/supervise_official_sft_render-20260902T194605Z-381805.gpu-peaks.tsv
/data/jiahao/task/LlamaFactory/tmp_bash/sft_official_verify_20260906.log
/data/jiahao/task/BrickNet/outputs_gt/.bricknet_render_v2_official/sft_completion_gate.json
```

verify log SHA-256=`9afecbc4717484dd55331b93f496045eff24b2079881e133242b59625d22166a`；completion
gate SHA-256=`7b68c82e1116aa72d7168c6087858168ca886c57413f841ac3f93b465e325f97`，生成于
`2026-09-06 02:39 +08:00`。raw/collage logical bytes 为 `119394717747` / `22019373017`，
selected row IDs SHA-256=`c9d740a06f550b750e1fa6af58d3f874e8ec75be87864c0272cb7483b28a62b8`。

当前严格 verify 证据日志为：

```text
/data/jiahao/task/LlamaFactory/tmp_bash/sft_official_verify_20260906.log
```

只有该文件包含：

```text
[SFT] verify passed: 67178 rows, exactly 8 views and valid 1024x512 RGB collages
```

已出现该 marker，strict verify 已 exit `0`，且 completion gate 当前为 `PASS`；上述 gate 文件和
verify log 必须在迁移时显式同步。若未来重新运行 verify，仍须重新检查 marker、gate SHA 和 gate schema，
不得手工创建或改写 `sft_completion_gate.json`。

### 10.3 迁移注意事项

源机的 `/home/jiahao/task` 当前解析到 `/data/jiahao/task`；目标机不一定有同样的 symlink。
媒体副本必须保留源 metadata 作为不可变 provenance；如果目标路径不同，只用 rsync checksum/结构
gate 验证同步，禁止手工改写已完成 run 的 `sft_identity.json`、`sft_summary.json` 或 gate，不能
通过改路径冒充同一 run。若要在目标机继续 render 或运行 identity-bound strict verify，必须保持相同
canonical layout，或新建目标机专用 config 并走明确的 refreeze/new identity 流程。复现渲染还需要
与 identity 一致的 BrickNet-Render commit、renderer Python 环境、GLB 库、`sft.npz`、SFT source
images 和 ShareGPT source JSON；只复制 PNG 不能复现。

`sft.npz`、SFT source images、ShareGPT source JSON 和 GLB inventory 的 hash/数量以
`sft_identity.json` 为准。目标端还要确认空间足够：正式 raw views 约 113G，collages 约 21G，
metadata 约 10M；本轮 SFT 同步和完整性验证建议至少保留 160 GiB 可用空间，且 rsync
过程中不得使用 `--delete`。

pilot 目录
`/data/jiahao/task/BrickNet-Render/tmp/official_sft_pilot_20260902T194605Z-381805`
只包含 32 个验收 row（256 raw views、32 collages）；它是 provenance，不是正式全量数据。
并发测试目录和 `car` acceptance 图片同样不应被当作 SFT 全量输入。可执行的 gate、迁移、
dry-run 和目标端计数命令见 `record.md` 的同日期补充。
