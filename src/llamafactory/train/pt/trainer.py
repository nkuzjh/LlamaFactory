# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from types import MethodType
from typing import TYPE_CHECKING, Optional

import torch
from transformers import Trainer
from typing_extensions import override

from ...extras import logging
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, patch_accelerator_for_fp8, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, create_fast_length_grouped_sampler


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from ...hparams import FinetuningArguments, ModelArguments, TrainingArguments


logger = logging.get_logger(__name__)


def _supports_liger_skip_logits(model: "torch.nn.Module", model_args: Optional["ModelArguments"]) -> bool:
    r"""Return whether a Liger-patched base model explicitly accepts ``skip_logits``."""
    if model_args is None or not model_args.enable_liger_kernel:
        return False

    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    try:
        return "skip_logits" in inspect.signature(base_model.forward).parameters
    except (TypeError, ValueError):
        return False


class CustomTrainer(Trainer):
    r"""Inherit Trainer for custom optimizer."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        # Configure FP8 environment if enabled
        training_args: TrainingArguments = kwargs.get("args")
        if training_args.fp8:
            configure_fp8_environment(training_args)
            if getattr(training_args, "fp8_backend", "auto") == "te":
                patch_accelerator_for_fp8()

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        self._use_liger_loss_only_eval = _supports_liger_skip_logits(self.model, model_args)
        if model_args is not None and model_args.enable_liger_kernel and not self._use_liger_loss_only_eval:
            logger.warning_rank0(
                "Liger loss-only eval skip_logits is unavailable; falling back to the standard evaluation path."
            )

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if training_args.fp8 and hasattr(self, "accelerator"):  # verify FP8 status after trainer initialization
            verify_fp8_status(self.accelerator, training_args)

    @override
    def create_optimizer(self, *args, **kwargs) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer(*args, **kwargs)

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, train_dataset=None) -> Optional["torch.utils.data.Sampler"]:
        train_dataset = train_dataset if train_dataset is not None else self.train_dataset
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(train_dataset)

        sampler = create_fast_length_grouped_sampler(
            self,
            train_dataset,
            self.args.train_batch_size * self.args.gradient_accumulation_steps,
        )
        return sampler if sampler is not None else super()._get_train_sampler(train_dataset)

    @override
    def _get_eval_sampler(self, eval_dataset) -> Optional["torch.utils.data.Sampler"]:
        sampler = create_fast_length_grouped_sampler(self, eval_dataset, self.args.eval_batch_size)
        return sampler if sampler is not None else super()._get_eval_sampler(eval_dataset)

    @override
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if prediction_loss_only and self._use_liger_loss_only_eval:
            inputs = dict(inputs)
            inputs["skip_logits"] = True

        return super().prediction_step(
            model,
            inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
        )

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        return super().compute_loss(model, inputs, *args, **kwargs)
