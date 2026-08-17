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

from types import SimpleNamespace

from transformers import Trainer

from llamafactory.train.pt.trainer import CustomTrainer, _supports_liger_skip_logits


class _BaseModelWithSkipLogits:
    def forward(self, input_ids=None, skip_logits=None):
        pass


class _BaseModelWithoutSkipLogits:
    def forward(self, input_ids=None):
        pass


class _WrappedModel:
    def __init__(self, base_model):
        self.base_model = base_model

    def get_base_model(self):
        return self.base_model


def _model_args(enable_liger_kernel: bool = True) -> SimpleNamespace:
    return SimpleNamespace(enable_liger_kernel=enable_liger_kernel)


def test_skip_logits_capability_requires_liger_and_explicit_forward_parameter():
    supported = _WrappedModel(_BaseModelWithSkipLogits())
    unsupported = _WrappedModel(_BaseModelWithoutSkipLogits())

    assert _supports_liger_skip_logits(supported, _model_args())
    assert not _supports_liger_skip_logits(supported, _model_args(False))
    assert not _supports_liger_skip_logits(unsupported, _model_args())
    assert not _supports_liger_skip_logits(supported, None)


def test_loss_only_eval_injects_skip_logits_without_mutating_batch(monkeypatch):
    captured = {}

    def fake_prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        captured["inputs"] = inputs
        captured["prediction_loss_only"] = prediction_loss_only
        return "loss", None, None

    monkeypatch.setattr(Trainer, "prediction_step", fake_prediction_step)
    trainer = object.__new__(CustomTrainer)
    trainer._use_liger_loss_only_eval = True
    inputs = {"input_ids": [1, 2, 3]}

    result = trainer.prediction_step(None, inputs, prediction_loss_only=True)

    assert result == ("loss", None, None)
    assert captured["inputs"]["skip_logits"] is True
    assert captured["prediction_loss_only"] is True
    assert "skip_logits" not in inputs


def test_logits_eval_and_unsupported_models_preserve_original_inputs(monkeypatch):
    captured_inputs = []

    def fake_prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        captured_inputs.append(inputs)
        return None, "logits", None

    monkeypatch.setattr(Trainer, "prediction_step", fake_prediction_step)
    inputs = {"input_ids": [1, 2, 3]}

    trainer = object.__new__(CustomTrainer)
    trainer._use_liger_loss_only_eval = True
    trainer.prediction_step(None, inputs, prediction_loss_only=False)

    trainer._use_liger_loss_only_eval = False
    trainer.prediction_step(None, inputs, prediction_loss_only=True)

    assert captured_inputs == [inputs, inputs]
    assert all("skip_logits" not in captured for captured in captured_inputs)
