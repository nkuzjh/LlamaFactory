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

import numpy as np
from datasets import Dataset

from llamafactory.train.trainer_utils import create_fast_length_grouped_sampler


def _trainer(strategy: str = "group_by_length") -> SimpleNamespace:
    return SimpleNamespace(
        args=SimpleNamespace(train_sampling_strategy=strategy, length_column_name="length"),
        processing_class=SimpleNamespace(model_input_names=["input_ids"]),
    )


def test_fast_length_grouped_sampler_uses_persisted_arrow_column():
    dataset = Dataset.from_dict(
        {
            "input_ids": [[1], [2, 3, 4], [5, 6]],
            "length": [1, 3, 2],
        }
    )

    sampler = create_fast_length_grouped_sampler(_trainer(), dataset, batch_size=2)

    assert sampler is not None
    assert isinstance(sampler.lengths, np.ndarray)
    assert sampler.lengths.dtype == np.int64
    indices = list(iter(sampler))
    assert len(indices) == len(dataset)
    assert sampler.lengths[indices[0]] == 3


def test_fast_length_grouped_sampler_preserves_standard_fallbacks():
    plain_dataset = Dataset.from_dict({"input_ids": [[1], [2, 3]]})
    length_dataset = plain_dataset.add_column("length", [1, 2])

    assert create_fast_length_grouped_sampler(_trainer(), plain_dataset, batch_size=2) is None
    assert create_fast_length_grouped_sampler(_trainer("random"), length_dataset, batch_size=2) is None
