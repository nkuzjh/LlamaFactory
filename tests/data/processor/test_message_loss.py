from copy import deepcopy

from llamafactory.data.processor import SupervisedDatasetProcessor
from llamafactory.data.template import TEMPLATES
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import DataArguments


class CharacterTokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return [1000 + ord(char) for char in text]


def build_processor(*, mask_history=False):
    return SupervisedDatasetProcessor(
        template=deepcopy(TEMPLATES["default"]),
        tokenizer=CharacterTokenizer(),
        processor=None,
        data_args=DataArguments(cutoff_len=4096, mask_history=mask_history),
    )


def example_messages():
    prompt = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "BAD", "loss": False},
        {"role": "observation", "content": "REJECTED"},
        {"role": "assistant", "content": "GOOD1", "loss": True},
        {"role": "observation", "content": "ACCEPTED"},
    ]
    response = [{"role": "assistant", "content": "GOOD2", "loss": True}]
    return prompt, response


def test_message_loss_masks_only_selected_assistant_turns():
    processor = build_processor()
    prompt, response = example_messages()
    input_ids, labels = processor._encode_data_example(prompt, response, None, None, [], [], [])
    pairs = processor.template.encode_multiturn(processor.tokenizer, prompt + response)
    offset = 0
    for turn_index, (source_ids, target_ids) in enumerate(pairs):
        offset += len(source_ids)
        target_labels = labels[offset : offset + len(target_ids)]
        if turn_index == 0:
            assert target_labels == [IGNORE_INDEX] * len(target_ids)
        else:
            assert target_labels == target_ids
        offset += len(target_ids)
    assert len(input_ids) == len(labels) == offset


def test_message_loss_stays_aligned_when_mask_history_reverses_turns():
    processor = build_processor(mask_history=True)
    prompt, response = example_messages()
    _, labels = processor._encode_data_example(prompt, response, None, None, [], [], [])
    supervised = [token for token in labels if token != IGNORE_INDEX]
    final_target = processor.template.encode_multiturn(processor.tokenizer, prompt + response)[-1][1]
    assert supervised == final_target
