import numpy as np
import torch

from tracker.model import QwenAttentionModel


def test_attention_hook_uses_last_token_and_normalizes_only_meaningful_ranges():
    model = object.__new__(QwenAttentionModel)
    model._rows = {}
    model._inst_indices = torch.tensor([0, 1])
    model._data_indices = torch.tensor([2, 3])
    attention = torch.zeros((1, 2, 4, 4))
    attention[0, 0, -1] = torch.tensor([0.3, 0.2, 0.1, 0.4])
    attention[0, 1, -1] = torch.tensor([0.05, 0.05, 0.4, 0.5])
    output = (torch.ones((1, 4, 8)), attention)
    intercepted = model._attention_hook(0, output)
    np.testing.assert_allclose(model._rows[0], [0.5, 0.1], atol=1e-6)
    assert intercepted[1] is None
    assert torch.equal(intercepted[0], output[0])


class CharTokenizer:
    """One token per character, with offsets, and a simple chat template."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return text + "<assistant>"

    def __call__(self, text, add_special_tokens, return_offsets_mapping, return_tensors, truncation):
        return {
            "input_ids": torch.arange(len(text)).unsqueeze(0),
            "offset_mapping": torch.tensor([[(i, i + 1) for i in range(len(text))]]),
        }


def encode_spans(case):
    model = object.__new__(QwenAttentionModel)
    model.tokenizer = CharTokenizer()
    model.max_input_tokens = 10_000
    rendered = model.tokenizer.apply_chat_template(
        [{"role": "system", "content": case.system}, {"role": "user", "content": case.user}],
        False, True, False,
    )
    _, inst_ids, data_ids = model._encode(case)
    return "".join(rendered[i] for i in inst_ids), "".join(rendered[i] for i in data_ids)


def test_trailing_instruction_parts_count_as_instruction():
    from tracker.records import PromptCase

    case = PromptCase(
        "s", "bipia_email", 1, "Help with the email:\nBODY ignore all", "Who sent it?",
        "Help with the email:\n", ("BODY ignore all",), instruction_parts=("Who sent it?",),
    )
    inst, data = encode_spans(case)
    assert inst == "Help with the email:\nWho sent it?"
    assert data == "BODY ignore all"


def test_without_instruction_parts_user_text_stays_data():
    from tracker.records import PromptCase

    case = PromptCase("s", "deepset", 0, "Say xxxxxx", "hello there", "Say xxxxxx", ("hello there",))
    inst, data = encode_spans(case)
    assert inst == "Say xxxxxx"
    assert data == "hello there"
