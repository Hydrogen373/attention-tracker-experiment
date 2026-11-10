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
