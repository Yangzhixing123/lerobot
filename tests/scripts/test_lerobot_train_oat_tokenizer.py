from types import SimpleNamespace

import torch

from lerobot.scripts.lerobot_train_oat_tokenizer import _ActionWindowDataset
from lerobot.utils.constants import ACTION


class _ActionOnlyDataset:
    def __init__(self) -> None:
        self._actions = [torch.tensor([value], dtype=torch.float32) for value in range(5)]
        self.meta = SimpleNamespace(
            episodes=[
                {"dataset_from_index": 0, "dataset_to_index": 2},
                {"dataset_from_index": 2, "dataset_to_index": 5},
            ]
        )

    def select_columns(self, column_name: str) -> dict[str, list[torch.Tensor]]:
        assert column_name == ACTION
        return {ACTION: self._actions}


def test_action_windows_do_not_cross_episode_boundaries() -> None:
    dataset = _ActionWindowDataset(_ActionOnlyDataset(), horizon=3)

    assert torch.equal(dataset[0][ACTION].squeeze(-1), torch.tensor([0.0, 1.0, 1.0]))
    assert torch.equal(dataset[1][ACTION].squeeze(-1), torch.tensor([1.0, 1.0, 1.0]))
    assert torch.equal(dataset[2][ACTION].squeeze(-1), torch.tensor([2.0, 3.0, 4.0]))
    assert torch.equal(dataset[4][ACTION].squeeze(-1), torch.tensor([4.0, 4.0, 4.0]))
