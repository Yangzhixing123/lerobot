#!/usr/bin/env python

"""Train an OAT FSQ or paired-RFSQ action tokenizer on a LeRobotDataset."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from safetensors.torch import save_model
from torch.utils.data import DataLoader, Dataset

from lerobot.configs import parser
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.oat.action_tokenizer import OATActionTokenizer
from lerobot.policies.oat_fsq.configuration_oat_fsq import OATFSQConfig
from lerobot.policies.oat_rfsq_pair.configuration_oat_rfsq_pair import OATRFSQPairConfig
from lerobot.utils.constants import ACTION
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


class _ActionWindowDataset(Dataset):
    """Serve episode-aware action windows without decoding visual observations."""

    def __init__(self, dataset, horizon: int) -> None:
        if horizon < 1:
            raise ValueError("horizon must be positive.")

        action_column = dataset.select_columns(ACTION)[ACTION]
        self.actions = torch.stack(list(action_column)).contiguous()
        self.horizon = horizon

        # Store each frame's exclusive episode end so windows can repeat the final
        # action instead of crossing into the next episode. This matches the padding
        # behavior of LeRobotDataset delta timestamps.
        self.episode_end_indices = torch.full((len(self.actions),), -1, dtype=torch.long)
        for episode in dataset.meta.episodes:
            start = int(episode["dataset_from_index"])
            end = int(episode["dataset_to_index"])
            if not 0 <= start < end <= len(self.actions):
                raise ValueError(
                    f"Invalid episode bounds [{start}, {end}) for an action table with "
                    f"{len(self.actions)} frames."
                )
            self.episode_end_indices[start:end] = end

        if (self.episode_end_indices < 0).any():
            raise ValueError("Episode metadata does not cover every action frame.")

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_end = int(self.episode_end_indices[index])
        window_end = min(index + self.horizon, episode_end)
        actions = self.actions[index:window_end]
        if len(actions) < self.horizon:
            padding = actions[-1:].expand(self.horizon - len(actions), -1)
            actions = torch.cat((actions, padding), dim=0)
        return {ACTION: actions}


@dataclass
class OATTokenizerTrainingConfig:
    repo_id: str
    output_dir: str
    policy_type: str = "oat_fsq"
    root: str | None = None
    horizon: int = 32
    latent_horizon: int = 8
    fsq_levels: tuple[int, ...] = (8, 5, 5, 5)
    tokenizer_dim: int = 256
    tokenizer_encoder_layers: int = 2
    tokenizer_decoder_layers: int = 4
    tokenizer_heads: int = 4
    tokenizer_dropout: float = 0.1
    tokenizer_token_dropout_mode: str = "pow2"
    batch_size: int = 256
    num_workers: int = 4
    steps: int = 100_000
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 100
    grad_clip_norm: float = 1.0
    log_freq: int = 100
    save_freq: int = 10_000
    device: str = "cuda"


def _normalize_actions(actions: torch.Tensor, stats: dict) -> torch.Tensor:
    minimum = torch.as_tensor(stats["min"], dtype=actions.dtype, device=actions.device)
    maximum = torch.as_tensor(stats["max"], dtype=actions.dtype, device=actions.device)
    return 2 * (actions - minimum) / (maximum - minimum).clamp_min(1e-8) - 1


def _save_tokenizer(tokenizer: OATActionTokenizer, output_dir: Path, cfg: OATTokenizerTrainingConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_model(tokenizer, str(output_dir / "action_tokenizer.safetensors"))
    with open(output_dir / "tokenizer_config.json", "w") as file:
        json.dump(asdict(cfg), file, indent=2)


@parser.wrap()
def train_oat_tokenizer(cfg: OATTokenizerTrainingConfig) -> None:
    # Keep --help/config parsing available in minimal LeRobot environments. The dataset extra is
    # required only once training actually starts.
    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    if cfg.policy_type not in {"oat_fsq", "oat_rfsq_pair"}:
        raise ValueError("policy_type must be 'oat_fsq' or 'oat_rfsq_pair'.")
    if cfg.steps < 1 or cfg.log_freq < 1 or cfg.save_freq < 1:
        raise ValueError("steps, log_freq, and save_freq must be positive.")
    if cfg.warmup_steps < 0 or cfg.grad_clip_norm <= 0:
        raise ValueError("warmup_steps must be non-negative and grad_clip_norm must be positive.")
    requested_device = cfg.device if torch.cuda.is_available() or not cfg.device.startswith("cuda") else "cpu"
    device = torch.device(requested_device)
    meta = LeRobotDatasetMetadata(cfg.repo_id, root=cfg.root)
    if ACTION not in meta.features:
        raise ValueError(f"Dataset {cfg.repo_id!r} has no {ACTION!r} feature.")
    action_dim = meta.features[ACTION]["shape"][0]
    # The tokenizer only consumes action chunks. Avoid LeRobotDataset.__getitem__,
    # which would otherwise decode every visual observation even though it is unused.
    source_dataset = LeRobotDataset(cfg.repo_id, root=cfg.root, download_videos=False)
    dataset = _ActionWindowDataset(source_dataset, cfg.horizon)
    if not meta.stats or ACTION not in meta.stats:
        raise ValueError("OAT tokenizer training requires action min/max statistics in the dataset metadata.")
    action_stats = meta.stats[ACTION]
    if "min" not in action_stats or "max" not in action_stats:
        raise ValueError("OAT MIN_MAX normalization requires action 'min' and 'max' statistics.")

    config_class = OATFSQConfig if cfg.policy_type == "oat_fsq" else OATRFSQPairConfig
    policy_config = config_class(
        device=str(device),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        horizon=cfg.horizon,
        n_action_steps=min(16, cfg.horizon),
        latent_horizon=cfg.latent_horizon,
        fsq_levels=cfg.fsq_levels,
        tokenizer_dim=cfg.tokenizer_dim,
        tokenizer_encoder_layers=cfg.tokenizer_encoder_layers,
        tokenizer_decoder_layers=cfg.tokenizer_decoder_layers,
        tokenizer_heads=cfg.tokenizer_heads,
        tokenizer_dropout=cfg.tokenizer_dropout,
        tokenizer_token_dropout_mode=cfg.tokenizer_token_dropout_mode,
    )
    tokenizer = OATActionTokenizer(
        policy_config, action_dim=action_dim, residual=cfg.policy_type == "oat_rfsq_pair"
    ).to(device)
    optimizer = torch.optim.AdamW(
        tokenizer.parameters(),
        lr=cfg.learning_rate,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, (step + 1) / max(1, cfg.warmup_steps)),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    iterator = iter(loader)
    output_dir = Path(cfg.output_dir)
    tokenizer.train()
    for step in range(1, cfg.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        actions = _normalize_actions(batch[ACTION].to(device, non_blocking=True), action_stats)
        loss, _ = tokenizer(actions)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), cfg.grad_clip_norm)
        optimizer.step()
        scheduler.step()
        if step % cfg.log_freq == 0:
            logger.info("step=%d tokenizer_mse=%.6f", step, loss.item())
        if step % cfg.save_freq == 0:
            _save_tokenizer(tokenizer, output_dir / f"checkpoint-{step:08d}", cfg)
    _save_tokenizer(tokenizer, output_dir, cfg)
    logger.info("Saved %s action tokenizer to %s", cfg.policy_type, output_dir)


def main() -> None:
    init_logging()
    train_oat_tokenizer()


if __name__ == "__main__":
    main()
