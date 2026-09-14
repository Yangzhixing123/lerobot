#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.optim.schedulers import ConstantWithWarmupSchedulerConfig


@dataclass
class OATConfig(PreTrainedConfig):
    """Common configuration for the LeRobot-native OAT policies."""

    n_obs_steps: int = 2
    horizon: int = 32
    n_action_steps: int = 16

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MIN_MAX,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    action_tokenizer_path: str | None = None
    latent_horizon: int = 8
    fsq_levels: tuple[int, ...] = (8, 5, 5, 5)
    tokenizer_dim: int = 256
    tokenizer_encoder_layers: int = 2
    tokenizer_decoder_layers: int = 4
    tokenizer_heads: int = 4
    tokenizer_dropout: float = 0.1
    tokenizer_token_dropout_mode: str = "pow2"

    vision_feature_dim: int = 64
    share_camera_encoder: bool = False
    policy_dim: int = 256
    policy_layers: int = 4
    policy_heads: int = 4
    policy_dropout: float = 0.1

    num_tasks: int = 1
    task_names: tuple[str, ...] = ()

    temperature: float = 1.0
    top_k: int = 10
    use_k_tokens: int | None = None

    optimizer_lr: float = 5e-5
    optimizer_lr_observation_encoder: float = 1e-5
    optimizer_weight_decay: float = 0.0
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 100

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_obs_steps < 1:
            raise ValueError("n_obs_steps must be positive.")
        if not 1 <= self.n_action_steps <= self.horizon:
            raise ValueError("n_action_steps must be in [1, horizon].")
        if not 1 <= self.latent_horizon <= self.horizon:
            raise ValueError("latent_horizon must be in [1, horizon].")
        if not self.fsq_levels or any(level < 2 for level in self.fsq_levels):
            raise ValueError("fsq_levels must contain integers greater than one.")
        if self.tokenizer_heads < 1 or self.tokenizer_dim % self.tokenizer_heads:
            raise ValueError("tokenizer_heads must be positive and divide tokenizer_dim.")
        if not 0 <= self.tokenizer_dropout < 1:
            raise ValueError("tokenizer_dropout must be in [0, 1).")
        valid_token_dropout_modes = {
            "disable",
            "uniform",
            "pow2",
            "uniform_pow2",
            "linear_biased",
            "quadratic_biased",
            "cubic_biased",
        }
        if self.tokenizer_token_dropout_mode not in valid_token_dropout_modes:
            raise ValueError(
                "tokenizer_token_dropout_mode must be one of "
                f"{sorted(valid_token_dropout_modes)}, got {self.tokenizer_token_dropout_mode!r}."
            )
        if self.tokenizer_token_dropout_mode == "pow2" and self.latent_horizon & (self.latent_horizon - 1):
            raise ValueError("latent_horizon must be a power of two for pow2 token dropout.")
        if self.policy_heads < 1 or self.policy_dim % self.policy_heads:
            raise ValueError("policy_heads must be positive and divide policy_dim.")
        if not 0 <= self.policy_dropout < 1:
            raise ValueError("policy_dropout must be in [0, 1).")
        if self.use_k_tokens is not None and self.use_k_tokens < 1:
            raise ValueError("use_k_tokens must be positive when set.")
        if self.num_tasks < 1:
            raise ValueError("num_tasks must be positive.")
        if self.task_names and len(self.task_names) != self.num_tasks:
            raise ValueError("task_names must be empty or contain exactly num_tasks entries.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            weight_decay=0.0,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> ConstantWithWarmupSchedulerConfig:
        return ConstantWithWarmupSchedulerConfig(num_warmup_steps=self.scheduler_warmup_steps)

    def validate_features(self) -> None:
        if not self.image_features and self.robot_state_feature is None and self.env_state_feature is None:
            raise ValueError("OAT needs at least one visual, robot-state, or environment-state input.")
        if self.action_feature is None:
            raise ValueError("OAT requires an action output feature.")

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
