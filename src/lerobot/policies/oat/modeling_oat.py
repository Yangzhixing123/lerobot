#!/usr/bin/env python

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from safetensors.torch import load_file
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION

from .action_tokenizer import RFSQ, OATActionTokenizer
from .autoregressive import OATAutoregressiveModel
from .configuration_oat import OATConfig
from .observation_encoder import OATObservationEncoder


class OATPolicyBase(PreTrainedPolicy):
    """LeRobot adapter shared by the original FSQ and paired-RFSQ OAT routes."""

    config_class = OATConfig
    name = "oat_base"
    residual = False

    def __init__(
        self,
        config: OATConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        dataset_meta: Any | None = None,
    ) -> None:
        super().__init__(config)
        del dataset_stats, dataset_meta  # Normalization is handled by LeRobot's processors.
        config.validate_features()
        assert config.action_feature is not None
        self.action_dim = config.action_feature.shape[0]
        self.action_tokenizer = OATActionTokenizer(config, self.action_dim, self.residual)
        self._load_action_tokenizer(config.action_tokenizer_path)
        for parameter in self.action_tokenizer.parameters():
            parameter.requires_grad_(False)
        self.action_tokenizer.eval()

        self.observation_encoder = OATObservationEncoder(config)
        self.max_tokens = self.action_tokenizer.token_seq_len
        self.bos_id = self.action_tokenizer.codebook_size
        self.model = OATAutoregressiveModel(
            config,
            vocab_size=self.action_tokenizer.codebook_size,
            cond_dim=self.observation_encoder.output_dim,
            max_tokens=self.max_tokens,
        )
        observation_keys = (*self.observation_encoder.image_keys, *self.observation_encoder.state_keys)
        self.observation_queues = {key: deque(maxlen=config.n_obs_steps) for key in observation_keys}
        self.action_queue: deque[Tensor] = deque(maxlen=config.n_action_steps)

    def train(self, mode: bool = True) -> OATPolicyBase:
        super().train(mode)
        # The pretrained tokenizer is always frozen, including its dropout behavior.
        self.action_tokenizer.eval()
        return self

    def _load_action_tokenizer(self, tokenizer_path: str | None) -> None:
        if tokenizer_path is None:
            return
        path = Path(tokenizer_path).expanduser()
        if path.is_dir():
            path = path / "action_tokenizer.safetensors"
        if not path.is_file():
            raise FileNotFoundError(f"OAT action tokenizer weights not found: {path}")
        metadata_path = path.with_name("tokenizer_config.json")
        if metadata_path.is_file():
            with open(metadata_path) as file:
                metadata = json.load(file)
            expected = {
                "policy_type": "oat_rfsq_pair" if self.residual else "oat_fsq",
                "horizon": self.config.horizon,
                "latent_horizon": self.config.latent_horizon,
                "fsq_levels": list(self.config.fsq_levels),
                "tokenizer_dim": self.config.tokenizer_dim,
                "tokenizer_encoder_layers": self.config.tokenizer_encoder_layers,
                "tokenizer_decoder_layers": self.config.tokenizer_decoder_layers,
                "tokenizer_heads": self.config.tokenizer_heads,
                "tokenizer_token_dropout_mode": self.config.tokenizer_token_dropout_mode,
            }
            mismatches = {
                key: (metadata.get(key), value)
                for key, value in expected.items()
                if metadata.get(key) != value
            }
            if mismatches:
                raise ValueError(f"OAT tokenizer config does not match the policy: {mismatches}")
        self.action_tokenizer.load_state_dict(load_file(path), strict=True)

    def _save_pretrained(self, save_directory: Path, state_dict: dict[str, Tensor] | None = None) -> None:
        # A full LeRobot policy checkpoint already contains the frozen tokenizer.
        external_path = self.config.action_tokenizer_path
        self.config.action_tokenizer_path = None
        try:
            super()._save_pretrained(save_directory, state_dict=state_dict)
        finally:
            self.config.action_tokenizer_path = external_path

    def reset(self) -> None:
        for queue in self.observation_queues.values():
            queue.clear()
        self.action_queue.clear()

    def get_optim_params(self) -> list[dict[str, Any]]:
        def split(module: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
            decay, no_decay = [], []
            for parameter in module.parameters():
                if parameter.requires_grad:
                    (decay if parameter.ndim >= 2 else no_decay).append(parameter)
            return decay, no_decay

        policy_decay, policy_no_decay = split(self.model)
        encoder_decay, encoder_no_decay = split(self.observation_encoder)
        groups = [
            {
                "params": policy_decay,
                "lr": self.config.optimizer_lr,
                "weight_decay": self.config.optimizer_weight_decay,
            },
            {"params": policy_no_decay, "lr": self.config.optimizer_lr, "weight_decay": 0.0},
            {
                "params": encoder_decay,
                "lr": self.config.optimizer_lr_observation_encoder,
                "weight_decay": self.config.optimizer_weight_decay,
            },
            {
                "params": encoder_no_decay,
                "lr": self.config.optimizer_lr_observation_encoder,
                "weight_decay": 0.0,
            },
        ]
        return [group for group in groups if group["params"]]

    def _with_history(self, batch: dict[str, Any]) -> dict[str, Any]:
        result = dict(batch)
        assert self.config.input_features is not None
        for key, queue in self.observation_queues.items():
            value = batch[key]
            configured_rank = len(self.config.input_features[key].shape)
            if value.ndim == configured_rank + 2:
                result[key] = value[:, -self.config.n_obs_steps :]
                continue
            if value.ndim != configured_rank + 1:
                raise ValueError(f"Unexpected shape for {key}: {tuple(value.shape)}")
            if not queue:
                for _ in range(self.config.n_obs_steps):
                    queue.append(value)
            else:
                queue.append(value)
            result[key] = torch.stack(tuple(queue), dim=1)
        return result

    def _stage_major_to_interleaved(self, tokens: Tensor) -> Tensor:
        horizon = self.config.latent_horizon
        if tokens.ndim != 2:
            raise ValueError(f"Expected RFSQ token tensor [B, T], got {tuple(tokens.shape)}.")
        if tokens.shape[-1] != 2 * horizon:
            raise ValueError(f"Expected {2 * horizon} RFSQ tokens, got {tokens.shape[-1]}.")
        return torch.stack((tokens[:, :horizon], tokens[:, horizon:]), dim=-1).flatten(start_dim=1)

    def _interleaved_to_latents(self, tokens: Tensor) -> tuple[Tensor, int]:
        if tokens.shape[-1] % 2:
            raise ValueError("Paired RFSQ decoding requires an even number of tokens.")
        keep_k = tokens.shape[-1] // 2
        if not 1 <= keep_k <= self.config.latent_horizon:
            raise ValueError(f"Expected 1 <= K <= {self.config.latent_horizon}, got K={keep_k}.")
        quantizer = self.action_tokenizer.quantizer
        assert isinstance(quantizer, RFSQ)
        q1 = quantizer.stage1.indices_to_embedding(tokens[:, 0::2])
        q2 = quantizer.stage2.indices_to_embedding(tokens[:, 1::2]) * quantizer.sigma2() + quantizer.mu2
        latents = q1 + q2
        if keep_k < self.config.latent_horizon:
            padding = latents.new_zeros(
                latents.shape[0], self.config.latent_horizon - keep_k, latents.shape[-1]
            )
            latents = torch.cat((latents, padding), dim=1)
        return latents, keep_k

    def _policy_tokens(self, actions: Tensor) -> Tensor:
        tokens = self.action_tokenizer.tokenize(actions)
        return self._stage_major_to_interleaved(tokens) if self.residual else tokens

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        with torch.inference_mode():
            target_tokens = self._policy_tokens(batch[ACTION])
        condition = self.observation_encoder(batch)
        bos = torch.full(
            (target_tokens.shape[0], 1), self.bos_id, dtype=torch.long, device=target_tokens.device
        )
        sequence = torch.cat((bos, target_tokens), dim=1)
        logits = self.model(sequence[:, :-1], condition)
        loss = functional.cross_entropy(logits.flatten(0, 1), sequence[:, 1:].flatten())
        return loss, {"cross_entropy": float(loss.detach())}

    @torch.inference_mode()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        batch = self._with_history(batch)
        return self._predict_action_chunk_from_history(batch)

    def _predict_action_chunk_from_history(self, batch: dict[str, Tensor]) -> Tensor:
        condition = self.observation_encoder(batch)
        keep_k = self.config.latent_horizon
        if self.config.use_k_tokens is not None:
            keep_k = min(max(1, self.config.use_k_tokens), self.config.latent_horizon)
        token_count = 2 * keep_k if self.residual else keep_k
        bos = torch.full((condition.shape[0], 1), self.bos_id, dtype=torch.long, device=condition.device)
        tokens = self.model.generate(
            bos,
            condition,
            count=token_count,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
        )
        if self.residual:
            latents, keep_k = self._interleaved_to_latents(tokens)
            return self.action_tokenizer.decode(latents, keep_k=keep_k)
        return self.action_tokenizer.detokenize(tokens, keep_k=keep_k)

    @torch.inference_mode()
    def select_action(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        self.eval()
        # Keep consecutive robot observations even while actions from the previous chunk are executing.
        batch = self._with_history(batch)
        if not self.action_queue:
            chunk = self._predict_action_chunk_from_history(batch)[:, : self.config.n_action_steps]
            self.action_queue.extend(chunk.transpose(0, 1))
        return self.action_queue.popleft()
