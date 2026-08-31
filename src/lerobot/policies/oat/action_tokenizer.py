#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .configuration_oat import OATConfig


def _round_ste(value: Tensor) -> Tensor:
    rounded = value.round()
    return value + (rounded - value).detach()


# --------------------------------------------------------------------
# Copyright (C) 2024-2025 EPFL & Apple Inc.
# Licensed under the EPFL-Apple Sample Code License (Non-Commercial).
# Adapted from the OAT FSQ/RFSQ implementation and
# lucidrains/vector-quantize-pytorch finite scalar quantization.
# --------------------------------------------------------------------
class FSQ(nn.Module):
    """Finite scalar quantizer used by the original OAT action tokenizer."""

    def __init__(self, levels: tuple[int, ...]) -> None:
        super().__init__()
        level_tensor = torch.tensor(levels, dtype=torch.int32)
        basis = torch.cumprod(torch.tensor((1, *levels[:-1])), dim=0, dtype=torch.int32)
        self.register_buffer("_levels", level_tensor, persistent=True)
        self.register_buffer("_basis", basis, persistent=True)
        self.dim = len(levels)
        self.codebook_size = math.prod(levels)

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        half_width = self._levels // 2
        return _round_ste(self.bound(z.float())) / half_width

    def codes_to_indices(self, codes: Tensor) -> Tensor:
        half_width = self._levels // 2
        shifted = codes * half_width + half_width
        return (shifted * self._basis).sum(dim=-1).long()

    def indices_to_embedding(self, indices: Tensor) -> Tensor:
        values = (indices.unsqueeze(-1) // self._basis) % self._levels
        half_width = self._levels // 2
        return (values - half_width) / half_width

    def forward(self, latents: Tensor) -> tuple[Tensor, Tensor]:
        if latents.shape[-1] != self.dim:
            raise ValueError(f"Expected FSQ dimension {self.dim}, got {latents.shape[-1]}.")
        quantized = self.quantize(latents)
        return quantized, self.codes_to_indices(quantized)


class RFSQ(nn.Module):
    """Two-stage residual FSQ; tokenizer serialization remains stage-major."""

    num_stages = 2

    def __init__(self, levels: tuple[int, ...], latent_horizon: int) -> None:
        super().__init__()
        self.stage1 = FSQ(levels)
        self.stage2 = FSQ(levels)
        self.dim = self.stage1.dim
        self.codebook_size = self.stage1.codebook_size
        self.latent_horizon = latent_horizon
        self.mu2 = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.log_sigma2 = nn.Parameter(torch.zeros(1, 1, self.dim))

    def sigma2(self) -> Tensor:
        return self.log_sigma2.exp().clamp(min=1e-4, max=100.0)

    def forward(self, latents: Tensor) -> tuple[Tensor, Tensor]:
        if latents.shape[-2:] != (self.latent_horizon, self.dim):
            raise ValueError(
                f"Expected RFSQ latents (..., {self.latent_horizon}, {self.dim}), got {latents.shape}."
            )
        q1, idx1 = self.stage1(latents)
        residual = (latents.float() - q1 - self.mu2) / self.sigma2()
        q2_norm, idx2 = self.stage2(residual)
        q2 = q2_norm * self.sigma2() + self.mu2
        return q1 + q2, torch.cat((idx1, idx2), dim=-1)

    def indices_to_embedding(self, indices: Tensor, keep_k: int | None = None) -> Tensor:
        horizon = self.latent_horizon
        if indices.shape[-1] != 2 * horizon:
            raise ValueError(f"Expected {2 * horizon} stage-major tokens, got {indices.shape[-1]}.")
        q1 = self.stage1.indices_to_embedding(indices[..., :horizon])
        q2 = self.stage2.indices_to_embedding(indices[..., horizon:]) * self.sigma2() + self.mu2
        result = q1 + q2
        if keep_k is not None and keep_k < horizon:
            result = result.clone()
            result[..., keep_k:, :] = 0
        return result


class OATActionTokenizer(nn.Module):
    """Register action autoencoder with either FSQ or two-stage RFSQ latents."""

    def __init__(self, config: OATConfig, action_dim: int, residual: bool) -> None:
        super().__init__()
        self.horizon = config.horizon
        self.latent_horizon = config.latent_horizon
        self.action_dim = action_dim
        self.latent_dim = len(config.fsq_levels)
        self.residual = residual
        dim = config.tokenizer_dim

        self.action_projection = nn.Linear(action_dim, dim)
        self.action_positions = nn.Parameter(torch.empty(1, config.horizon, dim))
        self.register_tokens = nn.Parameter(torch.empty(1, config.latent_horizon, dim))
        self.token_dropout_mode = config.tokenizer_token_dropout_mode
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=config.tokenizer_heads,
            dim_feedforward=4 * dim,
            dropout=config.tokenizer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, config.tokenizer_encoder_layers, enable_nested_tensor=False
        )
        self.to_latent = nn.Linear(dim, self.latent_dim)
        self.quantizer: FSQ | RFSQ = (
            RFSQ(config.fsq_levels, config.latent_horizon) if residual else FSQ(config.fsq_levels)
        )

        self.from_latent = nn.Linear(self.latent_dim, dim)
        self.latent_positions = nn.Parameter(torch.empty(1, config.latent_horizon, dim))
        self.latent_mask_token = nn.Parameter(torch.empty(dim))
        self.action_queries = nn.Parameter(torch.empty(1, config.horizon, dim))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=config.tokenizer_heads,
            dim_feedforward=4 * dim,
            dropout=config.tokenizer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.tokenizer_decoder_layers)
        self.action_head = nn.Linear(dim, action_dim)
        nn.init.normal_(self.action_positions, std=0.02)
        nn.init.normal_(self.register_tokens, std=0.02)
        nn.init.normal_(self.latent_positions, std=0.02)
        nn.init.normal_(self.latent_mask_token, std=0.02)
        nn.init.normal_(self.action_queries, std=0.02)

    @property
    def codebook_size(self) -> int:
        return self.quantizer.codebook_size

    @property
    def token_seq_len(self) -> int:
        return self.latent_horizon * (2 if self.residual else 1)

    def encode(self, actions: Tensor) -> Tensor:
        if actions.shape[-2:] != (self.horizon, self.action_dim):
            raise ValueError(
                f"Expected action chunks (..., {self.horizon}, {self.action_dim}), got {actions.shape}."
            )
        actions = self.action_projection(actions) + self.action_positions
        registers = self.register_tokens.expand(actions.shape[0], -1, -1)
        total_length = self.horizon + self.latent_horizon
        attention_mask = torch.zeros(total_length, total_length, dtype=torch.bool, device=actions.device)
        attention_mask[: self.horizon, self.horizon :] = True
        attention_mask[self.horizon :, self.horizon :] = torch.triu(
            torch.ones(
                self.latent_horizon,
                self.latent_horizon,
                dtype=torch.bool,
                device=actions.device,
            ),
            diagonal=1,
        )
        encoded = self.encoder(torch.cat((actions, registers), dim=1), mask=attention_mask)
        return self.to_latent(encoded[:, -self.latent_horizon :])

    def tokenize(self, actions: Tensor) -> Tensor:
        _, indices = self.quantizer(self.encode(actions))
        return indices

    def _sample_keep_k(self, batch_size: int, device: torch.device) -> Tensor:
        if self.token_dropout_mode == "uniform":
            return torch.randint(1, self.latent_horizon + 1, (batch_size,), device=device)
        if self.token_dropout_mode == "pow2":
            exponents = torch.arange(int(math.log2(self.latent_horizon)) + 1, device=device)
            values = 2**exponents
            indices = torch.randint(values.numel(), (batch_size,), device=device)
            return values[indices]
        if self.token_dropout_mode == "uniform_pow2":
            sampled = torch.randint(1, self.latent_horizon + 1, (batch_size,), device=device)
            return 2 ** torch.ceil(torch.log2(sampled.float())).long()
        power_by_mode = {
            "linear_biased": 1.0,
            "quadratic_biased": 2.0,
            "cubic_biased": 3.0,
        }
        power = power_by_mode[self.token_dropout_mode]
        weights = torch.arange(1, self.latent_horizon + 1, dtype=torch.float32, device=device).pow(power)
        return torch.multinomial(weights, batch_size, replacement=True) + 1

    def _apply_nested_dropout(self, memory: Tensor, keep_k: int | None) -> Tensor:
        if self.training and self.token_dropout_mode != "disable":
            keep_ks = self._sample_keep_k(memory.shape[0], memory.device)
        elif keep_k is not None:
            if not 1 <= keep_k <= self.latent_horizon:
                raise ValueError(f"keep_k must be in [1, {self.latent_horizon}], got {keep_k}.")
            keep_ks = torch.full((memory.shape[0],), keep_k, device=memory.device)
        else:
            return memory
        positions = torch.arange(self.latent_horizon, device=memory.device)
        mask = positions[None] >= keep_ks[:, None]
        mask_token = self.latent_mask_token.to(dtype=memory.dtype)
        return torch.where(mask[..., None], mask_token, memory)

    def decode(self, latents: Tensor, keep_k: int | None = None) -> Tensor:
        if latents.shape[-2:] != (self.latent_horizon, self.latent_dim):
            raise ValueError(
                f"Expected latents (..., {self.latent_horizon}, {self.latent_dim}), got {latents.shape}."
            )
        memory = self.from_latent(latents) + self.latent_positions
        memory = self._apply_nested_dropout(memory, keep_k)
        queries = self.action_queries.expand(latents.shape[0], -1, -1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(self.horizon, device=latents.device)
        decoded = self.decoder(queries, memory, tgt_mask=causal_mask, tgt_is_causal=True)
        return self.action_head(decoded)

    def detokenize(self, tokens: Tensor, keep_k: int | None = None) -> Tensor:
        if self.residual:
            latents = self.quantizer.indices_to_embedding(tokens, keep_k=keep_k)
        else:
            if tokens.shape[-1] > self.latent_horizon:
                raise ValueError(f"Expected at most {self.latent_horizon} FSQ tokens.")
            keep_k = tokens.shape[-1] if keep_k is None else keep_k
            latents = self.quantizer.indices_to_embedding(tokens)
            if keep_k < self.latent_horizon:
                padding = latents.new_zeros(latents.shape[0], self.latent_horizon - keep_k, self.latent_dim)
                latents = torch.cat((latents, padding), dim=1)
        return self.decode(latents, keep_k=keep_k)

    def forward(self, actions: Tensor) -> tuple[Tensor, dict[str, float]]:
        quantized, _ = self.quantizer(self.encode(actions))
        reconstructed = self.decode(quantized)
        loss = functional.mse_loss(reconstructed, actions)
        return loss, {"tokenizer_mse": float(loss.detach())}
