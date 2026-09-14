#!/usr/bin/env python

import torch
from torch import Tensor, nn

from .configuration_oat import OATConfig


class OATAutoregressiveModel(nn.Module):
    def __init__(self, config: OATConfig, vocab_size: int, cond_dim: int, max_tokens: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size + 1, config.policy_dim)
        self.token_positions = nn.Parameter(torch.empty(1, max_tokens + 1, config.policy_dim))
        self.cond_projection = nn.Linear(cond_dim, config.policy_dim)
        self.cond_positions = nn.Parameter(torch.zeros(1, config.n_obs_steps, config.policy_dim))
        self.cond_encoder = nn.Sequential(
            nn.Linear(config.policy_dim, 4 * config.policy_dim),
            nn.Mish(),
            nn.Linear(4 * config.policy_dim, config.policy_dim),
        )
        layer = nn.TransformerDecoderLayer(
            d_model=config.policy_dim,
            nhead=config.policy_heads,
            dim_feedforward=4 * config.policy_dim,
            dropout=config.policy_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, config.policy_layers)
        self.output = nn.Linear(config.policy_dim, vocab_size + 1, bias=False)
        self.output.weight = self.token_embedding.weight
        nn.init.normal_(self.token_positions, std=0.02)

    def forward(self, tokens: Tensor, condition: Tensor) -> Tensor:
        if condition.shape[1] > self.cond_positions.shape[1]:
            raise ValueError(
                f"Expected at most {self.cond_positions.shape[1]} observations, got {condition.shape[1]}."
            )
        hidden = self.token_embedding(tokens) + self.token_positions[:, : tokens.shape[1]]
        memory = self.cond_projection(condition) + self.cond_positions[:, : condition.shape[1]]
        memory = self.cond_encoder(memory)
        mask = nn.Transformer.generate_square_subsequent_mask(tokens.shape[1], device=tokens.device)
        decoded = self.decoder(hidden, memory, tgt_mask=mask, tgt_is_causal=True)
        return self.output(decoded)

    def generate(self, bos: Tensor, condition: Tensor, count: int, temperature: float, top_k: int) -> Tensor:
        tokens = bos
        for _ in range(count):
            # BOS is an input-only symbol and is excluded from generated action tokens.
            logits = self(tokens, condition)[:, -1, : self.vocab_size]
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if 0 < top_k < logits.shape[-1]:
                    values, indices = logits.topk(top_k, dim=-1)
                    sampled = torch.multinomial(values.softmax(dim=-1), 1)
                    next_token = indices.gather(-1, sampled)
                else:
                    next_token = torch.multinomial(logits.softmax(dim=-1), 1)
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens[:, 1:]
