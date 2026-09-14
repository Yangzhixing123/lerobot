#!/usr/bin/env python

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torchvision.models import resnet18

from lerobot.utils.constants import OBS_STATE

from .configuration_oat import OATConfig


class SpatialSoftmaxEncoder(nn.Module):
    """ResNet-18 + spatial softmax, equivalent to OAT's 64-D RGB encoder output."""

    def __init__(self, input_channels: int, output_dim: int) -> None:
        super().__init__()
        if output_dim % 2:
            raise ValueError("vision_feature_dim must be even for spatial softmax.")

        def group_norm(channels: int) -> nn.GroupNorm:
            return nn.GroupNorm(max(1, channels // 16), channels)

        backbone = resnet18(weights=None, norm_layer=group_norm)
        if input_channels != 3:
            backbone.conv1 = nn.Conv2d(input_channels, 64, 7, 2, 3, bias=False)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.keypoint_projection = nn.Conv2d(512, output_dim // 2, kernel_size=1)

    def forward(self, image: Tensor) -> Tensor:
        feature_map = self.keypoint_projection(self.backbone(image))
        batch, channels, height, width = feature_map.shape
        attention = feature_map.reshape(batch, channels, -1).softmax(dim=-1)
        ys = torch.linspace(-1, 1, height, device=image.device, dtype=image.dtype)
        xs = torch.linspace(-1, 1, width, device=image.device, dtype=image.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        x_mean = (attention * grid_x.flatten()).sum(dim=-1)
        y_mean = (attention * grid_y.flatten()).sum(dim=-1)
        return torch.stack((x_mean, y_mean), dim=-1).flatten(start_dim=1)


class OATObservationEncoder(nn.Module):
    def __init__(self, config: OATConfig) -> None:
        super().__init__()
        self.config = config
        self.image_keys = tuple(sorted(config.image_features))
        state_keys: list[str] = []
        if config.robot_state_feature is not None:
            state_keys.append(OBS_STATE)
        if config.env_state_feature is not None:
            assert config.input_features is not None
            state_keys.append(
                next(
                    key
                    for key, feature in config.input_features.items()
                    if feature is config.env_state_feature
                )
            )
        self.state_keys = tuple(state_keys)

        modules: dict[str, nn.Module] = {}
        shared: nn.Module | None = None
        shared_channels: int | None = None
        for key in self.image_keys:
            channels = config.image_features[key].shape[0]
            if config.share_camera_encoder and shared is not None and channels != shared_channels:
                raise ValueError("All cameras must have the same channel count when sharing an encoder.")
            if shared is None or not config.share_camera_encoder:
                shared = SpatialSoftmaxEncoder(channels, config.vision_feature_dim)
                shared_channels = channels
            modules[key.replace(".", "__")] = shared
        self.camera_encoders = nn.ModuleDict(modules)

        assert config.input_features is not None
        state_dim = sum(config.input_features[key].shape[0] for key in self.state_keys)
        self.output_dim = len(self.image_keys) * config.vision_feature_dim + state_dim
        if config.num_tasks > 1:
            self.output_dim += 1

    def _task_feature(
        self, batch: dict[str, Any], batch_size: int, time: int, device: torch.device
    ) -> Tensor:
        task_index = batch.get("task_index")
        if task_index is None and self.config.task_names and "task" in batch:
            names = batch["task"]
            if isinstance(names, str):
                names = [names] * batch_size
            lookup = {name: index for index, name in enumerate(self.config.task_names)}
            try:
                task_index = torch.tensor([lookup[name] for name in names], device=device)
            except KeyError as exc:
                raise ValueError(f"Unknown OAT task {exc.args[0]!r}; configure policy.task_names.") from exc
        if task_index is None:
            task_index = torch.zeros(batch_size, device=device)
        if not isinstance(task_index, Tensor):
            task_index = torch.as_tensor(task_index, device=device)
        task_index = task_index.reshape(batch_size, -1)[:, -1].float()
        task_index = 2 * task_index / max(1, self.config.num_tasks - 1) - 1
        return task_index[:, None, None].expand(-1, time, 1)

    def forward(self, batch: dict[str, Any]) -> Tensor:
        features: list[Tensor] = []
        batch_size = time = None
        device = None
        for key in self.image_keys:
            image = batch[key]
            if image.ndim == 4:
                image = image[:, None]
            batch_size, time = image.shape[:2]
            device = image.device
            encoded = self.camera_encoders[key.replace(".", "__")](image.flatten(0, 1))
            features.append(encoded.reshape(batch_size, time, -1))
        for key in self.state_keys:
            state = batch[key]
            if state.ndim == 2:
                state = state[:, None]
            batch_size, time = state.shape[:2]
            device = state.device
            features.append(state)
        if batch_size is None or time is None or device is None:
            raise ValueError("OAT received no configured observation features.")
        if self.config.num_tasks > 1:
            features.append(self._task_feature(batch, batch_size, time, device))
        return torch.cat(features, dim=-1)
