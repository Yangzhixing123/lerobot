from typing import Any

import torch

from lerobot.processor import PolicyAction, PolicyProcessorPipeline, make_default_pre_post_processors

from .configuration_oat_rfsq_pair import OATRFSQPairConfig


def make_oat_rfsq_pair_pre_post_processors(
    config: OATRFSQPairConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    return make_default_pre_post_processors(config, dataset_stats)
