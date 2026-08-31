#!/usr/bin/env python

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.oat.configuration_oat import OATConfig


@PreTrainedConfig.register_subclass("oat_rfsq_pair")
@dataclass
class OATRFSQPairConfig(OATConfig):
    """OAT + two-stage residual FSQ with paired/interleaved policy tokens."""
