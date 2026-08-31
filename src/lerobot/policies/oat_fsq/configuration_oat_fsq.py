#!/usr/bin/env python

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.oat.configuration_oat import OATConfig


@PreTrainedConfig.register_subclass("oat_fsq")
@dataclass
class OATFSQConfig(OATConfig):
    """Original one-token-per-latent OAT + FSQ route."""
