#!/usr/bin/env python

from lerobot.policies.oat.modeling_oat import OATPolicyBase

from .configuration_oat_fsq import OATFSQConfig


class OATFSQPolicy(OATPolicyBase):
    config_class = OATFSQConfig
    name = "oat_fsq"
    residual = False
