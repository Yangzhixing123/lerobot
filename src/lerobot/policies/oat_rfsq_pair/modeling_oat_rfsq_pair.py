#!/usr/bin/env python

from lerobot.policies.oat.modeling_oat import OATPolicyBase

from .configuration_oat_rfsq_pair import OATRFSQPairConfig


class OATRFSQPairPolicy(OATPolicyBase):
    config_class = OATRFSQPairConfig
    name = "oat_rfsq_pair"
    residual = True
