#!/usr/bin/env python3

import numpy as np
import torch

import imitator.utils.tensor_utils as TensorUtils
from imitator.models.policy_nets import MLPActor, RNNActor
from imitator.utils.obs_utils import FloatVectorModality

from easydict import EasyDict as edict


from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

class PolicyRolloutBase(ABC):
    """
    Base class for policy rollout.
    """

    def __init__(self, cfg : dict) -> None:
        self.obs_keys = list(cfg.obs.keys())
        self.running_cnt = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_type = cfg.network.policy.type

    @abstractmethod
    def run(self, obs):
        pass

    @abstractmethod
    def load_model(self, model_cfg, model_path):
        pass



class PolicyRolloutRobosuite(PolicyRolloutBase):
    """
    Policy rollout for robosuite.
    """

    def __init__(self, cfg: dict) -> None:
        super(PolicyRolloutRobosuite, self).__init__(cfg)

    def run(self, obs):
        pass

    def load_model(self, model_cfg, model_path):
        pass


class PolicyRolloutROS(PolicyRolloutBase):
    """
    Policy rollout for ROS.
    """

    def __init__(self, cfg: dict) -> None:
        super(PolicyRolloutROS, self).__init__(cfg)


    def run(self, obs):
        pass

    def load_model(self, model_cfg, model_path):
        pass
