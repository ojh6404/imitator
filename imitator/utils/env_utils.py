#!/usr/bin/env python3

import os
import numpy as np
import cv2
import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from omegaconf import OmegaConf
from tqdm import tqdm
from collections import OrderedDict
import h5py

from imitator.utils import tensor_utils as TensorUtils
from imitator.utils import file_utils as FileUtils
from imitator.models.policy_nets import MLPActor, RNNActor, TransformerActor
from imitator.utils.obs_utils import *

try:
    import robosuite
except:
    print("robosuite cannot be imported")



yaml.add_representer(
    OrderedDict,
    lambda dumper, data: dumper.represent_mapping(
        "tag:yaml.org,2002:map", data.items()
    ),
)


def get_env_meta_from_dataset(dataset_path):
    h5py_file = h5py.File(dataset_path, "r")
    env_meta = h5py_file["data"].attrs["env_args"]
    env_meta = yaml.safe_load(env_meta)
    return env_meta


def create_env_from_env_meta(env_meta, render=False):
    env_name = env_meta["env_name"]
    env_kwargs = env_meta["env_kwargs"]
    env_kwargs["has_renderer"] = render
    env = robosuite.make(
        env_name=env_name,
        **env_kwargs,
    )
    return env


# def generate_config_from_dataset(dataset_path, project_name):
#     config = edict(OrderedDict())

#     h5py_file = h5py.File(dataset_path, "r")

#     print(h5py_file.keys())

#     return config


class RolloutBase(ABC):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        # super(nn.Module, self).__init__()

        # assert cfg.network has checkpoint
        assert cfg.network.policy.checkpoint is not None

        self.cfg = cfg
        self.actor_type = eval(cfg.network.policy.model)
        self.obs_keys = list(cfg.obs.keys())
        self.image_obs = []
        for obs in self.obs_keys:
            if cfg.obs[obs].modality == "ImageModality":
                self.image_obs.append(obs)
        self.running_cnt = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = cfg.network.policy.checkpoint
        self.project_name = cfg.project_name

        # render image if needed
        self.render_image = True and self.image_obs != []

        if self.actor_type == RNNActor:
            self.rnn_seq_length = cfg.network.policy.rnn.seq_length

        normalize = True  # TODO
        if normalize:
            normalizer_cfg = FileUtils.get_normalize_cfg(self.project_name)
            action_mean, action_std = get_normalize_params(
                normalizer_cfg.actions.min, normalizer_cfg.actions.max
            )
            action_mean, action_std = (
                torch.Tensor(action_mean).to(self.device).float(),
                torch.Tensor(action_std).to(self.device).float(),
            )
            cfg.actions.update(
                {"max": normalizer_cfg.actions.max, "min": normalizer_cfg.actions.min}
            )
            for obs in normalizer_cfg["obs"]:
                cfg.obs[obs].update(
                    {
                        "max": normalizer_cfg.obs[obs].max,
                        "min": normalizer_cfg.obs[obs].min,
                    }
                )
        else:
            action_mean, action_std = 0.0, 1.0

        if self.actor_type == TransformerActor:
            self.stacked_obs = OrderedDict()
            self.context_length = cfg.network.policy.transformer.context_length

        self.load_model(cfg)

    def reset(self):
        self.running_cnt = 0

    def frame_stack(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        # obs is dict of numpy ndarray [D]
        # if running_cnt == 0, then initialize the frame stack with the first obs like [10, D]
        # else, pop the oldest obs and append the new obs
        # return the stacked obs
        stacked_obs = OrderedDict()
        for obs_key in obs.keys():
            if self.running_cnt == 0:
                stacked_obs[obs_key] = np.stack([obs[obs_key]] * self.context_length)
            else:
                stacked_obs[obs_key] = np.concatenate(
                    [
                        self.stacked_obs[obs_key][1:, :],
                        np.expand_dims(obs[obs_key], axis=0),
                    ],
                    axis=0,
                )
        self.stacked_obs = stacked_obs
        return stacked_obs



    def process_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        if self.actor_type == TransformerActor:
            obs = self.frame_stack(obs)
        return obs


    def rollout(self, obs: Dict[str, Any]) -> None:
        obs = self.process_obs(obs)
        obs = TensorUtils.to_batch(obs)  # [1, D], if TransformerActor, [1, T, D]

        if self.actor_type == RNNActor:
            if self.running_cnt % self.rnn_seq_length == 0:
                self.rnn_state = self.model.get_rnn_init_state(
                    batch_size=1, device=self.device
                )
            with torch.no_grad():
                pred_action, self.rnn_state = self.model.forward_step(
                    obs, rnn_state=self.rnn_state, unnormalize=True
                )
        else:
            with torch.no_grad():
                pred_action = self.model.forward_step(obs, unnormalize=True)

        if self.render_image:
            self.render(obs)

        self.running_cnt += 1
        return TensorUtils.to_numpy(pred_action)[0]

    def load_model(self, cfg: Dict[str, Any]) -> None:
        self.model = self.actor_type(cfg)
        self.model.load_state_dict(torch.load(cfg.network.policy.checkpoint))
        self.model.eval()
        self.model.to(self.device)

        self.image_encoder = OrderedDict()
        self.image_decoder = OrderedDict()

        for image_obs in self.image_obs:
            self.image_encoder[image_obs] = self.model.nets["obs_encoder"].nets[
                image_obs
            ]
            has_decoder = cfg.obs[image_obs].obs_encoder.has_decoder
            if has_decoder:
                self.image_decoder[image_obs] = (
                    self.model.nets["obs_encoder"].nets[image_obs].nets["decoder"]
                )

    @torch.no_grad()
    def render(self, obs: Dict[str, Any]) -> None:
        # input : obs dict of numpy ndarray [1, H, W, C]
        if self.actor_type == TransformerActor: # obs is stacked if transformer like [1, T, D]
            # so we need to use last time step obs to render
            obs = {k: v[:, -1, :] for k, v in obs.items()}

        if self.image_obs:
            obs = TensorUtils.squeeze(obs, dim=0)
            for image_obs in self.image_obs:
                image_render = obs[image_obs]

                # if has_decoder, concat recon and original image to visualize
                if image_obs in self.image_decoder:
                    image_latent = self.image_encoder[image_obs](
                        image_render[None, ...]
                    )  # [1, C, H, W]
                    image_recon = (
                        self.image_decoder[image_obs](image_latent) * 255.0
                    )  # [1, C, H, W] TODO set unnormalizer
                    image_recon = image_recon.cpu().numpy().astype(np.uint8)
                    image_recon = np.transpose(
                        image_recon, (0, 2, 3, 1)
                    )  # [1, H, W, C]
                    image_recon = np.squeeze(image_recon)
                    image_render = concatenate_image(image_render, image_recon)
                cv2.imshow(image_obs, cv2.cvtColor(image_render, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)


class RobosuiteRollout(RolloutBase):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super(RobosuiteRollout, self).__init__(cfg)
        import robosuite
        self.env_meta = get_env_meta_from_dataset(cfg.dataset_path)
        self.env = create_env_from_env_meta(self.env_meta, render=True)

        print("Robosuite Rollout initialized")

    # @torch.no_grad()
    def rollout(self, obs: Dict[str, Any]) -> None:
        return super(RobosuiteRollout, self).rollout(obs)

    def reset(self):
        super(RobosuiteRollout, self).reset()
        if self.actor_type == RNNActor:
            self.rnn_state = self.model.get_rnn_init_state(
                batch_size=1, device=self.device
            )

    def process_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        processed_obs = obs # dict of numpy ndarray [D]
        # rename object-state to object
        processed_obs["object"] = obs["object-state"]
        processed_obs.pop("object-state")

        # flip image
        for obs_key in self.obs_keys:
            if self.cfg.obs[obs_key].modality == "ImageModality":
                processed_obs[obs_key] = obs[obs_key][::-1].copy()  # flip image
        processed_obs = super(RobosuiteRollout, self).process_obs(processed_obs)
        return processed_obs

    # def step(self, action: np.ndarray) -> Dict[str, Any]:
    #     obs, reward, done, info = self.env.step(action)
    #     return self.process_obs(obs), reward, done, info
