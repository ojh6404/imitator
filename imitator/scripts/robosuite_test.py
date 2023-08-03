import os
import cv2
import numpy as np
import robosuite as suite

import imitator.utils.tensor_utils as TensorUtils
from imitator.models.policy_nets import MLPActor, RNNActor
from imitator.utils.obs_utils import FloatVectorModality
import imitator.utils.file_utils as FileUtils
import imitator.utils.env_utils as EnvUtils
from imitator.utils.env_utils import RobosuiteRollout
from imitator.utils.obs_utils import get_normalize_params

from easydict import EasyDict as edict
import yaml
import argparse
import torch

ACTOR_TYPES = {"mlp": MLPActor, "rnn": RNNActor}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", "--project_name", type=str)
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-ckpt", "--checkpoint", type=str)
    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = FileUtils.get_best_runs(args.project_name, "rnn")

    dataset_path = (
        args.dataset
        if args.dataset
        else os.path.join(
            FileUtils.get_project_folder(args.project_name), "data/dataset.hdf5"
        )
    )
    config = FileUtils.get_config_from_project_name(args.project_name)
    config.network.policy.checkpoint = args.checkpoint
    config.project_name = args.project_name
    config.dataset_path = dataset_path

    image_obs_keys = [
        obs_key
        for obs_key in config.obs.keys()
        if config.obs[obs_key].modality == "ImageModality"
    ]
    for image_obs in image_obs_keys:
        if config.obs[image_obs].obs_encoder.model_path is None:
            obs_default_model_path = os.path.join(
                FileUtils.get_models_folder(args.project_name),
                f"{image_obs}_model.pth",
            )
            if not os.path.exists(obs_default_model_path):
                raise ValueError(
                    f"Model for {image_obs} does not exist. Please specify a model path in config file."
                )
            else:
                config.obs[image_obs].obs_encoder.model_path = obs_default_model_path

    env_meta = EnvUtils.get_env_meta_from_dataset(dataset_path)
    env = EnvUtils.create_env_from_env_meta(env_meta, render=True)

    # reset the environment
    env.reset()

    policy_executor = RobosuiteRollout(config)

    for j in range(10):
        for i in range(100):
            if i == 0:
                action = np.random.randn(policy_executor.model.action_dim)
            else:
                action = policy_executor.rollout(obs)
                # print("action policy", action.shape)
            obs, reward, done, info = env.step(action)  # take action in the environment

            if i == 99:
                env.reset()
                if policy_executor.actor_type == RNNActor:
                    policy_executor.reset()

            env.render()  # render on display

    env.close()  # close the environment
