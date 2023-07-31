#!/usr/bin/env python3

import numpy as np
import robosuite as suite

import imitator.utils.tensor_utils as TensorUtils
from imitator.models.policy_nets import RNNActor
from imitator.utils.obs_utils import FloatVectorModality

from easydict import EasyDict as edict
import yaml
import argparse
import torch

class PolicyExecutor(object):
    def __init__(self, args):

        self.args = args

        with open(args.config, "r") as f:
            config = edict(yaml.safe_load(f))

        self.obs_keys = list(config.obs.keys())
        self.running_cnt = 0
        self.rnn_seq_length = config.network.policy.rnn.seq_length

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.load_model(config)

    def load_model(self, cfg):
        self.model = RNNActor(cfg)
        self.model.load_state_dict(torch.load(self.args.model_path))
        self.model.eval()
        self.model.to(self.device)

    def run(self, obs):

        if self.running_cnt % self.rnn_seq_length == 0:



            pass

        obs = TensorUtils.to_tensor(obs)
        print(obs)

        self.running_cnt += 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--model", type=str, default="rnn")
    parser.add_argument("--model_path", type=str, default="models/rnn_model_best.pth")
    args = parser.parse_args()

    # create environment instance
    env = suite.make(
        env_name="Lift", # try with other tasks like "Stack" and "Door"
        robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )

    # reset the environment
    env.reset()

    policy_executor = PolicyExecutor(args)

    for i in range(1000):
        action = np.random.randn(env.robots[0].dof) # sample random action
        obs, reward, done, info = env.step(action)  # take action in the environment

        policy_executor.run(obs)

        # print("obs keys", obs.keys())
        # print("obs['robot0_eef_pos']", obs['robot0_eef_pos'])
        # print("obs['robot0_eef_quat']", obs['robot0_eef_quat'])
        # print("obs['robot0_gripper_qpos']", obs['robot0_gripper_qpos'])
        # print("obs['object']", obs['object-state'])

        env.render()  # render on display
