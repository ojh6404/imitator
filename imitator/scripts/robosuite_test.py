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

        obs["object"] = obs["object-state"]


        obs = TensorUtils.to_batch(obs)
        if self.running_cnt % self.rnn_seq_length == 0:
            self.rnn_state = self.model.get_rnn_init_state(batch_size=1, device=self.device)

        with torch.no_grad():
            pred_action, self.rnn_state = self.model.forward_step(obs, rnn_state=self.rnn_state)

        # print("pred_action", pred_action)

        self.running_cnt += 1

        return TensorUtils.to_numpy(pred_action)[0]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--model", type=str, default="rnn")
    parser.add_argument("--model_path", type=str, default="models/rnn_model_best.pth")
    args = parser.parse_args()


    env_kargs = {
        "env_name": "Lift", # try with other tasks like "Stack" and "Door"
        # "has_renderer": False,
        "has_renderer": True,
        "has_offscreen_renderer": False,
        "ignore_done": True,
        "use_object_obs": True,
        "use_camera_obs": False,
        "control_freq": 20,
        "controller_configs": {
            "type": "OSC_POSE",
            "input_max": 1,
            "input_min": -1,
            "output_max": [
                0.05,
                0.05,
                0.05,
                0.5,
                0.5,
                0.5
            ],
            "output_min": [
                -0.05,
                -0.05,
                -0.05,
                -0.5,
                -0.5,
                -0.5
            ],
            "kp": 150,
            "damping": 1,
            "impedance_mode": "fixed",
            "kp_limits": [
                0,
                300
            ],
            "damping_limits": [
                0,
                10
            ],
            "position_limits": None,
            "orientation_limits": None,
            "uncouple_pos_ori": True,
            "control_delta": True,
            "interpolation": None,
            "ramp_ratio": 0.2
        },
        "robots": [
            "Panda"
        ],
        "camera_depths": False,
        "camera_heights": 84,
        "camera_widths": 84,
        "reward_shaping": False
    }


    # create environment instance
    env = suite.make(
        **env_kargs,
    )
    print("env action dim", env.action_dim)

    # reset the environment
    env.reset()

    policy_executor = PolicyExecutor(args)

    for i in range(1000):
        if i == 0:
            action = np.random.randn(env.robots[0].dof - 1) # sample random action
            print("action is ", action.shape)
        else:
            action = policy_executor.run(obs)
            # print("action policy", action.shape)
        obs, reward, done, info = env.step(action)  # take action in the environment


        # print("action", action)

        # print("obs keys", obs.keys())
        # print("obs['robot0_eef_pos']", obs['robot0_eef_pos'])
        # print("obs['robot0_eef_quat']", obs['robot0_eef_quat'])
        # print("obs['robot0_gripper_qpos']", obs['robot0_gripper_qpos'])
        # print("obs['object']", obs['object-state'])

        env.render()  # render on display
