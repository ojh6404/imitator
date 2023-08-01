import os
import cv2
import numpy as np
import robosuite as suite

import imitator.utils.tensor_utils as TensorUtils
from imitator.models.policy_nets import MLPActor, RNNActor
from imitator.utils.obs_utils import FloatVectorModality
import imitator.utils.file_utils as FileUtils
import imitator.utils.env_utils as EnvUtils
from imitator.utils.obs_utils import get_normalize_params

from easydict import EasyDict as edict
import yaml
import argparse
import torch

ACTOR_TYPES = {"mlp": MLPActor, "rnn": RNNActor}

class PolicyExecutor(object):
    def __init__(self, args):

        self.args = args
        config = FileUtils.get_config_from_project_name(args.project_name)

        self.obs_keys = list(config.obs.keys())
        self.running_cnt = 0


        self.actor_type = eval(config.network.policy.model)

        if self.actor_type == RNNActor:
            self.rnn_seq_length = config.network.policy.rnn.seq_length

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        normalize = True  # TODO
        if normalize:
            normalizer_cfg = FileUtils.get_normalize_cfg(args.project_name)
            action_mean, action_std = get_normalize_params(normalizer_cfg.actions.min, normalizer_cfg.actions.max)
            action_mean, action_std = torch.Tensor(action_mean).to(self.device).float(), torch.Tensor(action_std).to(self.device).float()
            config.actions.update({"max": normalizer_cfg.actions.max, "min": normalizer_cfg.actions.min})
            for obs in normalizer_cfg["obs"]:
                config.obs[obs].update({"max": normalizer_cfg.obs[obs].max, "min": normalizer_cfg.obs[obs].min})
        else:
            action_mean, action_std = 0.0, 1.0

        self.load_model(config)



    def load_model(self, cfg: edict) -> None:
        self.model = self.actor_type(cfg)
        self.model.load_state_dict(torch.load(self.args.checkpoint))
        self.model.eval()
        self.model.to(self.device)

    def run(self, obs):

        obs["object"] = obs["object-state"] # TODO
        obs["agentview_image"] = obs["agentview_image"][::-1].copy()


        cv2.imshow("agentview", cv2.cvtColor(obs["agentview_image"], cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        obs = TensorUtils.to_batch(obs)

        if self.actor_type == RNNActor:
            if self.running_cnt % self.rnn_seq_length == 0:
                self.rnn_state = self.model.get_rnn_init_state(batch_size=1, device=self.device)

        if self.actor_type == RNNActor:
            with torch.no_grad():
                pred_action, self.rnn_state = self.model.forward_step(obs, rnn_state=self.rnn_state, unnormalize=True)
        else:
            with torch.no_grad():
                pred_action = self.model.forward_step(obs, unnormalize=True)

        self.running_cnt += 1
        return TensorUtils.to_numpy(pred_action)[0]

    def reset_state(self):
        self.rnn_state = self.model.get_rnn_init_state(batch_size=1, device=self.device)
        self.running_cnt = 0



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn","--project_name", type=str)
    parser.add_argument("-d","--dataset", type=str)
    parser.add_argument("-ckpt","--checkpoint", type=str, default="mlp_model_best.pth")
    args = parser.parse_args()


    dataset_path = args.dataset if args.dataset else os.path.join(FileUtils.get_project_folder(args.project_name), "data/dataset.hdf5")
    env_meta = EnvUtils.get_env_meta_from_dataset(dataset_path)
    env_kwargs = env_meta["env_kwargs"]
    env_kwargs["has_renderer"] = True


    # create environment instance
    env = suite.make(
        env_name=env_meta["env_name"],
        **env_kwargs,
    )

    # reset the environment
    env.reset()

    policy_executor = PolicyExecutor(args)

    for j in range(10):
        for i in range(100):
            if i == 0:
                action = np.random.randn(policy_executor.model.action_dim)
            else:
                action = policy_executor.run(obs)
                # print("action policy", action.shape)
            obs, reward, done, info = env.step(action)  # take action in the environment



            if i == 99:
                env.reset()
                if policy_executor.actor_type == RNNActor:
                    policy_executor.reset_state()

            env.render()  # render on display
