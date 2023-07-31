#!/usr/bin/env python3

from tqdm import tqdm
import os
import argparse
import numpy as np
import yaml
from easydict import EasyDict as edict
from collections import OrderedDict

from imitator.utils.datasets import SequenceDataset

yaml.add_representer(
    OrderedDict,
    lambda dumper, data: dumper.represent_mapping("tag:yaml.org,2002:map", data.items()),
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--dataset", type=str, default="data/dataset.hdf5")
    args = parser.parse_args()

    cfg = edict(yaml.safe_load(open(args.config, "r")))

    obs_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
    dataset_keys = ["actions"]


    dataset = SequenceDataset(
        hdf5_path=args.dataset,
        obs_keys=obs_keys,  # observations we want to appear in batches
        dataset_keys=dataset_keys,  # can optionally specify more keys here if they should appear in batches
        load_next_obs=True,
        frame_stack=1,
        seq_length=1,  # length-10 temporal sequences
        pad_frame_stack=True,
        pad_seq_length=True,  # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode="all",  # cache dataset in memory to avoid repeated file i/o
        # hdf5_cache_mode=None,  # cache dataset in memory to avoid repeated file i/o
        hdf5_use_swmr=True,
    )


    # test
    test_obs = dataset[0]["obs"]  # [T, ...]




    for obs in obs_keys:
        print(obs, test_obs[obs].shape)


    # traj_len

    traj_len = len(dataset)

    print(traj_len)
    # print normalize input of dataset
    # dataset[i]["obs"] is a dict of obs_keys like numpy ndarray

    obs_max_buf = {}
    obs_min_buf = {}
    for obs in obs_keys:
        obs_max_buf[obs] = np.ones_like(dataset[0]["obs"][obs]) * -np.inf
        obs_min_buf[obs] = np.ones_like(dataset[0]["obs"][obs]) * np.inf

    # for i in range(traj_len):
    for i in tqdm(range(traj_len)):
        # print(i)
        for obs in obs_keys:
            obs_max_buf[obs] = np.maximum(obs_max_buf[obs], dataset[i]["obs"][obs])
            obs_min_buf[obs] = np.minimum(obs_min_buf[obs], dataset[i]["obs"][obs])

    for obs in obs_keys:
        print(obs, obs_max_buf[obs], obs_min_buf[obs])
        print(obs, obs_max_buf[obs] - obs_min_buf[obs])


    # dump to yaml

    yaml_data = OrderedDict()
    yaml_data["action"] = OrderedDict()
    yaml_data["obs"] = OrderedDict()

    action_max = np.ones_like(dataset[0]["actions"])
    action_min = np.ones_like(dataset[0]["actions"]) * -1

    yaml_data["action"]["max"] = action_max.tolist()
    yaml_data["action"]["min"] = action_min.tolist()

    for obs in obs_keys:
        yaml_data["obs"][obs] = OrderedDict()
        yaml_data["obs"][obs]["max"] = obs_max_buf[obs].tolist()
        yaml_data["obs"][obs]["min"] = obs_min_buf[obs].tolist()


    yaml_file = open(os.path.join(os.path.dirname(args.config), "normalize.yaml"), "w")
    yaml.dump(yaml_data, yaml_file, default_flow_style=None)
    yaml_file.close()
