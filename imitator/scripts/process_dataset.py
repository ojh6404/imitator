#!/usr/bin/env python3

from tqdm import tqdm
import os
import argparse
import numpy as np
from omegaconf import OmegaConf

from imitator.utils import file_utils as FileUtils
from imitator.utils.datasets import SequenceDataset


# get min and max data from dataset
def main(args):
    config = FileUtils.get_config_from_project_name(args.project_name)
    hdf5_path = (
        args.dataset
        if args.dataset
        else os.path.join(
            FileUtils.get_project_folder(args.project_name), "data/dataset.hdf5"
        )
    )
    obs_keys = list(config.obs.keys())

    # extract obs keys that is not image modality
    obs_keys = [
        key
        for key in obs_keys
        if config.obs[key].modality != "ImageModality" and config.obs[key].normalize
    ]
    print("obs_keys: ", obs_keys)

    dataset_keys = ["actions"]
    dataset = SequenceDataset(
        hdf5_path=hdf5_path,
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
        hdf5_use_swmr=True,
    )

    obs_max_buf = dict()
    obs_min_buf = dict()
    for obs in obs_keys:
        obs_max_buf[obs] = np.ones_like(dataset[0]["obs"][obs]) * -np.inf
        obs_min_buf[obs] = np.ones_like(dataset[0]["obs"][obs]) * np.inf
    for i in tqdm(range(len(dataset))):
        # print(i)
        for obs in obs_keys:
            obs_max_buf[obs] = np.maximum(obs_max_buf[obs], dataset[i]["obs"][obs])
            obs_min_buf[obs] = np.minimum(obs_min_buf[obs], dataset[i]["obs"][obs])
    for obs in obs_keys:
        print(obs, "max and min:", obs_max_buf[obs], obs_min_buf[obs])
    # dump to yaml
    yaml_data = dict()
    yaml_data["actions"] = dict()
    yaml_data["obs"] = dict()

    action_max = np.ones_like(dataset[0]["actions"])
    action_min = np.ones_like(dataset[0]["actions"]) * -1

    yaml_data["actions"]["max"] = action_max.tolist()
    yaml_data["actions"]["min"] = action_min.tolist()

    for obs in obs_keys:
        yaml_data["obs"][obs] = dict()
        yaml_data["obs"][obs]["max"] = obs_max_buf[obs].tolist()
        yaml_data["obs"][obs]["min"] = obs_min_buf[obs].tolist()

    # yaml_file = open(
    #     os.path.join(FileUtils.get_config_folder(args.project_name), "normalize.yaml"), "w"
    # )

    normalize_cfg = OmegaConf.create(yaml_data)
    OmegaConf.save(
        normalize_cfg,
        os.path.join(FileUtils.get_config_folder(args.project_name), "normalize.yaml"),
    )

    # yaml.dump(yaml_data, yaml_file, default_flow_style=None)
    # yaml_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, help="path to hdf5 dataset")
    parser.add_argument(
        "-pn", "--project_name", type=str, required=True, help="project name"
    )
    args = parser.parse_args()

    main(args)
