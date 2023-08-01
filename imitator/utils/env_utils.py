#!/usr/bin/env python3

import os
import numpy as np

# import gym
import robosuite
import yaml
from easydict import EasyDict as edict
from omegaconf import OmegaConf
from tqdm import tqdm
from collections import OrderedDict
import h5py

from imitator.utils import file_utils as FileUtils
from imitator.utils.datasets import SequenceDataset

yaml.add_representer(
    OrderedDict,
    lambda dumper, data: dumper.represent_mapping(
        "tag:yaml.org,2002:map", data.items()
    ),
)

# get env meta from datasets (robosuite)


def get_env_meta_from_dataset(dataset_path):
    h5py_file = h5py.File(dataset_path, "r")
    env_meta = h5py_file["data"].attrs["env_args"]
    env_meta = yaml.safe_load(env_meta)
    return env_meta


# def generate_config_from_dataset()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None)
    args = parser.parse_args()
    dataset_path = args.dataset_path
    get_env_meta_from_dataset(dataset_path)
