#!/usr/bin/env python

import os
import numpy as np
from PIL import Image

from imitator.data.dataset import make_single_dataset
from imitator.data.utils.data_utils import NormalizationType
from imitator.utils.file_utils import get_config_from_project_name

DATASET_NAME = "imitator_dataset"
DATA_DIR = os.path.expanduser("~/tensorflow_datasets")

def main(args):
    config = get_config_from_project_name(args.project_name)

    primary_image_key = None
    wrist_image_key = None
    for obs_key in config.obs.keys():
        if config.obs[obs_key].get("camera") == "primary":
            primary_image_key = obs_key
            break
        if config.obs[obs_key].get("camera") == "wrist":
            wrist_image_key = obs_key
            break

    state_obs_keys = [obs_key for obs_key in config.obs.keys() if "image" not in obs_key]

    dataset = make_single_dataset(
        dataset_kwargs=dict(
            name=DATASET_NAME,
            data_dir=DATA_DIR,
            image_obs_keys={
                "primary": primary_image_key,
                "wrist": wrist_image_key,
            },
            state_obs_keys=state_obs_keys,
            language_key="language_instruction",
            action_state_normalization_type=NormalizationType.BOUNDS,
            action_normalization_mask=[True] * config.actions.dim,
        ),
        train=False,
    )

    train_data_iter = (
        dataset.repeat()
        .unbatch()
        .shuffle(1)
        .batch(1)
        .iterator()  # can reduce this if RAM consumption too high
    )

    # get a batch of data
    batch = next(train_data_iter)
    action_mean = dataset.dataset_statistics["action"]["mean"]
    action_std = dataset.dataset_statistics["action"]["std"]
    obs_mean = dataset.dataset_statistics["state"]["mean"]
    obs_std = dataset.dataset_statistics["state"]["std"]
    first_obs = batch["observation"]["state"][0] * obs_std + obs_mean
    first_action = batch["action"][0] * action_std + action_mean
    print("=============== Dataset Information ===============")
    print("Batch keys: ", batch.keys())
    print("Batch observation keys: ", batch["observation"].keys())
    print("State shapes: ", batch["observation"]["state"].shape)
    print("Action shapes: ", batch["action"].shape)
    print("First observation: ", first_obs)
    print("First action: ", first_action)
    if primary_image_key is not None:
        print("Primary image key: ", primary_image_key)
        print("Primary image shape: ", batch["observation"]["image_primary"].shape)
    if wrist_image_key is not None:
        print("Wrist image key: ", wrist_image_key)
        print("Wrist image shape: ", batch["observation"]["image_wrist"].shape)

    print("================ Dataset Statistics ================")
    print("Action mean: ", action_mean)
    print("Action std: ", action_std)
    print("State mean: ", obs_mean)
    print("State std: ", obs_std)
    print("====================================================")

    # visualize images
    if primary_image_key is not None:
        primary_images = batch["observation"]["image_primary"][0]  # remove batch dimension
        primary_images = Image.fromarray(np.concatenate(primary_images, axis=1))
        primary_images.show()
    if wrist_image_key is not None:
        wrist_images = batch["observation"]["image_wrist"][0]  # remove batch dimension
        wrist_images = Image.fromarray(np.concatenate(wrist_images, axis=1))
        wrist_images.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", "--project_name", type=str)
    args = parser.parse_args()
    main(args)
