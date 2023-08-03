#!/usr/bin/env python3

from tqdm import tqdm
import os
import argparse
import numpy as np
import torch
import h5py
import copy

import imitator.utils.file_utils as FileUtils
import imitator.utils.tensor_utils as TensorUtils
from imitator.models.base_nets import R3M, CLIP, MVP


def save_dict_to_hdf5(dic, filename):
    """
    ....
    """
    with h5py.File(filename, "w") as h5file:
        recursively_save_dict_contents_to_group(h5file, "/", dic)


def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    ....
    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + "/", item)
        else:
            raise ValueError("Cannot save %s type" % type(item))


def load_dict_from_hdf5(filename):
    """
    ....
    """
    with h5py.File(filename, "r") as h5file:
        return recursively_load_dict_contents_from_group(h5file, "/")


def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(
                h5file, path + key + "/"
            )
    return ans


# get min and max data from dataset
def main(args):
    config = FileUtils.get_config_from_project_name(args.project_name)
    obs_keys = list(config.obs.keys())
    print("obs_keys: ", obs_keys)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_obs_keys = [
        key for key in obs_keys if config.obs[key].modality == "ImageModality"
    ]

    # extract obs keys that is not image modality
    # obs_keys = [key for key in obs_keys if config.obs[key].modality != "ImageModality"]

    hdf5_path = (
        args.dataset
        if args.dataset
        else os.path.join(
            FileUtils.get_project_folder(args.project_name), "data/dataset.hdf5"
        )
    )

    h5py_file = h5py.File(hdf5_path, "r")
    converted_h5py_file = h5py.File(
        os.path.join(os.path.dirname(hdf5_path), "converted_dataset.hdf5"), "w"
    )

    # copy all attrs and group data from h5py_file to coverted_h5py_file including nested groups and attrs
    h5py_file.copy("data", converted_h5py_file)
    demos = FileUtils.sort_names_by_number(list(h5py_file["data"].keys()))

    # copy all attrs and group data from h5py_file to coverted_h5py_file

    latent_extractor = R3M().to(device)
    # freeze latent extractor
    for param in latent_extractor.parameters():
        param.requires_grad = False
    latent_extractor.eval()

    # convert image to latent and save to converted_h5py_file, delete image obs
    print("Converting image obs to latent obs...")
    for image_obs_key in image_obs_keys:
        for demo in tqdm(demos):
            with torch.no_grad():
                image_obs = np.array(
                    converted_h5py_file["data"][demo]["obs"][image_obs_key]
                ).transpose(0, 3, 1, 2)
                image_obs = TensorUtils.to_device(
                    TensorUtils.to_tensor(image_obs), device
                )
                image_latent = TensorUtils.to_numpy(latent_extractor(image_obs))
                converted_h5py_file["data"][demo]["obs"][
                    image_obs_key + "_latent"
                ] = image_latent
                del converted_h5py_file["data"][demo]["obs"][image_obs_key]
                # why file size is not reduced after deleting image obs?

    h5py_file.close()
    converted_h5py_file.close()

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-pn", "--project_name", type=str)
    args = parser.parse_args()

    main(args)
