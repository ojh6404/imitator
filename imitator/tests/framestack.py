#!/usr/bin/env python3

import argparse

from imitator.utils.datasets import ImageDataset, SequenceDataset

from torch.utils.data import DataLoader

import cv2
import torch
import numpy as np


def main(args):

    datasets = ImageDataset(
        hdf5_path=args.dataset,
        obs_keys=["image"],
        hdf5_cache_mode=True,
        hdf5_use_swmr=True,
    )

    data = datasets[0]

    print(data["obs"]["image"].shape)

    data_loader = DataLoader(
        dataset=datasets,
        sampler=None,  # no custom sampling logic (uniform sampling)
        batch_size=5,  # batches of size 100
        shuffle=True,
        num_workers=0,
        drop_last=True,  # don't provide last batch in dataset pass if it's less than 100 in size
        # collate_fn= # TODO collate fn to numpy ndarray
    )

    for epoch in range(3):  # epoch numbers start at 1
        data_loader_iter = iter(data_loader)
        try:
            batch = next(data_loader_iter)
        except StopIteration:
            data_loader_iter = iter(data_loader)
            batch = next(data_loader_iter)

        print(batch["obs"]["image"].shape)

        for image in batch["obs"]["image"]:
            image = image.numpy()
            cv2.imshow("image", image)
            cv2.waitKey(0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default=None)
    args = parser.parse_args()

    main(args)
