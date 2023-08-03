#!/usr/bin/env python3

import h5py
import numpy as np
import argparse
import json

from imitator.utils.file_utils import sort_names_by_number


def main(args):
    f = h5py.File(args.dataset, "r")

    demos = sort_names_by_number(f["data"].keys())

    traj_lengths = []
    for ep in demos:
        traj_lengths.append(len(f["data"][ep]["actions"]))

    total_traj_length = np.sum(traj_lengths)

    print("=============================")
    print("Dataset info")
    print("Total demos: {}".format(len(demos)))
    print("Total trajectories: {}".format(total_traj_length))
    print("Trajectory length mean: {}".format(np.mean(traj_lengths)))
    print("Trajectory length std: {}".format(np.std(traj_lengths)))
    print("Trajectory length min: {}".format(np.min(traj_lengths)))
    print("Trajectory length max: {}".format(np.max(traj_lengths)))
    print("Max length traj index: {}".format(np.argmax(traj_lengths)))
    print("Observations: {}".format(list(f["data"][demos[0]]["obs"].keys())))
    print("Actions dim: {}".format(f["data"][demos[0]]["actions"].shape[1]))
    print("Env Meta: {}".format(f["data"].attrs["env_args"]))
    print("=============================")

    if args.verbose:
        print("=============================")
        print("Demo Lenghts: {}".format(traj_lengths))
        print("Obs info")
        for obs in f["data"][demos[0]]["obs"].keys():
            print("Obs: {}".format(obs))
            print("Shape: {}".format(f["data"][demos[0]]["obs"][obs].shape))
            print("Type: {}".format(f["data"][demos[0]]["obs"][obs].dtype))
            print("First data: {}".format(f["data"][demos[0]]["obs"][obs][0]))
            print("")
        print("First action data: {}".format(f["data"][demos[0]]["actions"][0]))
        print("=============================")

    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, help="path to hdf5 dataset")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    args = parser.parse_args()
    main(args)
