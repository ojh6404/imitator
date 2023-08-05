#!/usr/bin/env python3

import argparse

from imitator.utils.datasets import SequenceDataset


def main(args):

    datasets = SequenceDataset(
        hdf5_path=args.dataset,
        obs_keys=["robot0_eef_pos"],
        dataset_keys=["actions"],
        frame_stack=10,
        seq_length=1,
        pad_frame_stack=True,
        pad_seq_length=True,
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode="all",
        hdf5_use_swmr=True,
        filter_by_attribute=None,
        load_next_obs=True,
        )


    # testing the frame stack
    first_stack = datasets[0]["actions"] # 10, 7
    second_stack = datasets[1]["actions"] # 10, 7
    third_stack = datasets[2]["actions"] # 10, 7

    diff = first_stack[1:] - second_stack[:-1]
    print(diff)

    print("first stack")
    print(first_stack)
    print("second stack")
    print(second_stack)
    print("third stack")
    print(third_stack)



if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset", type=str, default=None)
    args = parser.parse_args()

    main(args)
