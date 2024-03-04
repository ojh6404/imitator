#!/usr/bin/env python3

import h5py
import argparse


def main(args):

    f = h5py.File(args.dataset, "r")

    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]
    b_group_key = list(f.keys())[1]

    # Get the data
    data = list(f[a_group_key])
    b_data = list(f[b_group_key])

    print("data: ", data)
    print("b_data: ", b_data)

    train_mask = f["mask"]["train"]

    cnt = 0
    for i in train_mask:
        print(i.decode("utf-8"))
        print(cnt)
        cnt += 1

    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, help="path to hdf5 dataset")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    args = parser.parse_args()
    main(args)
