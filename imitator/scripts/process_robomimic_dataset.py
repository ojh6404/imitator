#!/usr/bin/env python

from tqdm import tqdm
import os
import numpy as np
import h5py
from absl import app, flags
from pyquaternion import Quaternion

from imitator.utils import file_utils as FileUtils

FLAGS = flags.FLAGS
# flags.DEFINE_string("project_name", None, "Name of the project to load config from.")
# flags.DEFINE_bool("vervose", False, "Verbose mode.")
flags.DEFINE_string("dataset", None, "Path to dataset, in HDF5 format.")
flags.DEFINE_string("proprio_type", "EEF", "Proprio type, EEF or JOINT")


# add proprio to obs
def main(_):
    print(f"Processing dataset {FLAGS.dataset}...")
    h5py_file = h5py.File(FLAGS.dataset, "a")
    data = h5py_file["data"]

    for demo_name, demo in tqdm(data.items()):
        demo = data[demo_name]
        obs = demo["obs"]

        print("obs types", type(obs))  # Group

        proprios = []

        for i in range(len(obs)):
            print("i", i)
            if FLAGS.proprio_type == "EEF":
                # quaternion to yaw, pitch, roll
                robot_eef_pos = obs["robot0_eef_pos"][i].copy()
                robot_eef_quat = obs["robot0_eef_quat"][i].copy()
                robot_eef_quat = Quaternion(
                    w=robot_eef_quat[0],
                    x=robot_eef_quat[1],
                    y=robot_eef_quat[2],
                    z=robot_eef_quat[3],
                )
                robot_eef_rpy = np.array(robot_eef_quat.yaw_pitch_roll).astype(
                    np.float32
                )
                robot_gripper_qpos = obs["robot0_gripper_qpos"][i].copy()[0][None]
                proprio = np.concatenate(
                    [robot_eef_pos, robot_eef_rpy, robot_gripper_qpos], axis=0
                )
                proprios.append(proprio)
            elif FLAGS.proprio_type == "JOINT":
                robot_joint_pos_cos = obs["robot0_joint_pos_cos"][i].copy()
                robot_joint_pos_sin = obs["robot0_joint_pos_sin"][i].copy()
                # get joint angles from cos and sin
                robot_joint_qpos = np.arctan2(robot_joint_pos_sin, robot_joint_pos_cos)
                robot_gripper_qpos = obs["robot0_gripper_qpos"][i].copy()[0][None]
                proprio = np.concatenate([robot_joint_qpos, robot_gripper_qpos], axis=0)
                proprios.append(proprio)
            else:
                raise ValueError(f"Proprio mode {FLAGS.proprio_type} not supported")

        obs.create_dataset("proprio", data=np.array(proprios, dtype=np.float32))


if __name__ == "__main__":
    app.run(main)
