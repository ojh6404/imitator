#!/usr/bin/env python3

from moviepy.editor import ImageSequenceClip
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm
import os
import argparse
import numpy as np
from omegaconf import OmegaConf
import h5py

import tkinter as tk

import torch
import numpy as np
import cv2

from imitator.utils import file_utils as FileUtils

from tracking_ros.tracker.base_tracker import BaseTracker
from tracking_ros.utils.util import (
    download_checkpoint,
)
from tracking_ros.utils.painter import mask_painter, point_drawer, bbox_drawer
from tracking_ros.utils.dino_utils import get_grounded_bbox

from groundingdino.util.inference import load_model
from groundingdino.config import GroundingDINO_SwinT_OGC


def compose_mask(masks):
    """
    input: list of numpy ndarray of 0 and 1, [H, W]
    output: numpy ndarray of 0, 1, ..., len(inputs) [H, W], 0 is background
    """
    template_mask = np.zeros_like(masks[0]).astype(np.uint8)
    for i, mask in enumerate(masks):
        template_mask = np.clip(
            template_mask + mask * (i + 1),
            0,
            i + 1,
        )
        # TODO : checking overlapping mask
        assert len(np.unique(template_mask)) == (i + 2)

    return template_mask


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

    print("Processing dataset: {}".format(hdf5_path))
    dino_config =  "/home/oh/ros/pr2_ws/src/eus_imitation/imitator/imitator/scripts/GroundingDINO_SwinT_OGC.py"

    dino_checkpoint = download_checkpoint("dino","./")
    grounding_dino = load_model(dino_config, dino_checkpoint, device="cpu")

    sam_checkpoint = download_checkpoint("sam_vit_b", "./")
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint).to("cpu")
    predictor = SamPredictor(sam)

    xmem_checkpoint = download_checkpoint("xmem", "./")
    tracker_config =  "/home/oh/ros/pr2_ws/src/eus_imitation/imitator/imitator/scripts/tracker_config.yaml"
    xmem = BaseTracker(xmem_checkpoint, tracker_config, device="cuda:0")


    original_dataset = h5py.File(hdf5_path, "r")
    processed_dataset = h5py.File(hdf5_path.replace(".hdf5", "_with_mask.hdf5"), "w")

    # copy attributes
    for key in original_dataset.attrs.keys():
        processed_dataset.attrs[key] = original_dataset.attrs[key]
    for key in original_dataset["data"].attrs.keys():
        processed_dataset["data"].attrs[key] = original_dataset["data"].attrs[key]


    # demos
    for demo in tqdm(original_dataset["data"].keys()):
        demo_group = processed_dataset.create_group(demo)
        demo_group.attrs["num_samples"] = original_dataset["data"][demo].attrs["num_samples"]

        # copy actions
        demo_group.create_dataset(
            "actions",
            data=original_dataset["data"][demo]["actions"],
            dtype=original_dataset["data"][demo]["actions"].dtype,
        )

        # copy obs
        obs_group = demo_group.create_group("obs")
        for obs_key in obs_keys:
            obs_group.create_dataset(
                obs_key,
                data=original_dataset["data"][demo]["obs"][obs_key],
                dtype=original_dataset["data"][demo]["obs"][obs_key].dtype,
            )

            # create mask if obs's modality is ImageModality
            if config.obs[obs_key].modality == "ImageModality":
                original_images = original_dataset["data"][demo]["obs"][obs_key] # [T, H, W, C]


                if args.interactive:
                    cv2.imshow("original", original_images[0])
                    cv2.setMouseCallback("original", get_points)
                    cv2.waitKey(0)
                else:
                    bboxes = get_grounded_bbox(
                        model=grounding_dino,
                        image=original_images[0],
                        text_prompt=args.text_prompt,
                        box_threshold=args.box_threshold,
                        text_threshold=args.text_threshold,
                        )

                    predictor.set_image(original_images[0])
                    bboxes_tensor = torch.Tensor(bboxes).to("cpu")
                    transformed_bboxes = predictor.transform.apply_boxes_torch(
                        bboxes_tensor, original_images.shape[:2]
                    )

                    with torch.no_grad():
                        first_masks, scores, logits = predictor.predict_torch(
                            point_coords=None,
                            point_labels=None,
                            boxes=transformed_bboxes,
                            multimask_output=False,
                            )

                first_masks = first_masks.cpu().squeeze(1).numpy() # [N, H, W]
                template_mask = compose_mask(first_masks) # [H, W] with 0, 1, 2, 3, ... N, 0 is background
                masks = []

                for i, original_image in enumerate(original_images):
                    if i == 0:
                        template_mask, logit = xmem.track(frame=original_image, first_frame_annotation=template_mask)
                    else:
                        template_mask, logit = xmem.track(frame=original_image)
                    masks.append(template_mask)
                    # for debug
                    # cv2.imshow("mask", template_mask* 50)
                    # cv2.waitKey(0)
                xmem.clear_memory()

                masks = np.stack(masks, axis=0) # [T, H, W]
                # save mask
                obs_group.create_dataset(
                    obs_key + "_mask",
                    data=masks,
                    dtype=masks.dtype,
                )

    original_dataset.close()
    processed_dataset.close()








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-pn", "--project_name", type=str)
    parser.add_argument("-tp", "--text_prompt", type=str)
    parser.add_argument("-bt", "--box_threshold", type=float, default=0.35)
    parser.add_argument("-tt", "--text_threshold", type=float, default=0.25)
    parser.add_argument("-i", "--interactive", action="store_true")
    args = parser.parse_args()

    main(args)
