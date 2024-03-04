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

from copy import deepcopy

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

global points
points = []


def get_points(event, x, y, flags, param):
    global points

    try:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
    except Exception as e:
        print(e)


def decompose_mask(template_mask, num_objects):
    """
    input: numpy ndarray of 0, 1, ..., N, size [H, W], 0 is background
    output: list of numpy ndarray of 0 and 1, [H, W]
    """
    masks = []
    for i in range(1, num_objects + 1):
        mask = np.zeros_like(template_mask).astype(np.uint8)
        mask[template_mask == i] = 1
        masks.append(mask)
    return masks


def compose_mask(masks):
    """
    input: list of numpy ndarray of 0 and 1, [H, W]
    output: numpy ndarray of 0, 1, ..., len(inputs) [H, W], 0 is background
    """
    template_mask = np.zeros_like(masks[0]).astype(np.uint8)
    for i, mask in enumerate(masks):
        template_mask = np.clip(
            0,
            template_mask + mask * (i + 1),
            i + 1,
        )
        # TODO : checking overlapping mask
        assert len(np.unique(template_mask)) == (i + 2)

    return template_mask


def seperate_each_object_from_image(image, mask, num_objects):
    """
    input: image [H, W, C], mask [H, W], num_objects
    output: list of images [H, W, C]
    """
    masks = decompose_mask(mask, num_objects)  # list of [H, W]
    images = []
    for mask in masks:
        masked_image = np.zeros_like(image).astype(np.uint8)
        masked_image[mask == 1] = image[mask == 1]
        images.append(masked_image)
    return images


def masking_image(image, mask, background=0):
    # image: [H, W, C]
    # mask: [H, W] with 0, 1, 2, 3, ... N, 0 is background
    # output: [H, W, C]
    masked_image = image.copy()
    masked_image[mask == 0] = background  # background to black

    return masked_image


def mask_to_bbox(mask, num_objects):
    # mask: [H, W] with 0, 1, 2, 3, ... N, 0 is background
    # output: [N, 4] of [x1, y1, x2, y2]

    bboxes = []

    for i in range(1, num_objects + 1):
        y, x = np.where(mask == i)
        # when there is no mask matching
        if len(y) == 0 or len(x) == 0:
            print("no mask matching")
            bboxes.append([0, 0, 0, 0])  # dummy bbox
        else:
            bboxes.append([x.min(), y.min(), x.max(), y.max()])
    return np.array(bboxes)


def resize_roi_from_bbox(image, bbox, shape=(64, 64)):

    # bbox: [x1, y1, x2, y2]
    # image: [H, W, C]
    # shape: [H, W]
    # output: [H, W, C]

    # extract image region of bbox and resize it to shape
    if bbox[0] == 0 and bbox[1] == 0 and bbox[2] == 0 and bbox[3] == 0:
        resized_roi_image = np.zeros((shape[0], shape[1], 3)).astype(np.uint8)
    else:
        roi_image = image[bbox[1] : bbox[3], bbox[0] : bbox[2], :]  # [H, W, C]
        # roi_image = image[bbox[1]-5:bbox[3]+5, bbox[0]-5:bbox[2]+5, :] # [H, W, C]
        resized_roi_image = cv2.resize(roi_image, shape)  # [C, H, W]
    return resized_roi_image


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

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("Processing dataset: {}".format(hdf5_path))
    dino_config = "/home/oh/ros/pr2_ws/src/eus_imitation/imitator/imitator/scripts/GroundingDINO_SwinT_OGC.py"

    dino_checkpoint = download_checkpoint("dino", "./")
    grounding_dino = load_model(dino_config, dino_checkpoint, device=device)

    sam_checkpoint = download_checkpoint("sam_vit_b", "./")
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint).to(device)
    predictor = SamPredictor(sam)

    xmem_checkpoint = download_checkpoint("xmem", "./")
    tracker_config = "/home/oh/ros/pr2_ws/src/eus_imitation/imitator/imitator/scripts/tracker_config.yaml"
    xmem = BaseTracker(xmem_checkpoint, tracker_config, device="cuda:0")

    original_dataset = h5py.File(hdf5_path, "r")
    processed_dataset = h5py.File(hdf5_path.replace(".hdf5", "_with_mask.hdf5"), "w")

    # copy attributes
    for key in original_dataset.attrs.keys():
        processed_dataset.attrs[key] = original_dataset.attrs[key]

    data_group = processed_dataset.create_group("data")

    # concatenate text into one string
    num_objects = len(args.text_prompt)
    text_prompt = ". ".join(args.text_prompt)
    print("text prompt : ", text_prompt)

    # demos
    for demo in tqdm(original_dataset["data"].keys()):
        demo_group = data_group.create_group(demo)
        demo_group.attrs["num_samples"] = original_dataset["data"][demo].attrs[
            "num_samples"
        ]

        # copy actions
        demo_group.create_dataset(
            "actions",
            data=original_dataset["data"][demo]["actions"],
            dtype=original_dataset["data"][demo]["actions"].dtype,
        )

        # copy obs
        obs_group = demo_group.create_group("obs")
        for obs_key in obs_keys:

            # create mask if obs's modality is ImageModality
            if config.obs[obs_key].modality == "ImageModality":
                original_images = original_dataset["data"][demo]["obs"][
                    obs_key
                ]  # [T, H, W, C]

                predictor.set_image(original_images[0])

                global points

                if args.interactive:
                    # global points
                    print("prompt manually")
                    prompt_masks = []  # list of mask [H, W]
                    prompt_image = original_images[0].copy()
                    while True:
                        cv2.imshow("original", prompt_image)
                        cv2.setMouseCallback("original", get_points)
                        prompt_image = point_drawer(
                            prompt_image, points, labels=[1] * len(points)
                        )
                        prompt_points = np.array(deepcopy(points))
                        if len(prompt_points) > 0:
                            masks, scores, logits = predictor.predict(
                                point_coords=prompt_points,
                                point_labels=[1] * len(prompt_points),
                                box=None,
                                mask_input=None,
                                multimask_output=False,
                            )
                            mask, logit = (
                                masks[np.argmax(scores)],
                                logits[np.argmax(scores)],
                            )  # choose the best mask [H, W]
                            prompt_image = mask_painter(
                                prompt_image, mask, color_index=len(prompt_masks) + 1
                            )

                            # refine mask
                            masks, scores, logits = predictor.predict(
                                point_coords=prompt_points,
                                point_labels=[1] * len(prompt_points),
                                box=None,
                                mask_input=logit[None, :, :],
                                multimask_output=False,
                            )
                            mask, logit = (
                                masks[np.argmax(scores)],
                                logits[np.argmax(scores)],
                            )

                        key = cv2.waitKey(1)
                        if key == ord("p"):
                            print("mask added")
                            try:
                                prompt_masks.append(mask.astype(np.uint8))
                            except:
                                print("no mask")
                            points = []
                        if key == ord("q"):
                            print("prompt end")
                            first_masks = np.array(prompt_masks)
                            cv2.destroyAllWindows()
                            break
                        if key == ord("r"):
                            print("reset points and masks")
                            points = []
                            prompt_masks = []
                            prompt_image = original_images[0].copy()

                else:
                    bboxes, phrases = get_grounded_bbox(
                        model=grounding_dino,
                        image=original_images[0],
                        text_prompt=text_prompt,
                        box_threshold=args.box_threshold,
                        text_threshold=args.text_threshold,
                    )

                    # ordering bbox with phrases be text order
                    # print("phrases", phrases)

                    ordered_bboxes = []
                    for txt in args.text_prompt:
                        for i, phrase in enumerate(phrases):
                            if txt in phrase:
                                ordered_bboxes.append(bboxes[i])
                                break

                    if len(phrases) == len(args.text_prompt):
                        bboxes_tensor = torch.Tensor(ordered_bboxes).to(
                            device
                        )  # [N, 4]
                        transformed_bboxes = predictor.transform.apply_boxes_torch(
                            bboxes_tensor, original_images.shape[:2]
                        )  # [N, 4]

                        with torch.no_grad():
                            first_masks, scores, logits = predictor.predict_torch(
                                point_coords=None,
                                point_labels=None,
                                boxes=transformed_bboxes,
                                multimask_output=False,
                            )
                        first_masks = first_masks.cpu().squeeze(1).numpy()  # [N, H, W]
                    else:
                        # global points
                        print("prompt manually")
                        prompt_masks = []  # list of mask [H, W]
                        prompt_image = original_images[0].copy()
                        while True:
                            cv2.imshow("original", prompt_image)
                            cv2.setMouseCallback("original", get_points)
                            prompt_image = point_drawer(
                                prompt_image, points, labels=[1] * len(points)
                            )
                            prompt_points = np.array(deepcopy(points))
                            if len(prompt_points) > 0:
                                masks, scores, logits = predictor.predict(
                                    point_coords=prompt_points,
                                    point_labels=[1] * len(prompt_points),
                                    box=None,
                                    mask_input=None,
                                    multimask_output=False,
                                )
                                mask, logit = (
                                    masks[np.argmax(scores)],
                                    logits[np.argmax(scores)],
                                )  # choose the best mask [H, W]
                                prompt_image = mask_painter(
                                    prompt_image,
                                    mask,
                                    color_index=len(prompt_masks) + 1,
                                )

                                # refine mask
                                masks, scores, logits = predictor.predict(
                                    point_coords=prompt_points,
                                    point_labels=[1] * len(prompt_points),
                                    box=None,
                                    mask_input=logit[None, :, :],
                                    multimask_output=False,
                                )
                                mask, logit = (
                                    masks[np.argmax(scores)],
                                    logits[np.argmax(scores)],
                                )

                            key = cv2.waitKey(1)
                            if key == ord("p"):
                                print("mask added")
                                try:
                                    prompt_masks.append(mask.astype(np.uint8))
                                except:
                                    print("no mask")
                                points = []
                            if key == ord("q"):
                                print("prompt end")
                                first_masks = np.array(prompt_masks)
                                # template_mask = compose_mask(prompt_masks) # [H, W] with 0, 1, 2, 3, ... N, where 0 is background
                                points = []
                                cv2.destroyAllWindows()
                                break

                template_mask = compose_mask(
                    first_masks
                )  # [H, W] with 0, 1, 2, 3, ... N, 0 is background
                masks = []  # [T, H, W]
                object_slot_images = []  # [T, N, H, W, C]
                cropped_rois = []  # [T, N, H, W, C]

                for i, original_image in enumerate(original_images):
                    if i == 0:
                        template_mask, logit = xmem.track(
                            frame=original_image, first_frame_annotation=template_mask
                        )
                    else:
                        template_mask, logit = xmem.track(frame=original_image)
                    masks.append(template_mask)

                    # masked_image = masking_image(original_image, template_mask)
                    bboxes = mask_to_bbox(
                        template_mask, num_objects=len(args.text_prompt)
                    )  # [N, 4]
                    assert len(bboxes) == len(args.text_prompt)

                    object_mask_images = seperate_each_object_from_image(
                        original_image, template_mask, num_objects=len(args.text_prompt)
                    )  # [N, H, W, C]
                    object_slot_images.append(object_mask_images)  # [T, N, H, W, C]

                    cropped_roi = []  # [N, H, W, C]
                    for i, bbox in enumerate(bboxes):
                        # object_mask_image = masking_image(original_image, bbox)
                        resized_roi_image = resize_roi_from_bbox(
                            object_mask_images[i], bbox, (224, 224)
                        )
                        cropped_roi.append(resized_roi_image)

                    cropped_roi = np.stack(cropped_roi, axis=0)  # [N, H, W, C]
                    cropped_rois.append(cropped_roi)  # [T, N, H, W, C]

                    # for debug
                    if args.debug:
                        # visualize cropped roi image like [H, W*N, C]
                        # debug_image = np.concatenate(cropped_roi, axis=1)
                        debug_image = np.concatenate(object_mask_images, axis=1)
                        cv2.imshow(
                            "debug", cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR)
                        )
                        cv2.waitKey(1)
                xmem.clear_memory()

                masks = np.stack(masks, axis=0)  # [T, H, W]
                object_slot_images = np.stack(
                    object_slot_images, axis=0
                )  # [T, N, H, W, C]
                cropped_rois = np.stack(cropped_rois, axis=0)  # [T, N, H, W, C]
                # save mask
                obs_group.create_dataset(
                    obs_key + "_mask",
                    data=masks,
                    dtype=masks.dtype,
                )
                obs_group.create_dataset(
                    obs_key + "_roi",
                    data=cropped_rois,
                    dtype=cropped_rois.dtype,
                )
                obs_group.create_dataset(
                    obs_key + "_object_slot",
                    data=object_slot_images,
                    dtype=object_slot_images.dtype,
                )
                # print("cropped roi shape", cropped_rois.shape)
                # print("mask shape", masks.shape)

            obs_group.create_dataset(
                obs_key,
                data=original_dataset["data"][demo]["obs"][obs_key],
                dtype=original_dataset["data"][demo]["obs"][obs_key].dtype,
            )

    original_dataset.close()
    processed_dataset.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("-pn", "--project_name", type=str)
    parser.add_argument("-tp", "--text_prompt", nargs="+", type=str, default=[])
    parser.add_argument("-bt", "--box_threshold", type=float, default=0.35)
    parser.add_argument("-tt", "--text_threshold", type=float, default=0.25)
    parser.add_argument("-i", "--interactive", action="store_true")
    args = parser.parse_args()

    main(args)
