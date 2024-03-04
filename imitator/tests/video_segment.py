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
from tracking_ros.utils.painter import (
    mask_painter,
    point_drawer,
    bbox_drawer,
    bbox_drawer_with_text,
)
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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # dino_config =  "/home/oh/ros/pr2_ws/src/eus_imitation/imitator/imitator/scripts/GroundingDINO_SwinT_OGC.py"

    # dino_checkpoint = download_checkpoint("dino","./")
    # grounding_dino = load_model(dino_config, dino_checkpoint, device=device)

    sam_checkpoint = download_checkpoint("sam_vit_b", "./")
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint).to(device)
    predictor = SamPredictor(sam)

    xmem_checkpoint = download_checkpoint("xmem", "./")
    tracker_config = "/home/oh/ros/pr2_ws/src/eus_imitation/imitator/imitator/scripts/tracker_config.yaml"
    xmem = BaseTracker(xmem_checkpoint, tracker_config, device="cuda:0")

    num_objects = args.num_objects

    video_path = args.dataset
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    original_images = np.array(frames).astype(np.uint8)

    print("num frames : ", len(frames))

    text_prompt = args.text_prompt

    # join with .
    # text_prompt = " . ".join(text_prompt)
    print("text prompt : ", text_prompt)

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
            prompt_image = point_drawer(prompt_image, points, labels=[1] * len(points))
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
                mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores)]

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

    template_mask = compose_mask(
        first_masks
    )  # [H, W] with 0, 1, 2, 3, ... N, 0 is background
    masks = []  # [T, H, W]
    object_slot_images = []  # [T, N, H, W, C]
    painted_images = []
    decomposed_masks = []

    masked_painted_images = []
    masked_painted_images_with_bbox = []

    for i, original_image in enumerate(original_images):
        if i == 0:
            template_mask, logit = xmem.track(
                frame=original_image, first_frame_annotation=template_mask
            )
        else:
            template_mask, logit = xmem.track(frame=original_image)
        masks.append(template_mask)

        decomposed_mask = decompose_mask(
            template_mask, num_objects=num_objects
        )  # [N, H, W]

        painted_image = original_image.copy()
        for i, mask in enumerate(decomposed_mask):
            painted_image = mask_painter(
                painted_image, mask.astype(np.bool_), color_index=i * 3 + 1, alpha=0.7
            )

        # masking out painted image with mask
        masked_painted_image = painted_image.copy()
        masked_painted_image[template_mask == 0] = 0
        masked_painted_images.append(masked_painted_image)

        bboxes = mask_to_bbox(template_mask, num_objects=num_objects)  # [N, 4]

        # print("this work?")
        # cv2.imshow("painted", painted_image)
        # cv2.waitKey(0)

        painted_images.append(painted_image)

        painted_image_with_label = masked_painted_image.copy()
        for i, bbox in enumerate(bboxes):
            text = text_prompt[i]
            painted_image_with_label = bbox_drawer_with_text(
                painted_image_with_label, bbox, text, i * 3 + 1
            )

        masked_painted_images_with_bbox.append(painted_image_with_label)
        # for debug
        if args.debug:
            # visualize cropped roi image like [H, W*N, C]
            # debug_image = np.concatenate(cropped_roi, axis=1)
            # debug_image = np.concatenate(object_mask_images, axis=1)
            # cv2.imshow("debug", cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))

            painted_image_with_label = masked_painted_image.copy()
            for i, bbox in enumerate(bboxes):
                text = text_prompt[i]
                painted_image_with_label = bbox_drawer_with_text(
                    painted_image_with_label, bbox, text, i * 3 + 1
                )
            # for i, bbox in enumerate(bboxes):
            #     painted_image_with_label = cv2.putText(painted_image_with_label, str(i), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            #     # draw bbox
            #     painted_image_with_label = cv2.rectangle(painted_image_with_label, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)

            cv2.imshow("painted", painted_image_with_label)
            cv2.waitKey(1)
    xmem.clear_memory()

    # write painted images to video
    height, width, _ = painted_images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter("painted.mp4", fourcc, 30.0, (width, height))
    for painted_image in painted_images:
        video.write(painted_image)
    video.release()

    # write masked painted images to video
    height, width, _ = masked_painted_images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter("masked_painted.mp4", fourcc, 30.0, (width, height))
    for masked_painted_image in masked_painted_images:
        video.write(masked_painted_image)
    video.release()

    # write concatenated original images and masked painted images to video
    height, width, _ = masked_painted_images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter("concat.mp4", fourcc, 30.0, (width, height * 2))
    for original_image, masked_painted_image_with_bbox in zip(
        original_images, masked_painted_images_with_bbox
    ):
        # concatenate original image and masked painted image to be (H*2, W, C)
        # concat_image = cv2.vconcat([original_image, masked_painted_image])
        concat_image = np.concatenate(
            [original_image, masked_painted_image_with_bbox], axis=0
        )
        video.write(concat_image)
    video.release()

    # write original images to video
    height, width, _ = original_images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter("original.mp4", fourcc, 30.0, (width, height))
    for original_image in original_images:
        video.write(original_image)
    video.release()

    # write masked painted images with bbox to video
    height, width, _ = masked_painted_images_with_bbox[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(
        "masked_painted_with_bbox.mp4", fourcc, 30.0, (width, height)
    )
    for masked_painted_image_with_bbox in masked_painted_images_with_bbox:
        video.write(masked_painted_image_with_bbox)
    video.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("-pn", "--project_name", type=str)
    parser.add_argument("-tp", "--text_prompt", nargs="+", type=str, default=[])
    parser.add_argument("-bt", "--box_threshold", type=float, default=0.35)
    parser.add_argument("-tt", "--text_threshold", type=float, default=0.25)
    parser.add_argument("-i", "--interactive", action="store_true")
    parser.add_argument("-n", "--num_objects", type=int, default=3)
    args = parser.parse_args()

    main(args)
