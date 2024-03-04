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

from PIL import Image as PILImage

import clip
from lavis.models import load_model_and_preprocess

from imitator.utils import file_utils as FileUtils

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


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

    original_dataset = h5py.File(hdf5_path, "r")
    processed_dataset = h5py.File(hdf5_path.replace(".hdf5", "_with_mask.hdf5"), "w")

    device = "cpu"

    # for clip
    # model, preprocess = clip.load("ViT-B/32", device=device)
    # text_prompt = ["bread on a plate", "bread not on a plate"]
    # text = clip.tokenize(text_prompt).to(device)

    # for BLIP2
    # model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)
    model, vis_processors, text_processors = load_model_and_preprocess(
        "blip2_image_text_matching", "pretrain", device=device, is_eval=True
    )
    # model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "coco", device=device, is_eval=True)

    caption = "bread on a white plate"

    # copy attributes
    for key in original_dataset.attrs.keys():
        processed_dataset.attrs[key] = original_dataset.attrs[key]

    # demos
    for demo in tqdm(original_dataset["data"].keys()):
        demo_group = processed_dataset.create_group(demo)
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
            obs_group.create_dataset(
                obs_key,
                data=original_dataset["data"][demo]["obs"][obs_key],
                dtype=original_dataset["data"][demo]["obs"][obs_key].dtype,
            )

            # create mask if obs's modality is ImageModality
            if config.obs[obs_key].modality == "ImageModality":
                original_images = original_dataset["data"][demo]["obs"][
                    obs_key
                ]  # [T, H, W, C]

                # NOTE describe your latent extractor here
                for original_image in original_images:
                    original_image_pil = PILImage.fromarray(original_image)

                    # extract latent using CLIP
                    # clip_image = preprocess(original_image_pil).unsqueeze(0).to(device)

                    # with torch.no_grad():
                    #     image_features = model.encode_image(clip_image)
                    #     text_features = model.encode_text(text)

                    #     logits_per_image, logits_per_text = model(clip_image, text)
                    #     probs = logits_per_image.softmax(dim=-1).cpu().numpy()

                    # print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

                    # extract latent using BLIP2
                    img = (
                        vis_processors["eval"](original_image_pil)
                        .unsqueeze(0)
                        .to(device)
                    )
                    txt = text_processors["eval"](caption)

                    with torch.no_grad():
                        itm_output = model(
                            {"image": img, "text_input": txt}, match_head="itm"
                        )
                        itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
                        print(
                            f"The image and text are matched with a probability of {itm_scores[:, 1].item():.3%}"
                        )
                        itc_score = model(
                            {"image": img, "text_input": txt}, match_head="itc"
                        )
                        print(
                            "The image feature and text feature has a cosine similarity of %.4f"
                            % itc_score
                        )

                    cv2.imshow(
                        "original_image",
                        cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR),
                    )
                    cv2.waitKey(0)

    original_dataset.close()
    processed_dataset.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-pn", "--project_name", type=str)
    parser.add_argument("-tp", "--text_prompt", type=str)
    args = parser.parse_args()

    main(args)
