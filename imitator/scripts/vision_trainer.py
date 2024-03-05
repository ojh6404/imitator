#!/usr/bin/env python3
import argparse
import os
import time

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import imitator.utils.tensor_utils as TensorUtils
import imitator.utils.train_utils as TrainUtils
import imitator.utils.file_utils as FileUtils
import imitator.utils.obs_utils as ObsUtils
from imitator.utils.datasets import ImageDataset
from imitator.models.obs_nets import AutoEncoder, VariationalAutoEncoder

from torchvision import transforms as T


@torch.no_grad()
def validate(model, dataloader, obs_key, obs_dim, data_augmentation=False):
    model.eval()
    device = next(model.parameters()).device
    valid_loss = 0.0
    for batch in dataloader:
        batch_image = TensorUtils.to_device(batch["obs"][obs_key], device).reshape(-1, *obs_dim)
        batch_image = batch_image.permute(0, 3, 1, 2)  # (B, C, H, W)
        batch_image = batch_image.contiguous().float() / 255.0
        if data_augmentation:
            batch_image = random_resize_crop(batch_image)
        ground_truth = batch_image.detach().clone()
        if data_augmentation:
            batch_image = transform(batch_image).contiguous()
        valid_info = model.forward_train({"obs": batch_image, "ground_truth": ground_truth})
        _valid_loss = valid_info["loss"]
        valid_loss += _valid_loss.item()
    valid_loss /= len(dataloader.dataset)
    return valid_loss


def main(args):
    config = FileUtils.get_config_from_project_name(args.project_name)

    TrainUtils.set_seed(args.seed)
    device = torch.device(args.device)
    obs_key = args.obs_key
    data_augmentation = config.obs[obs_key].data_augmentation
    obs_dim = config.obs[obs_key].obs_encoder.input_dim

    if config.obs[obs_key].obs_encoder.model_path is None:
        config.obs[obs_key].obs_encoder.model_path = os.path.join(
            FileUtils.get_models_folder(args.project_name),
            obs_key + "_model.pth",
        )
    model_path = config.obs[obs_key].obs_encoder.model_path
    config.dataset.hdf5_path = (
        args.dataset
        if args.dataset
        else os.path.join(FileUtils.get_project_folder(args.project_name), "data/dataset.hdf5")
    )
    config.dataset.seq_length = config.dataset.frame_stack = 1
    model_type = config.obs[obs_key].obs_encoder.model
    model = eval(model_type)(**config.obs[obs_key].obs_encoder.model_kwargs).to(device)
    train_dataloader, valid_dataloader = TrainUtils.build_dataloader([obs_key], config.dataset, args.batch_size)

    print("\n================ Config ================")
    FileUtils.print_config(config)
    print("========================================")
    print("\n================ Dataset ================")
    print("Loaded Train Dataset Trajectory Lengths: ", len(train_dataloader.dataset))
    print("Loaded Valid Dataset Trajectory Lengths: ", len(valid_dataloader.dataset))
    print("========================================")
    print("\n================ Model ================")
    print(model)
    print("========================================")

    # data augmentation
    if data_augmentation:
        transform = T.Compose(
            [
                ObsUtils.AddGaussianNoise(mean=0.0, std=0.1, p=0.2),
                # RGBShifter(r_shift_limit=0.2, g_shift_limit=0.2, b_shift_limit=0.2, p=1.0),
                ObsUtils.RGBShifter(r_shift_limit=0.1, g_shift_limit=0.1, b_shift_limit=0.1, p=0.2),
                # T.RandomApply([T.RandomResizedCrop(size=config.obs[obs_key].obs_encoder.input_dim[:2], scale=(0.8, 1.0), ratio=(0.8, 1.2), antialias=True)], p=0.5),
                # T.RandomApply([T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p=1.0),
            ]
        )
        random_resize_crop = T.RandomApply(
            [
                T.RandomResizedCrop(
                    size=config.obs[obs_key].obs_encoder.input_dim[:2],
                    scale=(0.8, 1.0),
                    ratio=(0.8, 1.2),
                    antialias=True,
                )
            ],
            p=0.2,
        )

    # load checkpoint if resuming
    if args.resume:
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint))
        else:
            model.load_state_dict(torch.load(FileUtils.get_best_runs(args.project_name, model_type)))
    elif args.verify:
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint))
        else:
            model.load_state_dict(torch.load(FileUtils.get_best_runs(args.project_name, model_type)))
        TrainUtils.verify_image(model, valid_dataloader.dataset, obs_key=obs_key, noise=False)
        return

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[args.num_epochs // 2],
        gamma=0.1,
    )

    # make dir and tensorboard writer
    os.makedirs(
        os.path.join(FileUtils.get_project_folder(args.project_name), "runs"),
        exist_ok=True,
    )
    output_dir = os.path.join(
        FileUtils.get_project_folder(args.project_name),
        "runs",
        model_type + "_" + time.strftime("%Y-%m-%d_%H-%M-%S"),
    )
    summary_writer = SummaryWriter(output_dir)

    best_loss = np.inf
    train_dataloader_iter = iter(train_dataloader)

    print("\n================ Training ================")
    for epoch in range(1, args.num_epochs + 1):  # epoch numbers start at 1
        try:
            batch = next(train_dataloader_iter)
        except StopIteration:
            # data loader ran out of batches - reset and yield first batch
            train_dataloader_iter = iter(train_dataloader)
            batch = next(train_dataloader_iter)
        batch = TensorUtils.to_device(batch["obs"][obs_key], device)  # [B, 1, H, W, C]
        batch = TensorUtils.squeeze(batch, 1)  # [B, H, W, C]
        batch_image = batch.permute(0, 3, 1, 2)  # (B, C, H, W)
        batch_image = batch_image.contiguous().float() / 255.0
        if data_augmentation:
            batch_image = random_resize_crop(batch_image)
        ground_truth = batch_image.detach().clone()
        if data_augmentation:
            batch_image = transform(batch_image).contiguous()

        batch = {"obs": batch_image, "ground_truth": ground_truth}

        # calculate time and loss
        start_time = time.time()
        train_loss, grad_norm = TrainUtils.train(
            model,
            batch,
            optimizer,
            max_grad_norm=None,  # TODO
        )
        end_time = time.time()

        # validation
        valid_loss = validate(
            model,
            valid_dataloader,
            obs_key,
            obs_dim,
        )

        logger_dict = {
            "project_name": args.project_name,
            "model_type": model_type,
            "output_dir": output_dir,
            "model_path": model_path,
            "epoch": epoch,
            "best_loss": best_loss,
            "train/loss": train_loss,
            "train/mse": train_loss,
            "train/grad_norm": grad_norm,
            "valid/loss": valid_loss,
            "train/lr": optimizer.param_groups[0]["lr"],
            "train/inference_time": end_time - start_time,
        }

        best_loss = TrainUtils.save_and_log(
            model,
            summary_writer,
            logger_dict,
        )

        scheduler.step()
    summary_writer.close()

    del model

    # load model for test
    model = eval(model_type)(**config.obs[obs_key].obs_encoder.model_kwargs).to(device)
    model.load_state_dict(torch.load(os.path.join(output_dir, model_type + "_model_best.pth")))
    TrainUtils.verify_image(model, valid_dataloader.dataset, obs_key=obs_key, noise=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", "--project_name", type=str, required=True, help="project name")
    parser.add_argument("-d", "--dataset", type=str, help="path to hdf5 dataset")
    parser.add_argument("-e", "--num_epochs", type=int, default=3000, help="num epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=512, help="batch size")
    parser.add_argument("-obs", "--obs_key", type=str, default="image")
    parser.add_argument("-r", "--resume", action="store_true", default=False, help="resume training")
    parser.add_argument("-v", "--verify", action="store_true", default=False, help="verify mode")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--seed", type=int, default=None, help="seed")
    parser.add_argument("-ckpt", "--checkpoint", type=str, help="checkpoint path")
    args = parser.parse_args()

    main(args)
