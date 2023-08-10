#!/usr/bin/env python3
import argparse
import os
import time

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

import imitator.utils.tensor_utils as TensorUtils
import imitator.utils.file_utils as FileUtils
from imitator.utils.datasets import ImageDataset
from imitator.models.obs_nets import AutoEncoder, VariationalAutoEncoder
from imitator.utils.obs_utils import concatenate_image, AddGaussianNoise, RGBShifter

from torchvision import transforms as T

# verify model
@torch.no_grad()
def verify(model, dataset, obs_key="image"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    random_index = np.random.randint(0, len(dataset))
    test_image = dataset[random_index]["obs"][obs_key]  # numpy ndarray (H, W, C)


    test_image_tensor =  TensorUtils.to_float(TensorUtils.to_device(TensorUtils.to_tensor(test_image), device))
    test_image_tensor = test_image_tensor.unsqueeze(0).permute(0, 3, 1, 2).contiguous() / 255.0 # (1, C, H, W)
    if args.model == "ae":
        x, z = model(test_image_tensor)
    elif args.model == "vae":
        x, z, mu, logvar = model(test_image_tensor)

    test_image_recon = (
        TensorUtils.to_numpy(x.squeeze(0).permute(1, 2, 0)) * 255.0
    ).astype(np.uint8)
    concat_image = concatenate_image(test_image, test_image_recon)
    concat_image = cv2.cvtColor(concat_image, cv2.COLOR_RGB2BGR)
    print("Embedding shape: ", z.shape)
    cv2.imshow("verify", concat_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def main(args):
    device = torch.device(args.device)
    hdf5_path = (
        args.dataset
        if args.dataset
        else os.path.join(
            FileUtils.get_project_folder(args.project_name), "data/image_dataset.hdf5"
        )
    )
    obs_key = args.obs_key
    config = FileUtils.get_config_from_project_name(args.project_name)


    if config.obs[obs_key].obs_encoder.model_path is None:
        config.obs[obs_key].obs_encoder.model_path = os.path.join(
            FileUtils.get_models_folder(args.project_name),
            obs_key + "_model.pth",
        )

    # data augmentation
    if config.obs[obs_key].data_augmentation:
        transform = T.Compose(
            [
                AddGaussianNoise(mean=0.0, std=0.1, p=0.5),
                # RGBShifter(r_shift_limit=0.2, g_shift_limit=0.2, b_shift_limit=0.2, p=1.0),
                RGBShifter(r_shift_limit=0.1, g_shift_limit=0.1, b_shift_limit=0.1, p=0.5),
                T.RandomApply([T.RandomResizedCrop(size=config.obs[obs_key].obs_encoder.input_dim[:2], scale=(0.8, 1.0), ratio=(0.8, 1.2), antialias=True)], p=0.5),
                # T.RandomApply([T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p=1.0),
            ]
        )


    dataset = ImageDataset(
        hdf5_path=hdf5_path,
        obs_keys=[obs_key],  # observations we want to appear in batches
        hdf5_cache_mode=True,  # cache dataset in memory to avoid repeated file i/o
        hdf5_use_swmr=True,
    )

    # train_dataset, valid_dataset = random_split(
    #     dataset, [int(len(dataset) * args.ratio), len(dataset) - int(len(dataset) * args.ratio)]
    # )

    print("Dataset size: ", len(dataset))

    data_loader = DataLoader(
        dataset=dataset,
        sampler=None,  # no custom sampling logic (uniform sampling)
        batch_size=args.batch_size,  # batches of size 100
        shuffle=True,
        num_workers=0,
        drop_last=True,  # don't provide last batch in dataset pass if it's less than 100 in size
        # collate_fn= # TODO collate fn to numpy ndarray
    )

    model = eval(config.obs[obs_key].obs_encoder.model)(
        **config.obs[obs_key].obs_encoder.model_kwargs
    ).to(device)

    # load checkpoint if resuming
    if args.resume:
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint))
        else:
            model.load_state_dict(
                torch.load(FileUtils.get_best_runs(args.project_name, args.model))
            )
    elif args.verify:
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint))
        else:
            model.load_state_dict(
                torch.load(FileUtils.get_best_runs(args.project_name, args.model))
            )
        verify(model, dataset, obs_key=obs_key)
        del dataset
        return

    print(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        # milestones=[args.num_epochs // 2, args.num_epochs // 4 * 3],
        milestones=[args.num_epochs // 2],
        gamma=0.1,
    )
    best_loss = np.inf

    # make dir and tensorboard writer
    os.makedirs(
        os.path.join(FileUtils.get_project_folder(args.project_name), "runs"),
        exist_ok=True,
    )
    output_dir = os.path.join(
        FileUtils.get_project_folder(args.project_name),
        "runs",
        args.model + "_" + time.strftime("%Y-%m-%d_%H-%M-%S"),
    )
    summary_writer = SummaryWriter(output_dir)

    for epoch in range(1, args.num_epochs + 1):  # epoch numbers start at 1
        data_loader_iter = iter(data_loader)
        try:
            batch = next(data_loader_iter)
        except StopIteration:
            data_loader_iter = iter(data_loader)
            batch = next(data_loader_iter)

        batch_image = TensorUtils.to_device(batch["obs"][obs_key], device) # (B, H, W, C)
        batch_image = batch_image.permute(0, 3, 1, 2)  # (B, C, H, W)
        batch_image = batch_image.contiguous().float() / 255.0
        if config.obs[obs_key].data_augmentation:
            batch_image = transform(batch_image).contiguous()

        # debug for verify data augmentation, concatenate 2 images original and augmented
        # original = (batch_image.detach().cpu().numpy() * 255).astype(np.uint8)
        # original = original[0].transpose(1, 2, 0)
        # cv2.imshow("original", cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)


        loss_sum = 0
        loss_dict = model.loss(x=batch_image, ground_truth=batch_image)
        for loss in loss_dict.values():
            loss_sum += loss

        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()

        summary_writer.add_scalar("train/loss", loss_sum.item(), global_step=epoch)
        # lr rate
        summary_writer.add_scalar(
            "train/lr", optimizer.param_groups[0]["lr"], global_step=epoch
        )

        # print loss with 5 significant digits every 100 epochs
        if epoch % 100 == 0:
            loss = loss_sum.item()
            if config.obs[obs_key].obs_encoder.model == "AutoEncoder":
                print(f"epoch: {epoch}, loss: {loss:.5g}")
            elif config.obs[obs_key].obs_encoder.model == "VariationalAutoEncoder":
                recons_loss = loss_dict["reconstruction_loss"].item()
                kl_loss = loss_dict["kld_loss"].item()
                print(
                    f"epoch: {epoch}, loss: {loss:.5g}, recons_loss: {recons_loss:.5g}, kl_loss: {kl_loss:.5g}"
                )
            # mkdir if not exist
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, args.model + "_model_" + str(epoch) + ".pth"),
            )

        if loss_sum.item() < best_loss and (epoch > args.num_epochs / 10):
            print(f"best model saved with loss {loss_sum.item():.5g}")
            # save in log dir
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, args.model + "_model_best.pth"),
            )
            # save in models dir
            os.makedirs(
                os.path.dirname(config.obs[obs_key].obs_encoder.model_path), exist_ok=True
                )
            # get files dir
            torch.save(
                model.state_dict(),
                config.obs[obs_key].obs_encoder.model_path,
            )

            best_loss = loss_sum.item()

        scheduler.step()

    summary_writer.close()

    del model
    # load model for test
    model = eval(config.obs[obs_key].obs_encoder.model)(
        **config.obs[obs_key].obs_encoder.model_kwargs
    ).to(device)

    model.load_state_dict(
        torch.load(os.path.join(output_dir, args.model + "_model_best.pth"))
    )
    verify(model, dataset, obs_key=obs_key)

    del dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", "--project_name", type=str)
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-e", "--num_epochs", type=int, default=3000)
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("--ratio", type=float, default=0.9)
    parser.add_argument("-m", "--model", type=str, default="ae")
    parser.add_argument("-obs", "--obs_key", type=str, default="image")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("-v", "--verify", action="store_true", default=False)
    parser.add_argument("-r", "--resume", action="store_true", default=False)
    parser.add_argument("-ckpt", "--checkpoint", type=str)
    args = parser.parse_args()

    main(args)
