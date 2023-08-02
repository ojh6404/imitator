#!/usr/bin/env python3
import argparse
import os
import time
from tqdm import tqdm
from collections import OrderedDict

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import imitator.utils.tensor_utils as TensorUtils
import imitator.utils.file_utils as FileUtils
from imitator.utils.datasets import SequenceDataset
from imitator.models.base_nets import AutoEncoder, VariationalAutoEncoder


def main(args):
    device = torch.device(args.device)
    hdf5_path = (
        args.dataset
        if args.dataset
        else os.path.join(
            FileUtils.get_project_folder(args.project_name), "data/dataset.hdf5"
        )
    )
    obs_key = args.obs_key

    config = FileUtils.get_config_from_project_name(args.project_name)

    dataset = SequenceDataset(
        hdf5_path=hdf5_path,
        obs_keys=[obs_key],  # observations we want to appear in batches
        dataset_keys=config.dataset.dataset_keys,
        load_next_obs=True,
        frame_stack=1,
        seq_length=1,  # length-10 temporal sequences
        pad_frame_stack=True,
        pad_seq_length=True,  # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=None,  # cache dataset in memory to avoid repeated file i/o
        hdf5_use_swmr=True,
    )

    data_loader = DataLoader(
        dataset=dataset,
        sampler=None,  # no custom sampling logic (uniform sampling)
        batch_size=args.batch_size,  # batches of size 100
        shuffle=True,
        # shuffle=False,
        num_workers=0,
        drop_last=True,  # don't provide last batch in dataset pass if it's less than 100 in size
    )


    if args.model == "ae":
        model = AutoEncoder(
            input_size=config.obs[obs_key].obs_encoder.input_dim[:2],  # [224, 224]
            input_channel=config.obs[obs_key].obs_encoder.input_dim[2],
            latent_dim=config.obs[obs_key].obs_encoder.output_dim,
            normalization=nn.BatchNorm2d,
            output_activation=nn.Sigmoid,
        ).to(device)
    elif args.model == "vae":
        model = VariationalAutoEncoder(
            input_size=config.obs[obs_key].obs_encoder.input_dim[:2],  # [224, 224]
            input_channel=config.obs[obs_key].obs_encoder.input_dim[2],
            latent_dim=config.obs[obs_key].obs_encoder.output_dim,
            normalization=nn.BatchNorm2d,
            output_activation=nn.Sigmoid,
        ).to(device)
    else:
        raise ValueError("Invalid model type")


    # verify model
    @torch.no_grad()
    def verify(model):
        model.eval()
        random_index = np.random.randint(0, len(dataset))
        test_image = dataset[random_index]["obs"][obs_key]  # numpy ndarray [B,H,W,C]
        test_image_numpy = test_image.squeeze(0).astype(np.uint8)
        test_image_tensor = TensorUtils.to_device(TensorUtils.to_tensor(test_image), device)
        test_image_tensor = (
            test_image_tensor.permute(0, 3, 1, 2).float().contiguous() / 255.0
        )
        if args.model == "ae":
            x, z = model(test_image_tensor)
        elif args.model == "vae":
            x, z, mu, logvar = model(test_image_tensor)

        test_image_recon = (
            TensorUtils.to_numpy(x.squeeze(0).permute(1, 2, 0)) * 255.0
        ).astype(np.uint8)
        concat_image = np.concatenate(
            [test_image_numpy, test_image_recon], axis=1
        )
        concat_image = cv2.cvtColor(concat_image, cv2.COLOR_RGB2BGR)
        print("Embedding shape: ", z.shape)
        cv2.imshow("verify", concat_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # load checkpoint if resuming
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
        if args.verify:
            verify(model)
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

        batch_image = TensorUtils.to_device(batch["obs"][obs_key], device)
        batch_image = batch_image.reshape(-1, *batch_image.shape[2:]).permute(
            0, 3, 1, 2
        )  # (B, C, H, W)
        batch_image = batch_image.contiguous().float() / 255.0

        loss_sum = 0
        loss_dict = model.loss(batch_image)
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
            if args.model == "ae":
                print(f"epoch: {epoch}, loss: {loss:.5g}")
            elif args.model == "vae":
                recons_loss = loss_dict["reconstruction_loss"].item()
                kl_loss = loss_dict["kld_loss"].item()
                print(
                    f"epoch: {epoch}, loss: {loss:.5g}, recons_loss: {recons_loss:.5g}, kl_loss: {kl_loss:.5g}"
                )
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, args.model + "_model_" + str(epoch) + ".pth"),
            )

        if loss_sum.item() < best_loss and (epoch > args.num_epochs / 10):
            print(f"best model saved with loss {loss_sum.item():.5g}")
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, args.model + "_model_best.pth"),
            )
            best_loss = loss_sum.item()

        scheduler.step()

    summary_writer.close()

    del model
    # load model for test
    if args.model == "ae":
        model = AutoEncoder(
            input_size=config.obs[obs_key].obs_encoder.input_dim[:2],  # [224, 224]
            input_channel=config.obs[obs_key].obs_encoder.input_dim[2],
            latent_dim=config.obs[obs_key].obs_encoder.output_dim,
            normalization=nn.BatchNorm2d,
            output_activation=nn.Sigmoid,
        ).to(device)
    elif args.model == "vae":
        model = VariationalAutoEncoder(
            input_size=config.obs[obs_key].obs_encoder.input_dim[:2],  # [224, 224]
            input_channel=config.obs[obs_key].obs_encoder.input_dim[2],
            latent_dim=config.obs[obs_key].obs_encoder.output_dim,
            normalization=nn.BatchNorm2d,
            output_activation=nn.Sigmoid,
        ).to(device)
    else:
        raise ValueError("Invalid model type")

    model.load_state_dict(
        torch.load(os.path.join(output_dir, args.model + "_model_best.pth"))
    )
    verify(model)

    del dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", "--project_name", type=str)
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-e", "--num_epochs", type=int, default=3000)
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-m", "--model", type=str, default="ae")
    parser.add_argument("-obs", "--obs_key", type=str, default="image")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("-v","--verify", action="store_true", default=False)
    parser.add_argument("-ckpt", "--checkpoint", type=str)
    args = parser.parse_args()

    main(args)
