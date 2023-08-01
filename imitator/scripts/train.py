#!/usr/bin/env python3

import os
import yaml
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from easydict import EasyDict as edict

from imitator.utils.datasets import SequenceDataset
from imitator.models.policy_nets import MLPActor, RNNActor
import imitator.utils.tensor_utils as TensorUtils
import imitator.utils.file_utils as FileUtils
from imitator.utils.obs_utils import get_normalize_params

ACTOR_TYPES = {"mlp": MLPActor, "rnn": RNNActor}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", "--project_name", type=str)
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-e", "--num_epochs", type=int, default=3000)
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("-m", "--model", type=str, default="mlp")
    args = parser.parse_args()


    config = FileUtils.get_config_from_project_name(args.project_name)
    hdf5_path = args.dataset if args.dataset else os.path.join(FileUtils.get_project_folder(args.project_name), "data/dataset.hdf5")
    obs_keys = list(config.obs.keys())
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    seq_length = config.network.policy.rnn.seq_length if args.model == "rnn" else 1
    gradient_steps_per_epoch = 100

    dataset = SequenceDataset(
        hdf5_path=hdf5_path,
        obs_keys=obs_keys,  # observations we want to appear in batches
        **config.dataset,
    )
    data_loader = DataLoader(
        dataset=dataset,
        sampler=None,  # no custom sampling logic (uniform sampling)
        batch_size=batch_size,  # batches of size 100
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    normalize = True  # TODO
    if normalize:
        normalizer_cfg = FileUtils.get_normalize_cfg(args.project_name)
        action_mean, action_std = get_normalize_params(normalizer_cfg.actions.min, normalizer_cfg.actions.max)
        action_mean, action_std = torch.Tensor(action_mean).to(device).float(), torch.Tensor(action_std).to(device).float()
        config.actions.update({"max": normalizer_cfg.actions.max, "min": normalizer_cfg.actions.min})
        for obs in normalizer_cfg.obs:
            config.obs[obs].update({"max": normalizer_cfg.obs[obs].max, "min": normalizer_cfg.obs[obs].min})
    else:
        action_mean, action_std = 0.0, 1.0



    actor_type = ACTOR_TYPES[args.model]
    model = actor_type(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[args.num_epochs // 2, args.num_epochs // 4 * 3],
        gamma=0.1,
    )

    print(model)

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

    model.train()

    best_loss = np.inf

    for epoch in range(1, num_epochs + 1):  # epoch numbers start at 1
        data_loader_iter = iter(data_loader)
        for _ in range(gradient_steps_per_epoch):
            try:
                batch = next(data_loader_iter)
            except StopIteration:
                # data loader ran out of batches - reset and yield first batch
                data_loader_iter = iter(data_loader)
                batch = next(data_loader_iter)
            batch = TensorUtils.to_float(TensorUtils.to_device(batch, device))

            # calculate time and loss
            start_time = time.time()
            prediction = model(batch["obs"])  # [B, T, D]
            end_time = time.time()
            action = (batch["actions"] - action_mean) / action_std

            loss = nn.MSELoss()(prediction, action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if epoch == 1 and _ == 0:
            #     import cv2
            #     with torch.no_grad():
            #         test_image = TensorUtils.to_device(
            #             batch["obs"]["image"], "cuda:0"
            #         )  # [B, T, H, W, C]
            #         # batch_image = batch_image.reshape(
            #         #     -1, *batch_image.shape[2:]
            #         # ).permute(
            #         #     0, 3, 1, 2
            #         # )  # (B, C, H, W)
            #         batch_image = batch_image.contiguous().float() / 255.0
            #         vae_encoder = (
            #             model.nets["obs_encoder"].nets["image"].nets["encoder"]
            #         )
            #         vae_decoder = (
            #             model.nets["obs_encoder"].nets["image"].nets["decoder"]
            #         )
            #         print("batch_image: ", batch_image.shape, batch_image.dtype)
            #         latent, _, _ = vae_encoder(batch_image)
            #         recon = vae_decoder(latent)
            #         recon_image = recon.reshape(-1, *batch_image.shape[1:]).permute(
            #             0, 2, 3, 1
            #         )  # (B, H, W, C)
            #         recon_image = (recon_image * 255.0).byte().cpu().numpy()
            #         # visualize
            #         for i in range(0, batch_image.shape[0], 10):
            #             import cv2
            #             cv2.imshow(
            #                 "original", batch_image[i].permute(1, 2, 0).cpu().numpy()
            #             )
            #             cv2.imshow("recon", recon_image[i])
            #             cv2.waitKey(0)
            # input("Press Enter to continue...")

        if epoch % 10 == 0:
            print(f"epoch: {epoch}, loss: {loss.item():.5g}")
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, args.model + "_model_" + str(epoch) + ".pth"),
            )

        if loss.item() < best_loss:
            print(f"best model saved with loss {loss.item():.5g}")
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, args.model + "_model_best.pth"),
            )
            best_loss = loss.item()

        summary_writer.add_scalar("train/loss", loss.item(), global_step=epoch)
        # lr rate
        summary_writer.add_scalar(
            "train/lr", optimizer.param_groups[0]["lr"], global_step=epoch
        )
        # inference time
        summary_writer.add_scalar(
            "train/inference_time", end_time - start_time, global_step=epoch
        )

        scheduler.step()

    summary_writer.close()
    import matplotlib.pyplot as plt

    random_batch = TensorUtils.to_device(next(data_loader_iter), device)

    actions_of_first_batch = random_batch["actions"][0]  # [T, D]
    print("actions_of_first_batch shape: ", actions_of_first_batch.shape)  # [T, D]

    # testing
    model.eval()
    with torch.no_grad():
        prediction = model(random_batch["obs"])
        action = random_batch["actions"]

    prediction_of_first_batch = prediction[0]  # [T, D]
    print(
        "prediction_of_first_batch shape: ", prediction_of_first_batch.shape
    )  # [T, D]

    plt.figure()
    plt.plot(
        actions_of_first_batch.cpu().numpy(),
        label="actions",
    )
    plt.plot(
        prediction_of_first_batch.cpu().numpy(),
        label="prediction",
    )
    plt.legend()
    plt.show()

    # testing step by step and whole sequence
