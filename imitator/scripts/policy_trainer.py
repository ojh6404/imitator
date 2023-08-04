#!/usr/bin/env python3

import os
import argparse
from imitator.models.base_nets import Normalize, Unnormalize
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


# verify model
@torch.no_grad()
def verify(model, dataset):
    import matplotlib.pyplot as plt

    model.eval()
    random_index = np.random.randint(0, len(dataset))
    test_action_seq = dataset[random_index]["actions"]  # [T, D]
    test_obs_seq = dataset[random_index]["obs"]  # [T, D]
    test_obs_seq = TensorUtils.to_batch(test_obs_seq)  # [1, T, D]
    pred_action_seq = model(test_obs_seq, unnormalize=True)  # [1, T, D]
    pred_action_seq = TensorUtils.squeeze(pred_action_seq, 0)  # [T, D]
    plt.plot(test_action_seq, label="ground truth")
    plt.plot(pred_action_seq, label="prediction")
    plt.legend()
    plt.show()


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
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    seq_length = config.network.policy.rnn.seq_length if args.model == "rnn" else 1

    image_obs_keys = [
        obs_key
        for obs_key in obs_keys
        if config.obs[obs_key].modality == "ImageModality"
    ]
    for image_obs in image_obs_keys:
        if config.obs[image_obs].obs_encoder.model_path is None:
            obs_default_model_path = os.path.join(
                FileUtils.get_models_folder(args.project_name),
                f"{image_obs}_model.pth",
            )
            if not os.path.exists(obs_default_model_path):
                if config.obs[image_obs].obs_encoder.pretrained:
                    print("Use pretrained model")
                else:
                    raise ValueError(
                        f"Model for {image_obs} does not exist. Please specify a model path in config file."
                    )
            else:
                config.obs[image_obs].obs_encoder.model_path = obs_default_model_path

    if config.network.policy.model_path is None:
        default_model_path = os.path.join(
            FileUtils.get_models_folder(args.project_name),
            f"{args.model}_actor_model.pth",
        )
        config.network.policy.model_path = default_model_path

    dataset = SequenceDataset(
        hdf5_path=hdf5_path,
        obs_keys=obs_keys,  # observations we want to appear in batches
        seq_length=seq_length,  # length-10 temporal sequences
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

    device = torch.device(args.device)
    if config.actions.normalize:
        normalizer_cfg = FileUtils.get_normalize_cfg(args.project_name)
        action_mean, action_std = get_normalize_params(
            normalizer_cfg.actions.min, normalizer_cfg.actions.max
        )
        config.actions.update(
            {"max": normalizer_cfg.actions.max, "min": normalizer_cfg.actions.min}
        )
    else:
        action_mean, action_std = 0.0, 1.0
    action_normalizer = Normalize(mean=action_mean, std=action_std).to(device)

    for obs in obs_keys:
        normalizer_cfg = FileUtils.get_normalize_cfg(args.project_name)
        if config.obs[obs].normalize:
            config.obs[obs].update(
                {"max": normalizer_cfg.obs[obs].max, "min": normalizer_cfg.obs[obs].min}
            )

    actor_type = ACTOR_TYPES[args.model]
    model = actor_type(config).to(device)

    # load checkpoint if provided
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
        verify(model, dataset)
        del dataset
        return

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
    data_loader_iter = iter(data_loader)

    for epoch in range(1, num_epochs + 1):  # epoch numbers start at 1
        # data_loader_iter = iter(data_loader)
        # for _ in range(gradient_steps_per_epoch):
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
        groundtruth_action = action_normalizer(batch["actions"])

        loss = nn.MSELoss()(prediction, groundtruth_action)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
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

            # create model folder if not exists
            os.makedirs(
                FileUtils.get_models_folder(args.project_name), exist_ok=True
            )
            torch.save(model.state_dict(), config.network.policy.model_path)

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

    del model

    # test
    model = actor_type(config).to(device)
    model.load_state_dict(
        torch.load(FileUtils.get_best_runs(args.project_name, args.model))
    )
    verify(model, dataset)

    del dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", "--project_name", type=str)
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-e", "--num_epochs", type=int, default=3000)
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("-m", "--model", type=str, default="mlp")
    parser.add_argument("-r", "--resume", action="store_true", default=False)
    parser.add_argument("-v", "--verify", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("-ckpt", "--checkpoint", type=str)
    args = parser.parse_args()

    main(args)
