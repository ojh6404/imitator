#!/usr/bin/env python3

import os
import argparse
from imitator.models.base_nets import Normalize, Unnormalize
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from imitator.utils.datasets import SequenceDataset
from imitator.models.policy_nets import MLPActor, RNNActor, TransformerActor
import imitator.utils.tensor_utils as TensorUtils
import imitator.utils.file_utils as FileUtils
from imitator.utils.obs_utils import get_normalize_params

ACTOR_TYPES = {"mlp": MLPActor, "rnn": RNNActor, "transformer": TransformerActor}


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


def train(model, batch, optimizer, criterion, action_normalizer):
    model.train()
    prediction = model(batch["obs"])
    ground_truth = action_normalizer(batch["actions"])
    train_loss = criterion(prediction, ground_truth)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    return train_loss.item()


@torch.no_grad()
def validate(model, data_loader, criterion, action_normalizer):
    model.eval()
    device = next(model.parameters()).device
    valid_loss = 0.0
    for batch in data_loader:
        batch = TensorUtils.to_float(TensorUtils.to_device(batch, device))
        prediction = model(batch["obs"])
        ground_truth = action_normalizer(batch["actions"])
        valid_loss += criterion(prediction, ground_truth).item()
    valid_loss /= len(data_loader)
    return valid_loss


def save_and_log(model, writer, logger_dict):
    project_name = logger_dict["project_name"]
    output_dir = logger_dict["output_dir"]
    model_path = logger_dict["model_path"]
    epoch = logger_dict["epoch"]
    best_loss = logger_dict["best_loss"]
    train_loss = logger_dict["train/loss"]
    valid_loss = logger_dict["valid/loss"]
    train_lr = logger_dict["train/lr"]
    inference_time = logger_dict["train/inference_time"]
    if epoch % 100 == 0:
        print(
            f"epoch: {epoch}, train loss: {train_loss:.5g}, valid loss: {valid_loss:.5g}"
        )
        torch.save(
            model.state_dict(),
            os.path.join(output_dir, args.model + "_model_" + str(epoch) + ".pth"),
        )
    if valid_loss < best_loss:
        print(f"best model saved with valid loss {valid_loss:.5g}")
        torch.save(
            model.state_dict(),
            os.path.join(output_dir, args.model + "_model_best.pth"),
        )
        best_loss = valid_loss

        # create model folder if not exists
        os.makedirs(FileUtils.get_models_folder(project_name), exist_ok=True)
        torch.save(model.state_dict(), model_path)

    # write to tensorboard
    writer.add_scalar("train/loss", train_loss, global_step=epoch)
    writer.add_scalar("valid/loss", valid_loss, global_step=epoch)
    # lr rate
    writer.add_scalar("train/lr", train_lr, global_step=epoch)
    # inference time
    writer.add_scalar("train/inference_time", inference_time, global_step=epoch)

    return best_loss


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

    train_cfg = config.network.policy.train

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
                if config.obs[image_obs].obs_encoder.trainable:
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

    if config.network.policy.model == "TransformerActor":
        seq_length = 1
        print("TransformerActor only supports seq_length=1")

    train_dataset = SequenceDataset(
        hdf5_path=hdf5_path,
        obs_keys=obs_keys,  # observations we want to appear in batches
        seq_length=seq_length,  # length-10 temporal sequences
        filter_by_attribute="train",
        **config.dataset,
    )
    valid_dataset = SequenceDataset(
        hdf5_path=hdf5_path,
        obs_keys=obs_keys,  # observations we want to appear in batches
        seq_length=seq_length,  # length-10 temporal sequences
        filter_by_attribute="valid",
        **config.dataset,
    )

    print("Train Dataset Trajectory Lengths: ", len(train_dataset))
    print("Valid Dataset Trajectory Lengths: ", len(valid_dataset))

    train_data_loader = DataLoader(
        dataset=train_dataset,
        sampler=None,  # no custom sampling logic (uniform sampling)
        batch_size=batch_size,  # batches of size 100
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    valid_data_loader = DataLoader(
        dataset=valid_dataset,
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
        verify(model, valid_dataset)
        del train_dataset, valid_dataset
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
    train_data_loader_iter = iter(train_data_loader)

    for epoch in range(1, num_epochs + 1):  # epoch numbers start at 1
        try:
            batch = next(train_data_loader_iter)
        except StopIteration:
            # data loader ran out of batches - reset and yield first batch
            train_data_loader_iter = iter(train_data_loader)
            batch = next(train_data_loader_iter)
        batch = TensorUtils.to_float(TensorUtils.to_device(batch, device))

        # calculate time and loss
        start_time = time.time()
        train_loss = train(model, batch, optimizer, nn.MSELoss(), action_normalizer)
        end_time = time.time()

        # validation
        valid_loss = validate(model, valid_data_loader, nn.MSELoss(), action_normalizer)

        logger_dict = {
            "project_name": args.project_name,
            "output_dir": output_dir,
            "model_path": config.network.policy.model_path,
            "epoch": epoch,
            "best_loss": best_loss,
            "train/loss": train_loss,
            "valid/loss": valid_loss,
            "train/lr": optimizer.param_groups[0]["lr"],
            "train/inference_time": end_time - start_time,
        }

        best_loss = save_and_log(model, summary_writer, logger_dict)

        scheduler.step()
    summary_writer.close()

    del model

    # test
    model = actor_type(config).to(device)
    model.load_state_dict(
        torch.load(FileUtils.get_best_runs(args.project_name, args.model))
    )
    verify(model, valid_dataset)

    del train_dataset, valid_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-pn", "--project_name", type=str, required=True, help="project name"
    )
    parser.add_argument("-d", "--dataset", type=str, help="path to hdf5 dataset")
    parser.add_argument("-e", "--num_epochs", type=int, default=3000, help="num epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("-m", "--model", type=str, default="mlp", help="model type")
    parser.add_argument(
        "-r", "--resume", action="store_true", default=False, help="resume training"
    )
    parser.add_argument(
        "-v", "--verify", action="store_true", default=False, help="verify mode"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("-ckpt", "--checkpoint", type=str, help="checkpoint path")
    args = parser.parse_args()

    main(args)
