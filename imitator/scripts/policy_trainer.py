#!/usr/bin/env python3

import os
import argparse
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from imitator.models.policy_nets import MLPActor, RNNActor, TransformerActor
import imitator.utils.tensor_utils as TensorUtils
import imitator.utils.file_utils as FileUtils
import imitator.utils.train_utils as TrainUtils


def main(args):
    config = FileUtils.get_config_from_project_name(args.project_name)
    print("\n================ Config ================")
    FileUtils.print_config(config)
    print("========================================")

    TrainUtils.set_seed(args.seed)
    device = torch.device(args.device)
    model_type = config.network.policy.model
    model = eval(model_type)(config).to(device)

    train_config = config.network.policy.train
    train_config.batch_size = (
        args.batch_size if args.batch_size else train_config.batch_size
    )
    train_config.num_epochs = (
        args.num_epochs if args.num_epochs else train_config.num_epochs
    )
    config.dataset.hdf5_path = (
        args.dataset
        if args.dataset
        else os.path.join(
            FileUtils.get_project_folder(args.project_name), "data/dataset.hdf5"
        )
    )
    train_config.seq_length = (
        config.network.policy.rnn.rnn_horizon if isinstance(model, RNNActor) else 1
    )
    config.dataset.seq_length = train_config.seq_length
    config.dataset.frame_stack = (
        config.network.policy.transformer.context_length
        if isinstance(model, TransformerActor)
        else 1
    )
    obs_keys = list(config.obs.keys())
    train_dataloader, valid_dataloader = TrainUtils.build_dataloader(
        obs_keys, config.dataset, train_config.batch_size
    )

    print("\n================ Dataset ================")
    print("Loaded Train Dataset Trajectory Lengths: ", len(train_dataloader.dataset))
    print("Loaded Valid Dataset Trajectory Lengths: ", len(valid_dataloader.dataset))
    print("========================================")

    print("\n================ Model ================")
    print(model)
    print("=======================================")

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
            f"{model_type}_actor_model.pth",
        )
        config.network.policy.model_path = default_model_path

    # load checkpoint if provided
    if args.resume:
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint))
        else:
            model.load_state_dict(
                torch.load(FileUtils.get_best_runs(args.project_name, model_type))
            )
    elif args.verify:
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint))
        else:
            model.load_state_dict(
                torch.load(FileUtils.get_best_runs(args.project_name, model_type))
            )
        TrainUtils.verify(model, valid_dataloader.dataset, seed=args.seed)
        return

    optimizer = eval("optim." + train_config.get("optimizer", "Adam"))(
        model.parameters(),
        lr=train_config.get("lr"),
        weight_decay=train_config.get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[train_config.num_epochs // 2, train_config.num_epochs // 4 * 3],
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
    for epoch in range(1, train_config.num_epochs + 1):  # epoch numbers start at 1
        try:
            batch = next(train_dataloader_iter)
        except StopIteration:
            # data loader ran out of batches - reset and yield first batch
            train_dataloader_iter = iter(train_dataloader)
            batch = next(train_dataloader_iter)
        batch = TensorUtils.to_float(TensorUtils.to_device(batch, device))

        # calculate time and loss
        start_time = time.time()
        train_loss, grad_norm = TrainUtils.train(
            model,
            batch,
            optimizer,
            max_grad_norm=train_config.max_grad_norm,
        )
        end_time = time.time()

        # validation
        valid_loss = TrainUtils.validate(
            model,
            valid_dataloader,
        )

        logger_dict = {
            "project_name": args.project_name,
            "model_type": model_type,
            "output_dir": output_dir,
            "model_path": config.network.policy.model_path,
            "epoch": epoch,
            "best_loss": best_loss,
            "train/loss": train_loss,
            "train/mse": train_loss,
            "train/grad_norm": grad_norm,
            "valid/loss": valid_loss,
            "train/lr": optimizer.param_groups[0]["lr"],
            "train/inference_time": end_time - start_time,
        }

        best_loss = TrainUtils.save_and_log(model, summary_writer, logger_dict)

        scheduler.step()
    summary_writer.close()

    del model

    # test
    model = actor_type(config).to(device)
    model.load_state_dict(
        torch.load(FileUtils.get_best_runs(args.project_name, model_type))
    )
    TrainUtils.verify(model, valid_dataloader.dataset, seed=args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-pn", "--project_name", type=str, required=True, help="project name"
    )
    parser.add_argument("-d", "--dataset", type=str, help="path to hdf5 dataset")
    parser.add_argument("-e", "--num_epochs", type=int, default=3000, help="num epochs")
    parser.add_argument("-b", "--batch_size", type=int, help="batch size")
    parser.add_argument(
        "-r", "--resume", action="store_true", default=False, help="resume training"
    )
    parser.add_argument(
        "-v", "--verify", action="store_true", default=False, help="verify mode"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--seed", type=int, default=None, help="seed")
    parser.add_argument("-ckpt", "--checkpoint", type=str, help="checkpoint path")
    args = parser.parse_args()

    main(args)
