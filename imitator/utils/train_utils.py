#!/usr/bin/env python3


import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import imitator.utils.tensor_utils as TensorUtils
import imitator.utils.file_utils as FileUtils
from imitator.utils.datasets import SequenceDataset


# verify model
@torch.no_grad()
def verify(model, dataset, seed=None):
    import matplotlib.pyplot as plt

    if seed is not None:
        set_seed(seed)
    model.eval()
    random_index = np.random.randint(0, len(dataset))
    test_batch = TensorUtils.to_batch(dataset[random_index])  # [1, T, D]
    test_action = TensorUtils.squeeze(test_batch["actions"], 0)  # [T, D]
    test_batch = TensorUtils.to_tensor(test_batch)  # [1, T, D]
    test_batch = TensorUtils.to_device(test_batch, next(model.parameters()).device)
    pred_action = model.get_action(test_batch)  # [1, T, D]
    pred_action = TensorUtils.squeeze(pred_action, 0)  # [T, D]
    plt.plot(test_action, label="ground truth")
    plt.plot(pred_action, label="prediction")
    plt.legend()
    plt.show()


@torch.no_grad()
def verify_image(model, dataset, obs_key="image"):
    """
    only for vision model which has decoder
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    concat_image = np.concatenate([test_image_numpy, test_image_recon], axis=1)
    concat_image = cv2.cvtColor(concat_image, cv2.COLOR_RGB2BGR)
    print("Embedding shape: ", z.shape)
    cv2.imshow("verify", concat_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def train(model, batch, optimizer, max_grad_norm=None):
    model.train()
    train_info = model.forward_train(batch)  # all time steps
    train_loss = train_info["loss"]
    optimizer.zero_grad()
    train_loss.backward()
    if max_grad_norm is not None:
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.data.norm(2).pow(2).item()
    optimizer.step()
    train_loss = train_loss.item()
    return train_loss, grad_norm


@torch.no_grad()
def validate(model, dataloader):
    model.eval()
    device = next(model.parameters()).device
    valid_loss = 0.0
    for batch in dataloader:
        batch = TensorUtils.to_float(TensorUtils.to_device(batch, device))
        valid_info = model.forward_train(batch)
        _valid_loss = valid_info["loss"]
        valid_loss += _valid_loss.item()
    valid_loss /= len(dataloader.dataset)
    return valid_loss


def save_and_log(model, writer, logger_dict):
    project_name = logger_dict["project_name"]
    model_type = logger_dict["model_type"]
    output_dir = logger_dict["output_dir"]
    model_path = logger_dict["model_path"]
    epoch = logger_dict["epoch"]
    best_loss = logger_dict["best_loss"]
    train_loss = logger_dict["train/loss"]
    valid_loss = logger_dict["valid/loss"]
    train_lr = logger_dict["train/lr"]
    grad_norm = logger_dict["train/grad_norm"]
    inference_time = logger_dict["train/inference_time"]
    if epoch % 100 == 0:
        print(
            f"epoch: {epoch}, train loss: {train_loss:.5g}, valid loss: {valid_loss:.5g}"
        )
        torch.save(
            model.state_dict(),
            os.path.join(output_dir, model_type + "_model_" + str(epoch) + ".pth"),
        )
    if valid_loss < best_loss:
        print(f"best model saved with valid loss {valid_loss:.5g}")
        torch.save(
            model.state_dict(),
            os.path.join(output_dir, model_type + "_model_best.pth"),
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
    writer.add_scalar("train/grad_norm", grad_norm, global_step=epoch)
    # inference time
    writer.add_scalar("train/inference_time", inference_time, global_step=epoch)

    return best_loss


def build_dataloader(obs_keys, dataset_config, batch_size):
    train_dataset = SequenceDataset(
        obs_keys=obs_keys,  # observations we want to appear in batches
        filter_by_attribute="train",
        **dataset_config,
    )
    valid_dataset = SequenceDataset(
        obs_keys=obs_keys,  # observations we want to appear in batches
        filter_by_attribute="valid",
        **dataset_config,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=None,  # no custom sampling logic (uniform sampling)
        batch_size=batch_size,  # batches of size 100
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        sampler=None,  # no custom sampling logic (uniform sampling)
        batch_size=batch_size,  # batches of size 100
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    return train_dataloader, valid_dataloader


def set_seed(seed=42):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
