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
import imitator.utils.obs_utils as ObsUtils
from imitator.utils.datasets import SequenceDataset
from imitator.models.obs_nets import AutoEncoder, VariationalAutoEncoder


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


# verify model
@torch.no_grad()
def verify_image(model, dataset, obs_key="image", noise=False):
    import torchvision.transforms as T

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    random_index = np.random.randint(0, len(dataset))
    test_image = dataset[random_index]["obs"][obs_key]  # numpy ndarray [1, H, W, C]
    if noise:
        add_noise = T.Compose(
            [
                ObsUtils.AddGaussianNoise(mean=0.0, std=0.1, p=1.0),
                ObsUtils.RGBShifter(
                    r_shift_limit=0.1, g_shift_limit=0.1, b_shift_limit=0.1, p=1.0
                ),
                T.RandomApply(
                    [
                        T.RandomResizedCrop(
                            size=test_image.shape[1:3],
                            scale=(0.8, 1.0),
                            ratio=(0.8, 1.2),
                            antialias=True,
                        )
                    ],
                    p=1.0,
                ),
            ]
        )
    test_image_tensor = TensorUtils.to_float(
        TensorUtils.to_device(TensorUtils.to_tensor(test_image), device)
    )
    test_image_tensor = (
        test_image_tensor.permute(0, 3, 1, 2).contiguous() / 255.0
    )  # (1, C, H, W)
    if noise:
        test_image_tensor = add_noise(test_image_tensor)
    test_image = (
        test_image_tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy() * 255.0
    ).astype(np.uint8)

    x, z = model(test_image_tensor)
    test_image_recon = (
        TensorUtils.to_numpy(x.squeeze(0).permute(1, 2, 0)) * 255.0
    ).astype(np.uint8)
    concat_image = ObsUtils.concatenate_image(test_image, test_image_recon)
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
    # common info
    epoch = logger_dict.pop("epoch")
    best_loss = logger_dict.pop("best_loss")
    train_loss = logger_dict.pop("train/loss")
    valid_loss = logger_dict.pop("valid/loss")
    train_lr = logger_dict.pop("train/lr")
    grad_norm = logger_dict.pop("train/grad_norm")
    inference_time = logger_dict.pop("train/inference_time")
    output_dir = logger_dict.pop("output_dir")
    model_type = logger_dict.pop("model_type")
    model_path = logger_dict.pop("model_path")
    project_name = logger_dict.pop("project_name")

    # TODO add more info for each model

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
