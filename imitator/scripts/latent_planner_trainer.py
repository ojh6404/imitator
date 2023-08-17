#!/usr/bin/env python3

import os
import argparse
import cv2
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

from imitator.models.latent_planner import LatentPlanner

from torchvision import transforms as T

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


def main(args):
    config = FileUtils.get_config_from_project_name(args.project_name)
    hdf5_path = args.dataset
    obs_keys = ["image"]
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    seq_len = 60

    dataset = SequenceDataset(
        hdf5_path=hdf5_path,
        dataset_keys=["actions"],
        load_next_obs=True,
        frame_stack=1,
        pad_frame_stack= True,
        pad_seq_length= True,
        get_pad_mask= False,
        # hdf5_cache_mode="all",
        hdf5_cache_mode=None,
        hdf5_use_swmr=True,
        obs_keys=obs_keys,  # observations we want to appear in batches
        seq_length=seq_len,
        goal_mode="last"
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
    # device = "cpu"
    if config.actions.normalize:
        normalizer_cfg = FileUtils.get_normalize_cfg(args.project_name)
        action_mean, action_std = get_normalize_params(
            normalizer_cfg.actions.min, normalizer_cfg.actions.max
        )
        # config.actions.update(
        #     {"max": normalizer_cfg.actions.max, "min": normalizer_cfg.actions.min}
        # )
        print("action mean: ", action_mean)
        print("action std: ", action_std)
    else:
        action_mean, action_std = 0.0, 1.0
    action_normalizer = Normalize(mean=action_mean, std=action_std).to(device)

    model = LatentPlanner(horizon=seq_len)
    model.to(device)

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

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    transform = T.Compose([
        T.Normalize(mean=MEAN, std=STD)
    ])

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


        # print(batch["obs"]["image"].shape)
        goal_image = batch["goal_obs"]["image"] # [B,H,W,C]
        goal_image = goal_image.permute(0,3,1,2).contiguous() / 255.0 # [B,C,H,W]
        goal_image = transform(goal_image) # Normalized [B,C,H,W]

        current_image = batch["obs"]["image"][:,0,:,:,:] # [B,H,W,C]
        current_image = current_image.permute(0,3,1,2).contiguous() / 255.0 # [B,C,H,W]
        current_image = transform(current_image) # Normalized [B,C,H,W]


        trajectory = batch["actions"] # [B,T,D]
        first_action = trajectory[:,0,:] # [B,D]
        first_action = action_normalizer(first_action) # [B,D]
        groundtruth_action = action_normalizer(trajectory) # [B,T,D]

        # calculate time and loss
        start_time = time.time()
        prediction = model(goal_image , current_image, first_action)
        end_time = time.time()

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

    # del model

    action_unnormalizer = Unnormalize(mean=action_mean, std=action_std).to(device)

    random_data = dataset[0] # not batch, numpy array

    goal_image = random_data["goal_obs"]["image"] # [H,W,C]
    goal_image = goal_image.transpose(2,0,1) / 255.0 # [C,H,W]
    goal_image = TensorUtils.to_tensor(goal_image).unsqueeze(0).to(device) # [1,C,H,W]
    goal_image = transform(goal_image) # Normalized [1,C,H,W]
    goal_image = TensorUtils.to_float(goal_image)

    current_image = random_data["obs"]["image"][0,:,:,:] # [H,W,C]
    current_image = current_image.transpose(2,0,1) / 255.0 # [C,H,W]
    current_image = TensorUtils.to_tensor(current_image).unsqueeze(0).to(device) # [1,C,H,W]
    current_image = transform(current_image) # Normalized [1,C,H,W]
    current_image = TensorUtils.to_float(current_image)

    first_action = random_data["actions"][0,:] # [D]
    first_action = TensorUtils.to_tensor(first_action).unsqueeze(0).to(device) # [1,D]
    first_action = action_normalizer(first_action) # [1,D]



    prediction = model(goal_image , current_image, first_action) # [1,T,D]
    prediction = action_unnormalizer(prediction) # [1,T,D]
    prediction = prediction.squeeze(0).cpu().detach().numpy() # [T,D]

    # predicted action is actually xy point in image, so plot it to current image
    current_image = random_data["obs"]["image"][0,:,:,:] # [H,W,C]
    # draw circle on image
    for i in range(prediction.shape[0]):
        x = int(prediction[i,0])
        y = int(prediction[i,1])
        # draw circle with timestep wise color
        # first timestep is red, last timestep is blue
        current_image = cv2.circle(current_image, (x,y), 2, (i*255//prediction.shape[0],0,0), -1)
        # current_image = cv2.circle(current_image, (x,y), 2, (0,0,i*255//prediction.shape[0]), -1)
        # current_image = cv2.circle(current_image, (x,y), 2, (0,0,255), -1)

    cv2.imshow("current_image", current_image)
    cv2.waitKey(0)





    # test model


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
