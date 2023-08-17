#!/usr/bin/env python

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


from imitator.models.obs_nets import Resnet


class LatentPlanner(nn.Module):
    def __init__(self, horizon):
        super(LatentPlanner, self).__init__()

        self.goal_image_encoder = Resnet(
            input_size = [224, 224],
            input_channel = 3,
            resnet_type= "resnet18",  # resnet18, resnet34, resnet50, resnet101, resnet152
            pool=None,
            latent_dim=64,
        )

        self.current_image_encoder = Resnet(
            input_size = [224, 224],
            input_channel = 3,
            resnet_type= "resnet18",  # resnet18, resnet34, resnet50, resnet101, resnet152
            pool=None,
            latent_dim=64,
        )

        self.horizon = horizon
        self.dim = 2

        self.decoder = nn.Sequential(
            nn.Linear(64 + 64 + 2, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, self.horizon * self.dim),
        )


    def forward(self, goal_image, current_image, hand_xy):
        # goal_image: [B, 3, 224, 224]
        # current_image: [B, 3, 224, 224]
        # hand_xy: [B, 2]
        # output: [B, 30, 2]

        latent = self.encode(goal_image, current_image, hand_xy)
        traj = self.decode(latent)
        return traj

    def decode(self, latent):
        # latent: [B, 64 + 64 + 2]
        # output: [B, 30, 2]
        traj = self.decoder(latent)
        traj = traj.view(-1, self.horizon, self.dim)
        return traj

    def encode(self, goal_image, current_image, hand_xy):
        # goal_image: [B, 3, 224, 224]
        # current_image: [B, 3, 224, 224]
        # hand_xyz: [B, 2]
        # output: [B, 64 + 64 + 2]

        goal_latent = self.goal_image_encoder(goal_image)
        current_latent = self.current_image_encoder(current_image)
        # concat latent and hand_xyz
        latent = torch.cat([goal_latent, current_latent, hand_xy], dim=1)
        return latent
