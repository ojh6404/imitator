#!/usr/bin/env python3


import os
import numpy as np
import cv2
import torch

import imitator.utils.tensor_utils as TensorUtils


@torch.no_grad()
def verify_action(model, dataset):
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
