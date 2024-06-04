#!/usr/bin/env python

import os
import time
import numpy as np
import cv2
import jax
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags
from functools import partial


from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.train_callbacks import supply_rng

from imitator.utils.env.robomimic_env import RoboMimicEnv
from imitator.utils.file_utils import (
    get_models_folder,
    get_data_folder,
    get_config_from_project_name,
)

from imitator.utils.env.gym_wrappers import (
    HistoryWrapper,
    TemporalEnsembleWrapper,
    ResizeImageWrapper,
    ProcessObsWrapper,
    NormalizeState,
)

initialize_compilation_cache()
# prevent tensorflow from using GPU memory since it's only used for data loading
tf.config.set_visible_devices([], "GPU")

# VARIABLES
PREDICTION_HORIZON = 4
WINDOW_SIZE = 2


FLAGS = flags.FLAGS
flags.DEFINE_string("project_name", None, "Project name")
flags.DEFINE_string("env_name", "Square", "Environment name")
flags.DEFINE_string("data_dir", None, "TFDS dataset file")
flags.DEFINE_string("model_dir", None, "Path to the model checkpoint")

# must be defined
flags.mark_flag_as_required("project_name")


def main(argv):
    ################################
    # Parse command-line arguments #
    ################################

    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    if FLAGS.data_dir is None:
        FLAGS.data_dir = os.path.join(
            get_data_folder(FLAGS.project_name), "imitator_dataset/1.0.0"
        )
    if FLAGS.model_dir is None:
        FLAGS.model_dir = os.path.join(get_models_folder(FLAGS.project_name), "models")

    cfg = get_config_from_project_name(FLAGS.project_name)
    obs_keys = list(cfg.obs.keys())
    img_obs_keys = [key for key in obs_keys if cfg.obs[key].modality == "ImageModality"]
    primary_img_key, wrist_img_key = None, None
    for img_obs_key in img_obs_keys:
        if cfg.obs[img_obs_key].camera == "primary":
            primary_img_key = img_obs_key
        elif cfg.obs[img_obs_key].camera == "wrist":
            wrist_img_key = img_obs_key
        else:
            raise ValueError("Invalid camera type")

    ################################
    # Load dataset for Goal Image  #
    ################################

    builder = tfds.builder_from_directory(
        builder_dir=FLAGS.data_dir,
    )
    ds = builder.as_dataset(split="train[:1]")  # only one episode

    # sample episode for goal image
    episode = next(iter(ds))
    steps = list(episode["steps"])
    if primary_img_key is not None:
        primary_goal_image = cv2.resize(
            np.array(steps[-1]["observation"][primary_img_key]),
            cfg.obs[primary_img_key].dim[:2],
        )  # last image is the goal image
    if wrist_img_key is not None:
        wrist_goal_image = cv2.resize(
            np.array(steps[-1]["observation"][wrist_img_key]),
            cfg.obs[wrist_img_key].dim[:2],
        )  # last image is the goal image

    ################################
    # Load model and create tasks  #
    ################################

    model = OctoModel.load_pretrained(FLAGS.model_dir)
    language_instruction = cfg.task.get("language_instruction", "Dummy Instruction")
    image_goals = {}  # set goals for images
    if primary_img_key is not None:
        image_goals["image_primary"] = primary_goal_image[None]
    if wrist_img_key is not None:
        image_goals["image_wrist"] = wrist_goal_image[None]
    task = model.create_tasks(texts=[language_instruction], goals=image_goals)
    policy_fn = supply_rng(
        partial(
            jax.jit(model.sample_actions),
            unnormalization_statistics=model.dataset_statistics["action"],
        ),
    )

    ############################
    #    Create environment    #
    ############################

    env = RoboMimicEnv(
        env_name=FLAGS.env_name,
        render_mode="human",
        **{
            "has_renderer": True,
        },
    )
    env = ProcessObsWrapper(
        env,
        flatten_keys=["robot0_joint_pos"],
        image_keys={
            "primary": primary_img_key,
            "wrist": wrist_img_key,
        },
    )  # process obs
    env = NormalizeState(env, model.dataset_statistics["state"])
    env = ResizeImageWrapper(
        env,
        resize_size={
            "primary": tuple(cfg.obs[primary_img_key].dim[:2]),
            "wrist": tuple(cfg.obs[wrist_img_key].dim[:2]),
        },
    )
    env = HistoryWrapper(env, horizon=WINDOW_SIZE)
    env = TemporalEnsembleWrapper(env, pred_horizon=PREDICTION_HORIZON, exp_weight=0.0)

    ##############################
    # Rollout in the environment #
    ##############################

    for _ in range(10):  # 10 episodes
        obs, _ = env.reset()
        for _ in range(200):  # 200 steps
            # model returns actions of shape [batch, pred_horizon, action_dim] -- remove batch
            start = time.time()
            actions = policy_fn(jax.tree_map(lambda x: x[None], obs), task)
            actions = actions[0]
            print(f"Time taken: {time.time() - start}")

            # step env -- info contains full "chunk" of observations for logging
            # obs only contains observation for final step of chunk
            obs, _, done, trunc, _ = env.step(actions)
            if done or trunc:
                continue

if __name__ == "__main__":
    app.run(main)
