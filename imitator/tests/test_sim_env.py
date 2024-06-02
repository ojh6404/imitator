#!/usr/bin/env python

import gymnasium as gym
from imitator.utils.env.robomimic_env import RoboMimicEnv

if __name__ == "__main__":
    # env = gym.make("RoboMimic-v0", env_name="Lift", render_mode="human", **kwargs)
    env = RoboMimicEnv(
        env_name="Lift",
        render_mode="human",
        **{
            "has_renderer": True,
        },
    )

    obs, info = env.reset()

    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        print(obs.keys())
        if done:
            obs, info = env.reset()
