"""
A collection of utility functions for working with files, such as reading metadata from
demonstration datasets, loading model checkpoints, or downloading dataset files.
"""
import os
import re
from omegaconf import OmegaConf
import json

PROJECT_ROOT = os.path.expanduser("~/.imitator")


def extract_number(name):
    match = re.search(r"\d+", name)
    if match:
        return int(match.group())
    else:
        return 0


def sort_names_by_number(names):
    sorted_names = sorted(names, key=extract_number)
    return sorted_names


# function that create project folder in the root directory
def create_project_folder(project_name):
    project_dir = os.path.join(PROJECT_ROOT, project_name)
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
    return project_dir


# function that get project folder in the root directory
def get_project_folder(project_name):
    project_dir = os.path.join(PROJECT_ROOT, project_name)
    return project_dir


def get_log_folder(project_name):
    project_dir = get_project_folder(project_name)
    model_dir = os.path.join(project_dir, "runs")
    return model_dir


def get_models_folder(project_name):
    project_dir = get_project_folder(project_name)
    model_dir = os.path.join(project_dir, "models")
    return model_dir


def get_config_folder(project_name):
    project_dir = get_project_folder(project_name)
    config_dir = os.path.join(project_dir, "config")
    return config_dir


def get_config_file(project_name):
    config_dir = get_config_folder(project_name)
    config_file = os.path.join(config_dir, "config.yaml")
    return config_file


def get_config_from_project_name(project_name):
    config_file = get_config_file(project_name)
    config = OmegaConf.load(config_file)
    return config


def get_normalize_cfg(project_name):
    config_dir = get_config_folder(project_name)
    normalize_file = os.path.join(config_dir, "normalize.yaml")
    normalize_cfg = OmegaConf.load(normalize_file)
    return normalize_cfg

def update_normlize_cfg(project_name, config):
    normalize_cfg = get_normalize_cfg(project_name)
    if config.actions.normalize:
        config.actions.min = normalize_cfg.actions.min
        config.actions.max = normalize_cfg.actions.max
    for obs in normalize_cfg["obs"].keys():
        if config.obs[obs].normalize:
            config.obs[obs].min = normalize_cfg.obs[obs].min
            config.obs[obs].max = normalize_cfg.obs[obs].max
    return config

    

def get_latest_runs(project_name, model_type):
    model_dir = get_log_folder(project_name)
    model_type_dir = sort_names_by_number(
        [name for name in os.listdir(model_dir) if name.startswith(model_type)]
    )[-1]
    model_type_dir = os.path.join(model_dir, model_type_dir)
    return model_type_dir


def get_best_runs(project_name, model_type):
    model_type_dir = get_latest_runs(project_name, model_type)
    if os.path.exists(os.path.join(model_type_dir, model_type + "_model_best.pth")):
        model_file = os.path.join(model_type_dir, model_type + "_model_best.pth")
        print("Best model found: ", model_file)
        return model_file
    else:
        model_file = sort_names_by_number(
            [name for name in os.listdir(model_type_dir) if name.startswith(model_type)]
        )[-1]
        model_file = os.path.join(model_type_dir, model_file)
        print("Best model not found, use the latest model: ", model_file)
        return model_file

def print_config(config):
    dict_config = OmegaConf.to_container(config, resolve=True)
    print(json.dumps(dict_config, indent=4))
