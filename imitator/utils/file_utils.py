"""
A collection of utility functions for working with files, such as reading metadata from
demonstration datasets, loading model checkpoints, or downloading dataset files.
"""

import os
import re
from omegaconf import OmegaConf
import json
import shutil

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


def get_models_folder(project_name):
    project_dir = get_project_folder(project_name)
    model_dir = os.path.join(project_dir, "models")
    return model_dir


def get_config_folder(project_name):
    project_dir = get_project_folder(project_name)
    config_dir = os.path.join(project_dir, "config")
    return config_dir


def get_data_folder(project_name):
    project_dir = get_project_folder(project_name)
    data_dir = os.path.join(project_dir, "data")
    return data_dir


def get_config_file(project_name):
    config_dir = get_config_folder(project_name)
    config_file = os.path.join(config_dir, "config.yaml")
    return config_file


def get_config_from_project_name(project_name):
    config_file = get_config_file(project_name)
    config = OmegaConf.load(config_file)
    return config


def pprint_config(config):
    dict_config = OmegaConf.to_container(config, resolve=True)
    print(json.dumps(dict_config, indent=4))


def main():
    import sys

    command = sys.argv[1]
    assert command in ["init", "run"], "First argument must be 'init' or 'run'"

    def init_project(project_name):
        try:
            project_path = create_project_folder(project_name)
            config_folder = get_config_folder(project_name)
            os.makedirs(config_folder, exist_ok=True)

            # create config dir and
            default_config = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "config", "default.yaml"
            )
            config_file = get_config_file(project_name)
            shutil.copy(default_config, config_file)

            # create data dir
            data_folder = get_data_folder(project_name)
            os.makedirs(data_folder, exist_ok=True)
            print(f"Project '{project_name}' created at {project_path}")
        except OSError as e:
            print(f"Error: {e}")
            sys.exit(1)

    if command == "init":
        project_name = sys.argv[2]
        init_project(project_name)
    elif command == "run":
        pass  # TODO: implement run command
