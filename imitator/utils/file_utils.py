"""
A collection of utility functions for working with files, such as reading metadata from
demonstration datasets, loading model checkpoints, or downloading dataset files.
"""
import os
import h5py
import re

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

def get_config_file(project_name):
    project_dir = get_project_folder(project_name)
    config_file = os.path.join(project_dir, "config/config.yaml")
    return config_file

def get_model_folder(project_name):
    project_dir = get_project_folder(project_name)
    model_dir = os.path.join(project_dir, "models")
    return model_dir

def get_config_folder(project_name):
    project_dir = get_project_folder(project_name)
    config_dir = os.path.join(project_dir, "config")
    return config_dir
