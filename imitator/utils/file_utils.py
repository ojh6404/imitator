import os
import re
from omegaconf import OmegaConf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
BASE_CONFIG_ROOT = os.path.join(PROJECT_ROOT, "imitator", "cfg")
DATASET_ROOT = os.path.join(PROJECT_ROOT, "imitator", "datasets")
RUNS_ROOT = os.path.join(PROJECT_ROOT, "imitator", "runs")

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
def create_project_cfg_dir(project_name):
    project_dir = os.path.join(BASE_CONFIG_ROOT, project_name)
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
    return project_dir

# function that get project folder in the root directory
def get_project_cfg_folder(project_name):
    project_dir = os.path.join(BASE_CONFIG_ROOT, project_name)
    return project_dir

def get_runs_folder(project_name):
    runs_dir = os.path.join(RUNS_ROOT, project_name)
    return runs_dir

def get_cfg(project_name):
    config_dir = get_project_cfg_folder(project_name)
    config_file = os.path.join(config_dir, "config.yaml")
    config = OmegaConf.load(config_file)
    return config

def get_normalize_cfg(project_name):
    config_dir = get_project_cfg_folder(project_name)
    normalize_file = os.path.join(config_dir, "normalize.yaml")
    normalize_cfg = OmegaConf.load(normalize_file)
    return normalize_cfg

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
