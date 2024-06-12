from setuptools import setup, find_packages


def _post_install():
    import os

    PROJECT_ROOT = os.path.expanduser("~/.imitator")
    os.makedirs(PROJECT_ROOT, exist_ok=True)
    CACHE_DIR = os.path.expanduser("~/.cache/imitator")
    os.makedirs(CACHE_DIR, exist_ok=True)


setup(
    name="imitator",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.25.0",
        "scipy>=1.13.0",
        "opencv-python",
        "psutil",
        "tqdm",
        "imageio",
        "imageio-ffmpeg",
        "gymnasium",
        "absl-py",
        "Pillow",
        "ml-collections",
        "omegaconf",
        "moviepy",
        "plotly",
        "matplotlib",
        "h5py",
    ],
    extras_require={
        "cuda": [
            "wandb",
            "jax>=0.4.20",
            "flax>=0.7.5",
            "optax>=0.1.5",
            "orbax>=0.1.9",
            "distrax>=0.1.5",
            "chex>=0.1.85",
            "clu>=0.0.12",
            "tensorflow==2.15.0",
            "tensorflow_probability==0.23.0",
            "tensorflow_hub>=0.14.0",
            "tensorflow_text>=2.13.0",
            "tensorflow-datasets>=4.9.2",
            "tensorflow_graphics==2021.12.3",
            "transformers>=4.34.1",
            "huggingface_hub>=0.23.3",
            "einops>=0.6.1",
            "dlimp@git+https://github.com/kvablack/dlimp.git",
        ],
        "dev": [
            "pytest",
            "pytest-cov",
            "flake8",
            "black",
            "isort",
            "mypy",
            "pre-commit",
            "jupyter",
            "jupyterlab",
            "ipywidgets",
            "nbdev",
        ],
        "test": [
            "h5py",
        ]
    },
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3.8",
    description="imitator: Imitation learning library for robotics",
    author="Jihoon Oh",
    url="https://github.com/ojh6404/imitator",
    author_email="oh@jsk.imi.i.u-tokyo.ac.jp",
    version="0.0.4",
    entry_points={
        "console_scripts": [
            "imitator=imitator.utils.file_utils:main",
        ]
    },
)

_post_install()
