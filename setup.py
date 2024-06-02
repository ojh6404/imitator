from setuptools import setup

setup(
    name="imitator",
    packages=["imitator"],
    install_requires=[
        "numpy",
        "scipy",
        "pyquaternion",
        "opencv-python",
        "h5py",
        "psutil",
        "tqdm",
        "termcolor",
        "imageio",
        "imageio-ffmpeg",
        "gymnasium",
        "absl-py",
        "Pillow",
        "ml-collections",
        "omegaconf",
        "moviepy",
        "plotly",
    ],
    extras_require={
        "cuda": [
            "tensorboard",
            "tensorboardX",
            "wandb",
            "dlimp@git+https://github.com/ojh6404/dlimp.git",
            "octo@git+https://github.com/ojh6404/octo.git",
            "matplotlib",
            "jax",
            "flax",
            "optax",
            "orbax",
            "distrax",
            "chex",
            "clu",
            "tensorflow-datasets",
            "tensorflow[and-cuda]",
            "tensorflow_hub",
            "tensorflow_text",
            "huggingface_hub",
            "transformers",
            "einops",
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
        "sim": ["robosuite", "minari", "gymnasium-robotics"],
    },
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3.8",
    description="imitator: Imitation learning library for robotics",
    author="Jihoon Oh",
    url="https://github.com/ojh6404/imitator",
    author_email="oh@jsk.imi.i.u-tokyo.ac.jp",
    version="0.0.4",
)
