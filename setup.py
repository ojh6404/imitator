from setuptools import setup, find_packages

# read the contents of your README file
# from os import path
# this_directory = path.abspath(path.dirname(__file__))
# with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
#     lines = f.readlines()

# remove images from README
# lines = [x for x in lines if (('.png' not in x) and ('.gif' not in x))]
# long_description = ''.join(lines)

setup(
    name="imitator",
    packages=[
        package for package in find_packages() if package.startswith("imitator")
    ],
    install_requires=[
        "numpy",
        "opencv-python",
        "h5py",
        "psutil",
        "tqdm",
        "termcolor",
        "tensorboard",
        "tensorboardX",
        "imageio",
        "imageio-ffmpeg",
        "matplotlib",
        "torch",
        "torchvision",
        "einops",
        "omegaconf",
    ],
    eager_resources=['*'],
    include_package_data=True,
    python_requires='>=3',
    description="imitator: Imitation learning library for robotics",
    author="Jihoon Oh",
    url="https://github.com/ojh6404/imitator",
    author_email="oh@jsk.imi.i.u-tokyo.ac.jp",
    version="0.0.2",
    # long_description=long_description,
    # long_description_content_type='text/markdown'
)
