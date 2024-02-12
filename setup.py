from setuptools import find_packages
from setuptools import setup

setup(
    name="zoo",
    version="0.0.0",
    description="Library for training and comparing reinforcement learning and active inference agents.",
    author="Theophile Champion and Lisa Bonheme",
    author_email="tmac3@kent.ac.uk",
    url="https://github.com/ChampiB/Deconstructing_Deep_Active_Inference/",
    license="Apache 2.0",
    packages=find_packages(),
    scripts=["scripts/run_task"],
    include_package_data=True,
    install_requires=[
        "gym>=0.21.0",
        "scikit-image",
        "numpy==1.23.4",
        "omegaconf==2.3.0",
        "torchvision==0.15.1",
        "hydra-core==1.3.2",
        "stable-baselines3==1.6.2",
        "tensorboard==2.12.0",
        "prettytable==3.7.0",
        "paramiko==3.1.0",
        "pydot",
        "bigtree",
        "PyYAML==5.4.1",
        "docutils==0.19",
        "matplotlib==3.6.2",
        "Pillow==9.5.0",
        "pytest==7.3.1",
        "networkx==2.6.2",
        "pandas==1.3.5",
        "imageio==2.9.0",
        "tabulate==0.8.9",
        "karateclub==1.3.0",
        "platformdirs==2.5.2",
        "seaborn==0.12.1",
        "scipy==1.9.2",
        "ray[tune]>=2.4.0",
        "fsspec==2023.4.0",
        "ncps==0.0.7"
    ],
    python_requires="~=3.9",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    keywords="pytorch, machine learning, reinforcement learning, deep learning, active inference"
)
