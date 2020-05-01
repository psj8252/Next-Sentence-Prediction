from setuptools import find_packages, setup

setup(
    name="nsp-model",
    version="0.0.1",
    description="This repository is for developing next sentence prediction module.",
    install_requires=["torch", "torchtext", "konlpy", "dill"],
    url="https://github.com/psj8252/Next-Sentence-Prediction.git",
    author="Park Sangjun",
    packages=find_packages(exclude=["samples"]),
)
