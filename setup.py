from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="signalGenerator",
    version="0.1.0",
    description="Implementation of a signal generator with different test signals.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xaviliz/signalGenerator",
    author="Xavier Lizarraga-Seijas",
    author_email="xlizarraga@gmail.com",
    packages=["signal_generator"],
    install_requires=["scipy>=1.0.1", "numpy>=1.14.2"],
    classifiers=(
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
