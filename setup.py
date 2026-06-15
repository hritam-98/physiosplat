from setuptools import find_packages, setup

setup(
    name="physiosplat",
    version="1.0.0",
    description="Physics-Informed Dynamic Gaussian Splatting for Surgical Scene "
                "Reconstruction",
    author="Hritam Basak, Zhaozheng Yin",
    packages=find_packages(exclude=("tests", "examples", "scripts")),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.23",
        "scipy>=1.9",
        "imageio>=2.25",
        "pyyaml>=6.0",
        "tqdm>=4.64",
    ],
)
