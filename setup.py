from setuptools import setup, find_packages

setup(
    name="path_planning",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "tqdm",
        "transformers",
        "numpy",
        "pandas",
    ],
    author="Fred Zhangzhi Peng",
    author_email="zp70@duke.edu",
    description="P2 (Path Planning) sampling implementation for sequence generation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pengzhangzhi/path_planning",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="diffusion, sequence generation, masked language model, path planning",
    python_requires=">=3.8",
) 