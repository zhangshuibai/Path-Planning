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
) 