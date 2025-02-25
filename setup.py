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
    author="Your Name",
    author_email="your.email@example.com",
    description="P2 (Path Planning) sampling implementation for sequence generation",
    # long_description=open("README.md").read(),
    # long_description_content_type="text/markdown",
    python_requires=">=3.8",
) 