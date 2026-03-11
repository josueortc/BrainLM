from setuptools import find_packages, setup

setup(
    name="brainlm",
    version="0.1.0",
    description="BrainLM: A decoder-only foundation model for brain dynamics (fMRI)",
    author="BrainLM contributors",
    license="Apache-2.0",
    packages=find_packages(exclude=("tests", "scripts")),
    python_requires=">=3.8",
    install_requires=[
        "datasets>=2.0.0",
        "numpy",
        "torch",
        "transformers>=4.28.0",
        "scikit-learn",
        "matplotlib",
        "pyarrow",
    ],
    extras_require={
        "wandb": ["wandb"],
        "dev": ["pytest"],
    },
)
