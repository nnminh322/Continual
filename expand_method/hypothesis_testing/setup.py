from setuptools import setup, find_packages

setup(
    name="hypothesis_testing",
    version="0.1.0",
    description="SRT Hypothesis Testing Framework for SMoLoRA and HiDe-LLaVA",
    author="Research Project",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "Pillow>=9.0.0",
        "sentence-transformers>=2.2.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.3.0",
        "shortuuid>=1.0.0",
    ],
)
