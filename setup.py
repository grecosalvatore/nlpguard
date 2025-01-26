from setuptools import setup, find_packages
from pathlib import Path

here = Path(__file__).parent

# Read the contents of your README file
with open(here / 'README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="nlpguard",
    version="0.0.1",
    description="NLPGuard: a Framework for Mitigating the use of Protected Attributes by NLP Classifiers",
    long_description=long_description,  # Use the variable here
    long_description_content_type="text/markdown",
    packages=find_packages(),  # Let find_packages() search the entire project directory
    include_package_data=True,  # This ensures MANIFEST.in is respected
    author="Salvatore Greco",
    author_email="grecosalvatore94@gmail.com",
    url="https://github.com/grecosalvatore/nlpguard",
    install_requires=[
        "numpy>=1.22.4",
        "scikit-learn>=0.24.2",
        "matplotlib~=3.5.1",
        "scipy>=1.10.0",
        "tqdm~=4.64.1",
        "setuptools~=58.0.4",
        "openai~=1.3.9",
        "pandas~=2.1.4",
        "torch~=2.1.2",
        "transformers~=4.48.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
