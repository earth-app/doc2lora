#!/usr/bin/env python3
"""Setup script for doc2lora library."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="doc2lora",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A library for fine-tuning LLMs using LoRA by using a folder of documents as input",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/earth-app/doc2lora",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13.0",
        "transformers>=4.21.0",
        "peft>=0.3.0",
        "datasets>=2.0.0",
        "PyPDF2>=3.0.0",
        "python-docx>=0.8.11",
        "beautifulsoup4>=4.11.0",
        "lxml>=4.9.0",
        "pyyaml>=6.0",
        "click>=8.0.0",
        "tqdm>=4.64.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
    ],
    entry_points={
        "console_scripts": [
            "doc2lora=doc2lora.cli:main",
        ],
    },
)
