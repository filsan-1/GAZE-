"""
Setup and installation script for GAZE Research Platform.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gaze-research",
    version="1.0.0",
    author="Research Team",
    description="Research-grade gaze-tracking application for ASD research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/filsan-1/GAZE-",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8+",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "scikit-learn>=1.3.0",
        "torch>=2.0.0",
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "Pillow>=10.0.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
    ],
)
