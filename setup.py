from setuptools import setup, find_packages

setup(
    name="cluster-lensing-cov",              # Replace with your package's name
    version="0.1.0",                       # Start with a semantic version
    author="Heidi Wu",
    author_email="hywu@smu.edu",
    description="This package computes the cluster lensing signal and covariance matrices. Both in terms of gammaT and DeltaSigma",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/hywu/cluster-lensing-cov",  # Replace with your URL
    packages=find_packages(),              # Automatically find your package
    install_requires=[                     # List your dependencies here
                      "numpy",
                      "scipy",
                      "astropy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Or your chosen license
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',               # Specify the Python version you support
)