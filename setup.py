import setuptools
import sys


setuptools.setup(
    name="benepar",
    version="0.2.0a0",
    author="Nikita Kitaev",
    author_email="kitaev@cs.berkeley.edu",
    description="Berkeley Neural Parser",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nikitakit/self-attentive-parser",
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    python_requires=">=3.7",
    classifiers=(
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ),
    install_requires = ["nltk>=3.2", "torch>=1.6.0"],
    extras_require={
        "spacy": ["spacy>=2.0.9"],
    },
)
