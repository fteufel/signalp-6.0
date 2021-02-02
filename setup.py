import os
import re
from setuptools import setup, find_packages

here = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(here, "README.md")) as f:
    readme = f.read()
with open(os.path.join(here, "src", "signalp6", "__init__.py")) as f:
    version = re.search(r'__version__ = (["\'])([^"\']*)\1', f.read())[2]

setup(
    name="signalp6",
    version=version,
    description="SignalP 6.0 signal peptide prediction tool",
    long_description=readme,
    url="healthtech.dtu.dk",
    author="Felix Teufel",
    author_email="felix.teufel@gmail.com",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    keywords="torch",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6, <4",
)
