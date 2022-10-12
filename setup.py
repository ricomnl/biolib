"""setup.py"""
from setuptools import find_packages, setup

setup(
    name="biolib",
    version="0.0.1",
    packages=find_packages(),
    author="Rico Meinl",
    author_email="dev@rmeinl.com",
    description="Useful bio funcs",
    url="https://github.com/ricomnl/biolib/",
    include_package_data=True,
    python_requires=">=3.7",
)
