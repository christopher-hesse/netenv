from setuptools import setup

setup(
    name="netenv",
    version="0.0.2",
    packages=["netenv"],
    install_requires=["gym~=0.10", "numpy~=1.14"],
    extras_require={"dev": ["pytest", "pytest-benchmark"]},
)
