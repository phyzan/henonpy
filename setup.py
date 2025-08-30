from setuptools import setup, find_packages

setup(
    name="henonpy",
    version="1.0",
    python_requires=">=3.12, <=3.13",
    packages=find_packages(),
    package_data={
        "henonpy": ["*.so"],
    },
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib>=3.9.2",
        "scikit-image>=0.25.2",
        "numiphy>=1.0",
    ]
)