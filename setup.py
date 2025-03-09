# --- Now proceed with the rest of setup.py ---
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
        "numpy==2.1.2",
        "scipy==1.14.1",
        "matplotlib==3.9.2",
        "pybind11==2.13.6",
        "joblib==1.4.2",
        "scikit-image>=0.25.2",
        "numiphy@git+https://github.com/phyzan/numiphy.git",
    ],
    zip_safe=False
)