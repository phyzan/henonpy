import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install


class CustomInstall(install):
    def run(self):

        from numiphy.toolkit.compile_tools import compile

        package_dir = self.build_lib
        target_dir = os.path.join(package_dir, "henonpy")
        cwd = os.path.dirname(os.path.realpath(__file__))
        cpp_path = os.path.join(cwd, "henon.cpp")
        compile(cpp_path, target_dir, "henon")
        super().run()


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
    cmdclass=dict(install=CustomInstall)
)