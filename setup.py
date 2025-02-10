from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import subprocess
import sys


class CustomInstall(install):

    def run(self):
        try:
            import numiphy
        except ImportError:
            print("numiphy not found. Installing from GitHub...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "git+https://github.com/phyzan/numiphy.git"
            ])
        
        import numiphy.odesolvers as ods
        import tempfile

        package_dir = self.build_lib
        target_dir = os.path.join(package_dir, "henonpy")
        cwd = os.path.dirname(os.path.realpath(__file__))
        cpp_path = os.path.join(cwd, "henon.cpp")
        with open(cpp_path, 'r') as f:
            code = f.read()
        
        src = os.path.join(os.path.dirname(ods.__file__), 'odepack', 'pyode.hpp')
        code = code.replace("pyode.hpp", src)
        with tempfile.TemporaryDirectory() as temp_dir:
            cpp_path = os.path.join(temp_dir, "henon.cpp")
            with open(cpp_path, "w") as f:
                f.write(code)
            ods.compile(cpp_path, target_dir, "henon")
        super().run()


setup(
    name="henonpy",
    version="1.0",
    python_requires=">=3.12, <=3.13",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy==2.1.2",
        "scipy==1.14.1",
        "matplotlib==3.9.2",
        "pybind11==2.13.6",
        "joblib==1.4.2",
        "numiphy"
    ],
    cmdclass={"install": CustomInstall},
    zip_safe=False
)
