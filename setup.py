# --- Now proceed with the rest of setup.py ---
from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import tempfile

class CustomInstall(install):
    def run(self):
        # At this point, numiphy should already be importable.
        import numiphy.odesolvers as ods

        package_dir = self.build_lib
        target_dir = os.path.join(package_dir, "henonpy")
        cwd = os.path.dirname(os.path.realpath(__file__))
        cpp_path = os.path.join(cwd, "henon.cpp")
        with open(cpp_path, 'r') as f:
            code = f.read()
        
        src = os.path.join(os.path.dirname(ods.__file__), 'odepack', 'pyode.hpp')
        code = code.replace("pyode.hpp", src)
        with tempfile.TemporaryDirectory() as temp_dir:
            cpp_temp_path = os.path.join(temp_dir, "henon.cpp")
            with open(cpp_temp_path, "w") as f:
                f.write(code)
            ods.compile(cpp_temp_path, target_dir, "henon")
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
        "numiphy@git+https://github.com/phyzan/numiphy.git",
    ],
    cmdclass={"install": CustomInstall},
    zip_safe=False
)