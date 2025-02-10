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
        
        import numiphy.symlib.expressions as sym
        import numiphy.odesolvers as ods
        
        x, y, eps, alpha, beta, gamma, omega_x, omega_y = sym.variables('x, y, eps, alpha, beta, gamma, omega_x, omega_y')
        V = (omega_x**2*x**2 + omega_y**2*y**2)/2 + eps*(x*y**2 + alpha*x**3 + beta*x**2*y + gamma*y**3)
        dyn = ods.HamiltonianSystem(V, x, y, args=(eps, alpha, beta, gamma, omega_x, omega_y))

        package_dir = self.build_lib
        target_dir = os.path.join(package_dir, "henonpy")
        dyn.ode.compile(target_dir, "henon", stack=True)

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
