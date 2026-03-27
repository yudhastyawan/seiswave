import os
import subprocess
from setuptools import setup, find_packages, Distribution
from setuptools.command.build_ext import build_ext

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

class FortranBuildExt(build_ext):
    def run(self):
        fortran_dir = os.path.join(os.path.dirname(__file__), 'seiswave', 'src_fortran')
        if os.path.exists(fortran_dir):
            print("Compiling Fortran extension using f2py...")
            subprocess.run(["python", "build.py"], cwd=fortran_dir, check=True)
            
            # The .so/.pyd file is generated in src_fortran/build. 
            # It needs to be moved/copied to the seiswave package root so it can be imported.
            src_build_dir = os.path.join(fortran_dir, 'build')
            import glob
            import shutil
            
            extensions = glob.glob(os.path.join(src_build_dir, 'cps_core*.so')) + \
                         glob.glob(os.path.join(src_build_dir, 'cps_core*.pyd'))
                         
            target_dir = os.path.join(os.path.dirname(__file__), 'seiswave')
            for ext in extensions:
                print(f"Copying {ext} to {target_dir}")
                shutil.copy(ext, target_dir)
                
        super().run()

setup(
    packages=find_packages(include=['seiswave', 'seiswave.*']),
    include_package_data=True,
    cmdclass={
        'build_ext': FortranBuildExt,
    },
    distclass=BinaryDistribution,
    # Ensure any built .so or .pyd files are included in the wheel package
    package_data={
        'seiswave': ['*.so', '*.pyd', 'src_fortran/*']
    }
)
