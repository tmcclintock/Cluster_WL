import sys, os, glob
import setuptools
from setuptools import setup, Extension
import subprocess

os.system('ln -f -h -s ../include cluster_toolkit/include')

sources = glob.glob(os.path.join('src','*.c'))
headers = glob.glob(os.path.join('include','*.h'))
try:
    cflags = subprocess.check_output(['gsl-config', '--cflags'], universal_newlines=True).split()
    # TODO make this robust to compiler
    cflags.append('-std=c99')
    lflags = subprocess.check_output(['gsl-config', '--libs'], universal_newlines=True).split()
except OSError:
    raise Exception("Error: must have GSL installed and gsl-config working")

ext=Extension("cluster_toolkit._cluster_toolkit",
              sources,
              depends=headers,
              include_dirs=['include'],
              extra_compile_args=[os.path.expandvars(flag) for flag in cflags],
              extra_link_args=[os.path.expandvars(flag) for flag in lflags])

dist = setup(name="cluster_toolkit",
             author="Tom McClintock",
             author_email="mcclintock@bnl.gov",
             description="Modules for modeling galaxy clusters and related systematics.",
             license="MIT License",
             url="https://github.com/tmcclintock/cluster_toolkit",
             packages=['cluster_toolkit'],
             package_data={'cluster_toolkit' : headers },
             ext_modules=[ext],
             install_requires=['astropy', 'cffi', 'numpy'],
             setup_requires=['pytest_runner'],
             tests_require=['pytest']
)

#setup.py doesn't put the .so file in the cluster_toolkit directory,
#so this bit makes it possible to
#import cluster_toolkit from the root directory.
#Not really advisable, but everyone does it at some
#point, so might as well facilitate it.
build_lib = glob.glob(os.path.join('build','*','cluster_toolkit','_cluster_toolkit*.so'))
if len(build_lib) >= 1:
    lib = os.path.join('cluster_toolkit','_cluster_toolkit.so')
    if os.path.lexists(lib): os.unlink(lib)
    os.link(build_lib[0], lib)
