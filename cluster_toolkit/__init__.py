"""
cluster_toolkit is a module for computing galaxy cluster models.
"""

import cffi
import glob
import os
import numpy as np

__author__ = "Tom McClintock <mcclintock@bnl.gov>"

cluster_toolkit_dir = os.path.dirname(__file__)
include_dir = os.path.join(cluster_toolkit_dir,'include')
lib_file = os.path.join(cluster_toolkit_dir,'_cluster_toolkit.so')
# Some installation (e.g. Travis with python 3.x)
# name this e.g. _cluster_toolkit.cpython-34m.so,
# so if the normal name doesn't exist, look for something else.
# Note: we ignore this if we are building the docs on RTD
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if not os.path.exists(lib_file) and not on_rtd:
    alt_files = glob.glob(os.path.join(os.path.dirname(__file__),'_cluster_toolkit*.so'))
    if len(alt_files) == 0:
        raise IOError("No file '_cluster_toolkit.so' found in %s"%cluster_toolkit_dir)
    if len(alt_files) > 1:
        raise IOError("Multiple files '_cluster_toolkit*.so' found in %s: %s"%(cluster_toolkit_dir,alt_files))
    lib_file = alt_files[0]

_ffi = cffi.FFI()
for file_name in glob.glob(os.path.join(include_dir,'*.h')):
    _ffi.cdef(open(file_name).read())
_lib = _ffi.dlopen(lib_file)
_lib.gsl_set_error_handler_off()

def _dcast(x):
    if isinstance(x, list): x = np.asarray(x, dtype=np.float64, order='C')
    return _ffi.cast('double*', x.ctypes.data)

from . import (averaging, bias, boostfactors, concentration, deltasigma,
               density, exclusion, massfunction, miscentering, peak_height,
               pressure, profile_derivatives, sigma_reconstruction, xi)
