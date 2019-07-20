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
for file_name in glob.glob(os.path.join(include_dir, '*.h')):
    _ffi.cdef(open(file_name).read())
_lib = _ffi.dlopen(lib_file)


class _ArrayWrapper:
    def __init__(self, obj, name=None, allow_multidim=False):
        self.arr = np.require(obj, dtype=np.float64,
                              requirements=['C_CONTIGUOUS'])
        self.scalar = self.arr.ndim == 0
        self.ndim = self.arr.ndim
        self.shape = self.arr.shape

        if (self.ndim > 1) and not allow_multidim:
            if name is not None:
                raise ValueError('{} cannot be >1 dim'.format(name))
            raise ValueError('array cannot be >1 dim')

    def cast(self):
        return _ffi.cast('double*', self.arr.ctypes.data)

    def finish(self):
        if self.scalar:
            return self.arr[()]
        return self.arr

    def __len__(self):
        if self.scalar:
            return 1
        return self.arr.size

    @classmethod
    def zeros_like(cls, obj):
        if isinstance(obj, _ArrayWrapper):
            return cls(np.zeros_like(obj.arr))
        return cls(np.zeros_like(obj))

    @classmethod
    def zeros(cls, shape):
        return cls(np.zeros(shape, dtype=np.double))

    @classmethod
    def ones_like(cls, obj):
        return cls(np.ones_like(obj))

    @classmethod
    def ones(cls, shape):
        return cls(np.ones(shape, dtype=np.double))

from . import averaging, bias, boostfactors, concentration, deltasigma, density, exclusion, massfunction, miscentering, peak_height, profile_derivatives, sigma_reconstruction, xi
