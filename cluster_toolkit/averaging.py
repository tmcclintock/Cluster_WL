"""Averaging projected cluster profiles.

"""
import cluster_toolkit
from cluster_toolkit import _ArrayWrapper, _handle_gsl_error
import numpy as np


def average_profile_in_bins(Redges, R, prof):
    """Average profile in bins.

    Calculates the average of some projected profile in a
    radial bins in Mpc/h comoving.

    Args:
        Redges (array like): Array of radial bin edges.
        R (array like): Radii of the profile.
        prof (array like): Projected profile.

    Returns:
        numpy.array: Average profile in bins between the edges provided.

    """
    Redges = _ArrayWrapper(Redges)
    R = _ArrayWrapper(R)
    prof = _ArrayWrapper(prof)

    if Redges.ndim == 0:
        raise Exception("Must supply a left and right edge.")
    if Redges.ndim > 1:
        raise Exception("Redges cannot be a >1D array.")
    if np.min(Redges.arr) < np.min(R.arr):
        raise Exception("Minimum edge must be >= minimum R")
    if np.max(Redges.arr) > np.max(R.arr):
        raise Exception("Maximum edge must be <= maximum R")

    ave_prof = _ArrayWrapper(np.zeros(len(Redges) - 1))
    r = cluster_toolkit._lib.average_profile_in_bins(Redges.cast(), len(Redges),
                                                     R.cast(), len(R),
                                                     prof.cast(),
                                                     ave_prof.cast())

    _handle_gsl_error(r, average_profile_in_bins)

    return ave_prof.finish()


def average_profile_in_bin(Rlow, Rhigh, R, prof):
    """Average profile in a bin.

    Calculates the average of some projected profile in a
    radial bin in Mpc/h comoving.

    Args:
        Rlow (float): Inner radii.
        Rhigh (float): Outer radii.
        R (array like): Radii of the profile.
        prof (array like): Projected profile.

    Returns:
        float: Average profile in the radial bin, or annulus.

    """
    return np.squeeze(average_profile_in_bins([Rlow, Rhigh], R, prof))
