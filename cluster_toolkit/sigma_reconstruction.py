"""Galaxy cluster Sigma(R) reconstruction profiles, also known as 'Y' profiles.
"""

import cluster_toolkit
from cluster_toolkit import _ArrayWrapper
import numpy as np

def Sigma_REC_from_DeltaSigma(R, DeltaSigma):
    """Reconstructed Sigma(R) profile, also known as 'Y'
    in the same units as DeltaSigma.

    Note: R and DeltaSigma must have the same shape,
    and have more than 1 element. The returned array
    has one fewer elements.

    Note: R must have constant logarithmic spacing.

    Args:
        R (array like): Projected radii.
        DeltaSigma (array like): Differential surface mass density.

    Returns:
        Reconstructed surface mass density.
    """
    R = _ArrayWrapper(R, allow_multidim=True)
    DeltaSigma = _ArrayWrapper(DeltaSigma, allow_multidim=True)
    if R.shape != DeltaSigma.shape:
        raise Exception("R and DeltaSigma must have the same shape.")

    lnR = np.log(R)
    for i in range(len(lnR)-2):
        dlnR0 = lnR[i] - lnR[i+1]
        dlnR1 = lnR[i+1] - lnR[i+2]
        if not (np.fabs(dlnR0 - dlnR1) < 1e-6):
            raise Exception("R must have constant logarithmic spacing.")
        continue

    #Note the order here, we integrate DOWNWARD
    dlnR = lnR[0] - lnR[1]

    Sigma = _ArrayWrapper.zeros(len(R)-1)
    cluster_toolkit._lib.Sigma_REC_from_DeltaSigma(dlnR, DeltaSigma.cast(),
                                                   len(R), Sigma.cast())
    return Sigma.finish()
