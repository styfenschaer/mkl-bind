import numpy as np

import mkl_bind
from mkl_bind import DFTI_CONFIG_VALUE

x_inout = np.random.rand(42) + np.random.rand(42) * 1j
x_npref = np.fft.fft(x_inout)

handle = mkl_bind.DftiDescriptorNew()
mkl_bind.DftiCreateDescriptor(
    handle,
    DFTI_CONFIG_VALUE.DFTI_DOUBLE,
    DFTI_CONFIG_VALUE.DFTI_COMPLEX,
    x_inout.ndim,
    x_inout.size,
)
mkl_bind.DftiCommitDescriptor(handle)
mkl_bind.DftiComputeForward(handle, x_inout)
mkl_bind.DftiFreeDescriptor(handle)
mkl_bind.DftiDescriptorDelete(handle)

assert np.allclose(x_npref, x_inout)