import numpy as np

import mkl_bind
from mkl_bind import DFTI_CONFIG_PARAM, DFTI_CONFIG_VALUE


def check_status(status):
    print(mkl_bind.DftiErrorMessage(status))


x_in = np.random.rand(42) + np.random.rand(42) * 1j
y_out = np.empty_like(x_in)

handle = mkl_bind.DftiDescriptorNew()
status = mkl_bind.DftiCreateDescriptor(
    handle,
    DFTI_CONFIG_VALUE.DFTI_DOUBLE,
    DFTI_CONFIG_VALUE.DFTI_COMPLEX,
    x_in.ndim,
    x_in.size,
)
check_status(status)
status = mkl_bind.DftiSetValue(
    handle,
    DFTI_CONFIG_PARAM.DFTI_PLACEMENT,
    DFTI_CONFIG_VALUE.DFTI_NOT_INPLACE,
)
check_status(status)
status = mkl_bind.DftiCommitDescriptor(handle)
check_status(status)
status = mkl_bind.DftiComputeForward(handle, x_in, y_out)
check_status(status)
status = mkl_bind.DftiFreeDescriptor(handle)
check_status(status)
mkl_bind.DftiDescriptorDelete(handle)

assert np.allclose(y_out, np.fft.fft(x_in))
