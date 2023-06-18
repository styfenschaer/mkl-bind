import ctypes
import enum
from dataclasses import dataclass
from typing import Any, Iterable, overload

from numpy.typing import NDArray

class DFTI_CONFIG_PARAM(enum.IntEnum):
    DFTI_FORWARD_DOMAIN = 0
    DFTI_DIMENSION = 1
    DFTI_LENGTHS = 2
    DFTI_PRECISION = 3
    DFTI_FORWARD_SCALE = 4
    DFTI_BACKWARD_SCALE = 5
    DFTI_NUMBER_OF_TRANSFORMS = 7
    DFTI_COMPLEX_STORAGE = 8
    DFTI_REAL_STORAGE = 9
    DFTI_CONJUGATE_EVEN_STORAGE = 10
    DFTI_PLACEMENT = 11
    DFTI_INPUT_STRIDES = 12
    DFTI_OUTPUT_STRIDES = 13
    DFTI_INPUT_DISTANCE = 14
    DFTI_OUTPUT_DISTANCE = 15
    DFTI_WORKSPACE = 17
    DFTI_ORDERING = 18
    DFTI_TRANSPOSE = 19
    DFTI_DESCRIPTOR_NAME = 20
    DFTI_PACKED_FORMAT = 21
    DFTI_COMMIT_STATUS = 22
    DFTI_VERSION = 23
    DFTI_NUMBER_OF_USER_THREADS = 26
    DFTI_THREAD_LIMIT = 27
    DFTI_DESTROY_INPUT = 28
    DFTI_FWD_DISTANCE = 58
    DFTI_BWD_DISTANCE = 59


class DFTI_CONFIG_VALUE(enum.IntEnum):
    DFTI_COMMITTED = 30
    DFTI_UNCOMMITTED = 31
    DFTI_COMPLEX = 32
    DFTI_REAL = 33
    DFTI_SINGLE = 35
    DFTI_DOUBLE = 36
    DFTI_COMPLEX_COMPLEX = 39
    DFTI_COMPLEX_REAL = 40
    DFTI_REAL_COMPLEX = 41
    DFTI_REAL_REAL = 42
    DFTI_INPLACE = 43
    DFTI_NOT_INPLACE = 44
    DFTI_ORDERED = 48
    DFTI_BACKWARD_SCRAMBLED = 49
    DFTI_ALLOW = 51
    DFTI_AVOID = 52
    DFTI_NONE = 53
    DFTI_CCS_FORMAT = 54
    DFTI_PACK_FORMAT = 55
    DFTI_PERM_FORMAT = 56
    DFTI_CCE_FORMAT = 57


class DFTI_ERROR_CLASSES(enum.IntEnum):
    DFTI_NO_ERROR = 0
    DFTI_MEMORY_ERROR = 1
    DFTI_INVALID_CONFIGURATION = 2
    DFTI_INCONSISTENT_CONFIGURATION = 3
    DFTI_MULTITHREADED_ERROR = 4
    DFTI_BAD_DESCRIPTOR = 5
    DFTI_UNIMPLEMENTED = 6
    DFTI_MKL_INTERNAL_ERROR = 7
    DFTI_NUMBER_OF_THREADS_ERROR = 8
    DFTI_1D_LENGTH_EXCEEDS_INT32 = 9

@dataclass
class DFTI_DESCRIPTOR_HANDLE:
    """Wraps a pointer to the C structure of a descriptor."""

    pointer: ctypes.c_void_p

def DftiDescriptorNew() -> DFTI_DESCRIPTOR_HANDLE:
    """Allocates a new C structure for a descriptor."""

def DftiDescriptorDelete(handle: DFTI_DESCRIPTOR_HANDLE) -> None:
    """Frees the C structure of a descriptor."""

@overload
def DftiCreateDescriptor(
    handle: DFTI_DESCRIPTOR_HANDLE,
    precision: DFTI_CONFIG_VALUE,
    forward_domain: DFTI_CONFIG_VALUE,
    dimension: int,
    length: int,
) -> DFTI_ERROR_CLASSES:
    """Allocates the descriptor data structure and initializes it with default configuration values."""

@overload
def DftiCreateDescriptor(
    handle: DFTI_DESCRIPTOR_HANDLE,
    precision: DFTI_CONFIG_VALUE,
    forward_domain: DFTI_CONFIG_VALUE,
    dimension: int,
    length: Iterable[int],
) -> DFTI_ERROR_CLASSES:
    """Allocates the descriptor data structure and initializes it with default configuration values."""

def DftiCopyDescriptor(
    handle_original: DFTI_DESCRIPTOR_HANDLE,
    handle_copy: DFTI_DESCRIPTOR_HANDLE,
) -> DFTI_ERROR_CLASSES:
    """Makes a copy of an existing descriptor."""

def DftiFreeDescriptor(handle: DFTI_DESCRIPTOR_HANDLE) -> DFTI_ERROR_CLASSES:
    """Frees the memory allocated for a descriptor."""

@overload
def DftiSetValue(
    handle: DFTI_DESCRIPTOR_HANDLE,
    config_param: DFTI_CONFIG_PARAM,
    config_value: Any,
) -> DFTI_ERROR_CLASSES:
    """Sets one particular configuration parameter with the specified configuration value."""

@overload
def DftiSetValue(
    handle: DFTI_DESCRIPTOR_HANDLE,
    config_param: DFTI_CONFIG_PARAM,
    config_value: DFTI_CONFIG_VALUE,
) -> DFTI_ERROR_CLASSES:
    """Sets one particular configuration parameter with the specified configuration value."""

@overload
def DftiSetValue(
    handle: DFTI_DESCRIPTOR_HANDLE,
    config_param: DFTI_CONFIG_PARAM,
    config_value: Iterable[int],
) -> DFTI_ERROR_CLASSES:
    """Sets one particular configuration parameter with the specified configuration value."""

@overload
def DftiGetValue(
    handle: DFTI_DESCRIPTOR_HANDLE,
    config_param: DFTI_CONFIG_PARAM,
    config_value: Any,
) -> DFTI_ERROR_CLASSES:
    """Gets the configuration value of one particular configuration parameter."""

@overload
def DftiGetValue(
    handle: DFTI_DESCRIPTOR_HANDLE,
    config_param: DFTI_CONFIG_PARAM,
    config_value: ctypes.c_long,
) -> DFTI_ERROR_CLASSES:
    """Gets the configuration value of one particular configuration parameter."""

@overload
def DftiGetValue(
    handle: DFTI_DESCRIPTOR_HANDLE,
    config_param: DFTI_CONFIG_PARAM,
) -> tuple[DFTI_CONFIG_VALUE, DFTI_ERROR_CLASSES]:
    """Gets the configuration value of one particular configuration parameter."""

def DftiCommitDescriptor(handle: DFTI_DESCRIPTOR_HANDLE) -> DFTI_ERROR_CLASSES:
    """Performs all initialization for the actual FFT computation."""

@overload
def DftiComputeForward(
    handle: DFTI_DESCRIPTOR_HANDLE,
    x_inout: NDArray,
) -> DFTI_ERROR_CLASSES:
    """Computes the forward FFT."""

@overload
def DftiComputeForward(
    handle: DFTI_DESCRIPTOR_HANDLE,
    x_in: NDArray,
    y_out: NDArray,
) -> DFTI_ERROR_CLASSES:
    """Computes the forward FFT."""

@overload
def DftiComputeForward(
    handle: DFTI_DESCRIPTOR_HANDLE,
    xre_inout: NDArray,
    xim_inout: NDArray,
) -> DFTI_ERROR_CLASSES:
    """Computes the forward FFT."""

@overload
def DftiComputeForward(
    handle: DFTI_DESCRIPTOR_HANDLE,
    xre_in: NDArray,
    xim_in: NDArray,
    yre_out: NDArray,
    yim_out: NDArray,
) -> DFTI_ERROR_CLASSES:
    """Computes the forward FFT."""

@overload
def DftiComputeBackward(
    handle: DFTI_DESCRIPTOR_HANDLE,
    x_inout: NDArray,
) -> DFTI_ERROR_CLASSES:
    """Computes the backward FFT."""

@overload
def DftiComputeBackward(
    handle: DFTI_DESCRIPTOR_HANDLE,
    x_in: NDArray,
    y_out: NDArray,
) -> DFTI_ERROR_CLASSES:
    """Computes the backward FFT."""

@overload
def DftiComputeBackward(
    handle: DFTI_DESCRIPTOR_HANDLE,
    xre_inout: NDArray,
    xim_inout: NDArray,
) -> DFTI_ERROR_CLASSES:
    """Computes the backward FFT."""

@overload
def DftiComputeBackward(
    handle: DFTI_DESCRIPTOR_HANDLE,
    xre_in: NDArray,
    xim_in: NDArray,
    yre_out: NDArray,
    yim_out: NDArray,
) -> DFTI_ERROR_CLASSES:
    """Computes the backward FFT."""

def DftiErrorMessage(status: DFTI_ERROR_CLASSES) -> str:
    """Generates an error message."""

def DftiErrorClass(
    status: DFTI_ERROR_CLASSES,
    error_class: DFTI_ERROR_CLASSES,
) -> int:
    """Checks whether the status reflects an error of a predefined class."""
