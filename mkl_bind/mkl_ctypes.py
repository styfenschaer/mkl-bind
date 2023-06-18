import ctypes
import enum
from distutils.sysconfig import get_config_var
from pathlib import Path


def get_extension_path(lib_name):
    search_path = Path(__file__).parent.parent
    ext_suffix = get_config_var("EXT_SUFFIX")
    ext_path = f"**/{lib_name}{ext_suffix}"
    matches = search_path.glob(ext_path)
    return str(next(matches))


lib = ctypes.CDLL(get_extension_path("_mkl_bind"))


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


sizeof_mkl_long = lib.sizeof_mkl_long
sizeof_mkl_long.restype = ctypes.c_int
sizeof_mkl_long.argtypes = []

ctypes_mkl_long = {
    2: ctypes.c_int16,
    4: ctypes.c_int32,
    8: ctypes.c_int64,
}[sizeof_mkl_long()]


class types:
    MKL_LONG = ctypes_mkl_long
    MKL_LONG_OR_MKL_LONG_ARRAY = ctypes.c_void_p
    DFTI_CONFIG_VALUE = ctypes.c_int
    DFTI_CONFIG_PARAM = ctypes.c_int
    DFTI_DESCRIPTOR_HANDLE_POINTER = ctypes.c_void_p
    AT_LEAST_ONE_ARRAY = ctypes.c_void_p
    ANY = ctypes.c_void_p


DftiDescriptorNew = lib.Py_DftiDescriptorNew
DftiDescriptorNew.restype = types.DFTI_DESCRIPTOR_HANDLE_POINTER
DftiDescriptorNew.argtypes = []

DftiDescriptorDelete = lib.Py_DftiDescriptorDelete
DftiDescriptorDelete.restype = None
DftiDescriptorDelete.argtypes = [
    types.DFTI_DESCRIPTOR_HANDLE_POINTER,
]

DftiCreateDescriptor = lib.Py_DftiCreateDescriptor
DftiCreateDescriptor.restype = types.MKL_LONG
DftiCreateDescriptor.argtypes = [
    types.DFTI_DESCRIPTOR_HANDLE_POINTER,
    types.DFTI_CONFIG_VALUE,
    types.DFTI_CONFIG_VALUE,
    types.MKL_LONG,
    types.MKL_LONG_OR_MKL_LONG_ARRAY,
]

DftiCopyDescriptor = lib.Py_DftiCopyDescriptor
DftiCopyDescriptor.restype = types.MKL_LONG
DftiCopyDescriptor.argtypes = [
    types.DFTI_DESCRIPTOR_HANDLE_POINTER,
    types.DFTI_DESCRIPTOR_HANDLE_POINTER,
]

DftiFreeDescriptor = lib.Py_DftiFreeDescriptor
DftiFreeDescriptor.restype = types.MKL_LONG
DftiFreeDescriptor.argtypes = [
    types.DFTI_DESCRIPTOR_HANDLE_POINTER,
]

DftiSetValue = lib.Py_DftiSetValue
DftiSetValue.restype = types.MKL_LONG
DftiSetValue.argtypes = [
    types.DFTI_DESCRIPTOR_HANDLE_POINTER,
    types.DFTI_CONFIG_PARAM,
    types.ANY,
]

DftiGetValue = lib.Py_DftiGetValue
DftiGetValue.restype = types.MKL_LONG
DftiGetValue.argtypes = [
    types.DFTI_DESCRIPTOR_HANDLE_POINTER,
    types.DFTI_CONFIG_PARAM,
    types.ANY,
]

DftiCommitDescriptor = lib.Py_DftiCommitDescriptor
DftiCommitDescriptor.restype = types.MKL_LONG
DftiCommitDescriptor.argtypes = [
    types.DFTI_DESCRIPTOR_HANDLE_POINTER,
]

DftiComputeForward = lib.Py_DftiComputeForward
DftiComputeForward.restype = types.MKL_LONG
DftiComputeForward.argtypes = [
    types.DFTI_DESCRIPTOR_HANDLE_POINTER,
    types.AT_LEAST_ONE_ARRAY,
]

DftiComputeBackward = lib.Py_DftiComputeBackward
DftiComputeBackward.restype = types.MKL_LONG
DftiComputeBackward.argtypes = [
    types.DFTI_DESCRIPTOR_HANDLE_POINTER,
    types.AT_LEAST_ONE_ARRAY,
]

DftiErrorMessage = lib.Py_DftiErrorMessage
DftiErrorMessage.restype = ctypes.c_char_p
DftiErrorMessage.argtypes = [
    types.MKL_LONG,
]

DftiErrorClass = lib.Py_DftiErrorClass
DftiErrorClass.restype = types.MKL_LONG
DftiErrorClass.argtypes = [
    types.MKL_LONG,
    types.MKL_LONG,
]
