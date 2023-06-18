import ctypes
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray
from plum import dispatch

from . import mkl_ctypes
from .mkl_ctypes import (DFTI_CONFIG_PARAM, DFTI_CONFIG_VALUE,
                         DFTI_ERROR_CLASSES)


@dataclass
class DFTI_DESCRIPTOR_HANDLE:
    pointer: ctypes.c_void_p


def DftiDescriptorNew() -> DFTI_DESCRIPTOR_HANDLE:
    handle = mkl_ctypes.DftiDescriptorNew()
    return DFTI_DESCRIPTOR_HANDLE(handle)


def DftiDescriptorDelete(handle: DFTI_DESCRIPTOR_HANDLE) -> None:
    mkl_ctypes.DftiDescriptorDelete(handle.pointer)


@dispatch
def DftiCreateDescriptor(
    handle: DFTI_DESCRIPTOR_HANDLE,
    precision: DFTI_CONFIG_VALUE,
    forward_domain: DFTI_CONFIG_VALUE,
    dimension: int,
    length: int,
) -> DFTI_ERROR_CLASSES:
    status = mkl_ctypes.DftiCreateDescriptor(
        handle.pointer,
        precision,
        forward_domain,
        dimension,
        length,
    )
    return DFTI_ERROR_CLASSES(status)


@dispatch
def DftiCreateDescriptor(
    handle: DFTI_DESCRIPTOR_HANDLE,
    precision: DFTI_CONFIG_VALUE,
    forward_domain: DFTI_CONFIG_VALUE,
    dimension: int,
    length: Iterable[int],
) -> DFTI_ERROR_CLASSES:
    status = mkl_ctypes.DftiCreateDescriptor(
        handle.pointer,
        precision,
        forward_domain,
        dimension,
        np.asarray(
            length,
            dtype=np.int64,
        ).ctypes.data,
    )
    return DFTI_ERROR_CLASSES(status)


def DftiCopyDescriptor(
    handle_original: DFTI_DESCRIPTOR_HANDLE,
    handle_copy: DFTI_DESCRIPTOR_HANDLE,
) -> DFTI_ERROR_CLASSES:
    status = mkl_ctypes.DftiCopyDescriptor(
        handle_original.pointer,
        handle_copy.pointer,
    )
    return DFTI_ERROR_CLASSES(status)


def DftiFreeDescriptor(handle: DFTI_DESCRIPTOR_HANDLE) -> DFTI_ERROR_CLASSES:
    status = mkl_ctypes.DftiFreeDescriptor(handle.pointer)
    return DFTI_ERROR_CLASSES(status)


@dispatch
def DftiSetValue(
    handle: DFTI_DESCRIPTOR_HANDLE,
    config_param: DFTI_CONFIG_PARAM,
    config_value: Any,
) -> DFTI_ERROR_CLASSES:
    status = mkl_ctypes.DftiSetValue(
        handle.pointer,
        config_param,
        config_value,
    )
    return DFTI_ERROR_CLASSES(status)


@dispatch
def DftiSetValue(
    handle: DFTI_DESCRIPTOR_HANDLE,
    config_param: DFTI_CONFIG_PARAM,
    config_value: Iterable[int],
) -> DFTI_ERROR_CLASSES:
    status = mkl_ctypes.DftiSetValue(
        handle.pointer,
        config_param,
        np.asarray(
            config_value,
            dtype=np.int64,
        ).ctypes.data,
    )
    return DFTI_ERROR_CLASSES(status)


@dispatch
def DftiGetValue(
    handle: DFTI_DESCRIPTOR_HANDLE,
    config_param: DFTI_CONFIG_PARAM,
    config_value: Any,
) -> DFTI_ERROR_CLASSES:
    status = mkl_ctypes.DftiGetValue(
        handle.pointer,
        config_param,
        config_value,
    )
    return DFTI_ERROR_CLASSES(status)


@dispatch
def DftiGetValue(
    handle: DFTI_DESCRIPTOR_HANDLE,
    config_param: DFTI_CONFIG_PARAM,
    config_value: ctypes.c_long,
) -> DFTI_ERROR_CLASSES:
    status = mkl_ctypes.DftiGetValue(
        handle.pointer,
        config_param,
        ctypes.byref(config_value),
    )
    return DFTI_ERROR_CLASSES(status)


@dispatch
def DftiGetValue(
    handle: DFTI_DESCRIPTOR_HANDLE,
    config_param: DFTI_CONFIG_PARAM,
) -> tuple[int, DFTI_ERROR_CLASSES]:
    config_value = ctypes.c_long(0)
    status = mkl_ctypes.DftiGetValue(
        handle.pointer,
        config_param,
        ctypes.byref(config_value),
    )
    return (
        config_value.value,
        DFTI_ERROR_CLASSES(status),
    )


def DftiCommitDescriptor(handle: DFTI_DESCRIPTOR_HANDLE) -> DFTI_ERROR_CLASSES:
    status = mkl_ctypes.DftiCommitDescriptor(handle.pointer)
    return DFTI_ERROR_CLASSES(status)


@dispatch
def DftiComputeForward(
    handle: DFTI_DESCRIPTOR_HANDLE,
    x_inout: NDArray,
) -> DFTI_ERROR_CLASSES:
    status = mkl_ctypes.DftiComputeForward(
        handle.pointer,
        x_inout.ctypes.data_as(ctypes.c_void_p),
    )
    return DFTI_ERROR_CLASSES(status)


@dispatch
def DftiComputeForward(
    handle: DFTI_DESCRIPTOR_HANDLE,
    x_in: NDArray,
    y_out: NDArray,
) -> DFTI_ERROR_CLASSES:
    status = mkl_ctypes.DftiComputeForward(
        handle.pointer,
        x_in.ctypes.data_as(ctypes.c_void_p),
        y_out.ctypes.data_as(ctypes.c_void_p),
    )
    return DFTI_ERROR_CLASSES(status)


@dispatch
def DftiComputeForward(
    handle: DFTI_DESCRIPTOR_HANDLE,
    xre_inout: NDArray,
    xim_inout: NDArray,
) -> DFTI_ERROR_CLASSES:
    status = mkl_ctypes.DftiComputeForward(
        handle.pointer,
        xre_inout.ctypes.data_as(ctypes.c_void_p),
        xim_inout.ctypes.data_as(ctypes.c_void_p),
    )
    return DFTI_ERROR_CLASSES(status)


@dispatch
def DftiComputeForward(
    handle: DFTI_DESCRIPTOR_HANDLE,
    xre_in: NDArray,
    xim_in: NDArray,
    yre_out: NDArray,
    yim_out: NDArray,
) -> DFTI_ERROR_CLASSES:
    status = mkl_ctypes.DftiComputeForward(
        handle.pointer,
        xre_in.ctypes.data_as(ctypes.c_void_p),
        xim_in.ctypes.data_as(ctypes.c_void_p),
        yre_out.ctypes.data_as(ctypes.c_void_p),
        yim_out.ctypes.data_as(ctypes.c_void_p),
    )
    return DFTI_ERROR_CLASSES(status)


@dispatch
def DftiComputeBackward(
    handle: DFTI_DESCRIPTOR_HANDLE,
    x_inout: NDArray,
) -> DFTI_ERROR_CLASSES:
    status = mkl_ctypes.DftiComputeBackward(
        handle.pointer,
        x_inout.ctypes.data_as(ctypes.c_void_p),
    )
    return DFTI_ERROR_CLASSES(status)


@dispatch
def DftiComputeBackward(
    handle: DFTI_DESCRIPTOR_HANDLE,
    x_in: NDArray,
    y_out: NDArray,
) -> DFTI_ERROR_CLASSES:
    status = mkl_ctypes.DftiComputeBackward(
        handle.pointer,
        x_in.ctypes.data_as(ctypes.c_void_p),
        y_out.ctypes.data_as(ctypes.c_void_p),
    )
    return DFTI_ERROR_CLASSES(status)


@dispatch
def DftiComputeBackward(
    handle: DFTI_DESCRIPTOR_HANDLE,
    xre_inout: NDArray,
    xim_inout: NDArray,
) -> DFTI_ERROR_CLASSES:
    status = mkl_ctypes.DftiComputeBackward(
        handle.pointer,
        xre_inout.ctypes.data_as(ctypes.c_void_p),
        xim_inout.ctypes.data_as(ctypes.c_void_p),
    )
    return DFTI_ERROR_CLASSES(status)


@dispatch
def DftiComputeBackward(
    handle: DFTI_DESCRIPTOR_HANDLE,
    xre_in: NDArray,
    xim_in: NDArray,
    yre_out: NDArray,
    yim_out: NDArray,
) -> DFTI_ERROR_CLASSES:
    status = mkl_ctypes.DftiComputeBackward(
        handle.pointer,
        xre_in.ctypes.data_as(ctypes.c_void_p),
        xim_in.ctypes.data_as(ctypes.c_void_p),
        yre_out.ctypes.data_as(ctypes.c_void_p),
        yim_out.ctypes.data_as(ctypes.c_void_p),
    )
    return DFTI_ERROR_CLASSES(status)


def DftiErrorMessage(status: DFTI_ERROR_CLASSES) -> str:
    message: bytes = mkl_ctypes.DftiErrorMessage(status.value)
    return message.decode("utf-8")


def DftiErrorClass(
    status: DFTI_ERROR_CLASSES,
    error_class: DFTI_ERROR_CLASSES,
) -> int:
    return mkl_ctypes.DftiErrorClass(
        status.value,
        error_class.value,
    )
