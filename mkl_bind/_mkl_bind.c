#include <Python.h>
#include "mkl.h"

#ifdef _MSC_VER
#define DLL_EXPORT extern __declspec(dllexport)
#else
#define DLL_EXPORT extern
#endif

DLL_EXPORT int
sizeof_mkl_long()
{
	return sizeof(MKL_LONG);
}

DLL_EXPORT DFTI_DESCRIPTOR_HANDLE *
Py_DftiDescriptorNew()
{
	return (DFTI_DESCRIPTOR_HANDLE *)malloc(sizeof(DFTI_DESCRIPTOR_HANDLE));
}

DLL_EXPORT void
Py_DftiDescriptorDelete(DFTI_DESCRIPTOR_HANDLE *desc_handle)
{
	free(desc_handle);
}

DLL_EXPORT MKL_LONG
Py_DftiCreateDescriptor(DFTI_DESCRIPTOR_HANDLE *desc_handle,
						enum DFTI_CONFIG_VALUE precision,
						enum DFTI_CONFIG_VALUE forward_domain,
						MKL_LONG dimension, void *length)
{
	return DftiCreateDescriptor(desc_handle, precision, forward_domain, dimension, length);
}

DLL_EXPORT MKL_LONG
Py_DftiCopyDescriptor(DFTI_DESCRIPTOR_HANDLE *desc_handle_original,
					  DFTI_DESCRIPTOR_HANDLE *desc_handle_copy)
{
	return DftiCopyDescriptor(*desc_handle_original, desc_handle_copy);
}

DLL_EXPORT MKL_LONG
Py_DftiFreeDescriptor(DFTI_DESCRIPTOR_HANDLE *desc_handle)
{
	return DftiFreeDescriptor(desc_handle);
}

DLL_EXPORT MKL_LONG
Py_DftiSetValue(DFTI_DESCRIPTOR_HANDLE *desc_handle,
				enum DFTI_CONFIG_PARAM config_param,
				void* config_value)
{
	return DftiSetValue(*desc_handle, config_param, config_value);
}

DLL_EXPORT MKL_LONG
Py_DftiGetValue(DFTI_DESCRIPTOR_HANDLE *desc_handle,
				enum DFTI_CONFIG_PARAM config_param,
				void *config_value)
{
	return DftiGetValue(*desc_handle, config_param, config_value);
}

DLL_EXPORT MKL_LONG
Py_DftiCommitDescriptor(DFTI_DESCRIPTOR_HANDLE *desc_handle)
{
	return DftiCommitDescriptor(*desc_handle);
}

// status = DftiComputeForward( desc_handle, x_inout )
// status = DftiComputeForward( desc_handle, x_in, y_out )
// status = DftiComputeForward( desc_handle, xre_inout, xim_inout )
// status = DftiComputeForward( desc_handle, xre_in, xim_in, yre_out, yim_out )

DLL_EXPORT MKL_LONG
Py_DftiComputeForward(DFTI_DESCRIPTOR_HANDLE *desc_handle,
					  void *x0, void *x1, void *x2, void *x3)
{
	return DftiComputeForward(*desc_handle, x0, x1, x2, x3);
}

// status = DftiComputeBackward(desc_handle, x_inout);
// status = DftiComputeBackward(desc_handle, y_in, x_out);
// status = DftiComputeBackward(desc_handle, xre_inout, xim_inout);
// status = DftiComputeBackward(desc_handle, yre_in, yim_in, xre_out, xim_out);

DLL_EXPORT MKL_LONG
Py_DftiComputeBackward(DFTI_DESCRIPTOR_HANDLE *desc_handle,
					   void *x0, void *x1, void *x2, void *x3)
{
	return DftiComputeBackward(*desc_handle, x0, x1, x2, x3);
}

DLL_EXPORT char *
Py_DftiErrorMessage(MKL_LONG status)
{
	return DftiErrorMessage(status);
}

DLL_EXPORT MKL_LONG
Py_DftiErrorClass(MKL_LONG status, MKL_LONG error_class)
{
	return DftiErrorClass(status, error_class);
}
