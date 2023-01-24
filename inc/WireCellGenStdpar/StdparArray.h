/**
 * Similar like the WireCell::Array with Eigen backend,
 * this StdparArray provides interface for FFTs.
 */

#ifndef WIRECELL_STDPARARRAY
#define WIRECELL_STDPARARRAY

#include <string>
#include <typeinfo>

#if defined STDPAR_ENABLE_CUDA
    #include "WireCellGenStdpar/StdparArray_cuda.h"
#elif defined STDPAR_ENABLE_HIP
//    #include "WireCellGenStdpar/StdparArray_hip.h"
    #error Currently stdpar does not support AMD GPU
#else
    #include "WireCellGenStdpar/StdparArray_fftw.h"
#endif

#endif
