#ifndef WIRECELL_STDPARATOMICWRAPPER 
#define WIRECELL_STDPARATOMICWRAPPER

__device__ double atomicAddWrapper(double* address, double val);

__device__ float  atomicAddWrapper(float* address, float val);

#endif
