#include "WireCellGenStdpar/AtomicWrapper.h"

#include "cuda_runtime.h"

__device__ double atomicAddWrapper(double* address, double val)
{
  return atomicAdd(address, val);
}

__device__ float  atomicAddWrapper(float* address, float val)
{
  return atomicAdd(address, val);
}
