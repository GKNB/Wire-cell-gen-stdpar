#include "WireCellGenStdpar/AtomicWrapper.h"

#include "cuda_runtime.h"

using namespace WireCell;

__device__ double GenStdpar::atomicAddWrapper(double* address, double val)
{
  return atomicAdd(address, val);
}

__device__ float  GenStdpar::atomicAddWrapper(float* address, float val)
{
  return atomicAdd(address, val);
}
