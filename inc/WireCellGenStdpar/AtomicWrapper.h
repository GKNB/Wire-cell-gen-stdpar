#ifndef WIRECELL_STDPARATOMICWRAPPER 
#define WIRECELL_STDPARATOMICWRAPPER

namespace WireCell{
  namespace GenStdpar{

    __device__ double atomicAddWrapper(double* address, double val);

    __device__ float  atomicAddWrapper(float* address, float val);

  } //namespace GenStdpar
} //namespace WireCell

#endif  //WIRECELL_STDPARATOMICWRAPPER
