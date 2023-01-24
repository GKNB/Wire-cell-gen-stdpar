/**
 * Wrappers for cuFFT based FFT
 */

#ifndef WIRECELL_STDPARARRAY_CUDA
#define WIRECELL_STDPARARRAY_CUDA

#include <complex>
#include <cmath>
#include <cassert>
#include <iostream>   //FOR DEBUG
#include <algorithm>
#include <execution>


#include <cufft.h>

#include "WireCellGenStdpar/counting_iter.h"


#define CUFFT_CALL(x) do { if((x)!=CUFFT_SUCCESS) { \
  printf("Error at %s:%d, with code %d\n",__FILE__,__LINE__,x);\
  exit(EXIT_FAILURE);}} while(0)

namespace WireCell 
{
  namespace StdparArray 
  {

    inline void dft_rc_2d(std::complex<float>* out, const float* in, size_t N0, size_t N1)
    {
      cufftHandle plan;
      CUFFT_CALL(cufftPlan2d(&plan, (int)N0, (int)N1, CUFFT_R2C));
      CUFFT_CALL(cufftExecR2C(plan, (cufftReal*)in, (cufftComplex*)out));
      CUFFT_CALL(cufftDestroy(plan));
    }

    inline void dft_rc(std::complex<float>* out, const float* in, size_t N0, size_t N1, int dim = 0)
    {
      cufftHandle plan;

      if(dim == 0) 
      {
        int n[] = {(int)N1};
        int inembed[] = {(int)N1};
        int onembed[] = {(int)N1};
        CUFFT_CALL(cufftPlanMany(&plan, 1, n, inembed, (int)N0, 1, onembed, (int)N0, 1, CUFFT_R2C, (int)N0));
        CUFFT_CALL(cufftExecR2C(plan, (cufftReal*)in, (cufftComplex*)out));
        CUFFT_CALL(cufftDestroy(plan));
      }

      else if(dim == 1) 
      {
        int n[] = {(int)N0};
        int inembed[] = {(int)N0};
        int onembed[] = {(int)N0};
        CUFFT_CALL(cufftPlanMany(&plan, 1, n, inembed, 1, (int)N0, onembed, 1, (int)N0, CUFFT_R2C, (int)N1));
        CUFFT_CALL(cufftExecR2C(plan, (cufftReal*)in, (cufftComplex*)out));
        CUFFT_CALL(cufftDestroy(plan));
      }
    }

    //FIXME: This should be optimized to be in-place, and test performance diff (both speed and memory)
    //As the out and in can be the same, I remove the const
    inline void dft_cc(std::complex<float>* out, std::complex<float> *in, size_t N0, size_t N1, int dim = 0)
    {
      cufftHandle plan;

      if(dim == 0) 
      {
        int n[] = {(int)N1};
        int inembed[] = {(int)N1};
        int onembed[] = {(int)N1};
        CUFFT_CALL(cufftPlanMany(&plan, 1, n, inembed, (int)N0, 1, onembed, (int)N0, 1, CUFFT_C2C, (int)N0));
        CUFFT_CALL(cufftExecC2C(plan, (cufftComplex*)in, (cufftComplex*)out, CUFFT_FORWARD));
        CUFFT_CALL(cufftDestroy(plan));
      }

      else if(dim == 1) 
      {
        int n[] = {(int)N0};
        int inembed[] = {(int)N0};
        int onembed[] = {(int)N0};
        CUFFT_CALL(cufftPlanMany(&plan, 1, n, inembed, 1, (int)N0, onembed, 1, (int)N0, CUFFT_C2C, (int)N1));
        CUFFT_CALL(cufftExecC2C(plan, (cufftComplex*)in, (cufftComplex*)out, CUFFT_FORWARD));
        CUFFT_CALL(cufftDestroy(plan));
      }
    }

    //Can we do late evaluation for normalization? This takes several ms
    //FIXME: This should be optimized to be in-place like above
    //As the out and in could be the same, I remove const
    inline void idft_cc(std::complex<float>* out, std::complex<float> *in, size_t N0, size_t N1, int dim = 0)
    {
      cufftHandle plan;

      if(dim == 0) 
      {
        int n[] = {(int)N1};
        int inembed[] = {(int)N1};
        int onembed[] = {(int)N1};
        CUFFT_CALL(cufftPlanMany(&plan, 1, n, inembed, (int)N0, 1, onembed, (int)N0, 1, CUFFT_C2C, (int)N0));
        CUFFT_CALL(cufftExecC2C(plan, (cufftComplex*)in, (cufftComplex*)out, CUFFT_INVERSE));
        CUFFT_CALL(cufftDestroy(plan));

        std::for_each_n(par_unseq, counting_iterator(0), N0 * N1, 
                        [=](unsigned int i){ out[i] /= N1; });
      }

      else if(dim == 1) 
      {
        int n[] = {(int)N0};
        int inembed[] = {(int)N0};
        int onembed[] = {(int)N0};
        CUFFT_CALL(cufftPlanMany(&plan, 1, n, inembed, 1, (int)N0, onembed, 1, (int)N0, CUFFT_C2C, (int)N1));
        CUFFT_CALL(cufftExecC2C(plan, (cufftComplex*)in, (cufftComplex*)out, CUFFT_INVERSE));
        CUFFT_CALL(cufftDestroy(plan));

        std::for_each_n(par_unseq, counting_iterator(0), N0 * N1, 
                        [=](unsigned int i){ out[i] /= N0; });
      }
    }

    inline void idft_cr(float* out, const std::complex<float> *in, size_t N0, size_t N1, int dim = 0)
    {
      cufftHandle plan;

      if(dim == 0) 
      {
        int n[] = {(int)N1};
        int inembed[] = {(int)N1};
        int onembed[] = {(int)N1};
        CUFFT_CALL(cufftPlanMany(&plan, 1, n, inembed, (int)N0, 1, onembed, (int)N0, 1, CUFFT_C2R, (int)N0));
        CUFFT_CALL(cufftExecC2R(plan, (cufftComplex*)in, (cufftReal*)out));
        CUFFT_CALL(cufftDestroy(plan));

        std::for_each_n(par_unseq, counting_iterator(0), N0 * N1, 
                        [=](unsigned int i){ out[i] /= N1; });
      }

      else if(dim == 1) 
      {
        int n[] = {(int)N0};
        int inembed[] = {(int)N0};
        int onembed[] = {(int)N0};
        CUFFT_CALL(cufftPlanMany(&plan, 1, n, inembed, 1, (int)N0, onembed, 1, (int)N0, CUFFT_C2R, (int)N1));
        CUFFT_CALL(cufftExecC2R(plan, (cufftComplex*)in, (cufftReal*)out));
        CUFFT_CALL(cufftDestroy(plan));

        std::for_each_n(par_unseq, counting_iterator(0), N0 * N1, 
                        [=](unsigned int i){ out[i] /= N0; });
      }
    }

  }  // namespace StdparArray
}  // namespace WireCell

#endif
