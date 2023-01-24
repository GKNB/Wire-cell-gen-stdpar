/**
 * Wrappers for FFTW based FFT
 */

#ifndef WIRECELL_STDPARARRAY_FFTW 
#define WIRECELL_STDPARARRAY_FFTW

#include <complex>
#include <cmath>
#include <cassert>
#include <iostream>   //FOR DEBUG
#include <omp.h>      //FOR DEBUG

#include <WireCellUtil/Array.h>

namespace WireCell 
{
  namespace StdparArray 
  {

    thread_local static Eigen::FFT<float> gEigenFFT;

    inline void dft_rc(std::complex<float>* out, const float* in, size_t N0, size_t N1, int dim = 0)
    {
      Eigen::Map<const Eigen::ArrayXXf> in_eigen(in, N0, N1);
      auto out_eigen = WireCell::Array::dft_rc(in_eigen, dim);
      memcpy((void*)out, (void*)out_eigen.data(), N0*N1*sizeof(std::complex<float>));
    }

    //FIXME: This should be optimized to be in-place, and test performance diff (both speed and memory)
    //As the out and in can be the same, I remove the const
    inline void dft_cc(std::complex<float>* out, std::complex<float> *in, size_t N0, size_t N1, int dim = 0)
    {
      Eigen::Map<Eigen::ArrayXXcf> in_eigen(in, N0, N1);
      auto out_eigen = WireCell::Array::dft_cc(in_eigen, dim);
      memcpy((void*)out, (void*)out_eigen.data(), N0*N1*sizeof(std::complex<float>));
    }

    //Can we do late evaluation for normalization? This takes several ms
    //FIXME: This should be optimized to be in-place like above
    //As the out and in could be the same, I remove const
    inline void idft_cc(std::complex<float>* out, std::complex<float> *in, size_t N0, size_t N1, int dim = 0)
    {
      Eigen::Map<Eigen::ArrayXXcf> in_eigen(in, N0, N1);
      auto out_eigen = WireCell::Array::idft_cc(in_eigen, dim);
      memcpy((void*)out, (void*)out_eigen.data(), N0*N1*sizeof(std::complex<float>));
    }

    inline void idft_cr(float* out, const std::complex<float> *in, size_t N0, size_t N1, int dim = 0)
    {
      Eigen::Map<const Eigen::ArrayXXcf> in_eigen(in, N0, N1);
      auto out_eigen = WireCell::Array::idft_cr(in_eigen, dim);
      memcpy((void*)out, (void*)out_eigen.data(), N0*N1*sizeof(float));
    }

  }  // namespace StdparArray
}  // namespace WireCell

#endif
