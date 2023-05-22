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

#include <unsupported/Eigen/FFT>

namespace WireCell 
{
  namespace StdparArray 
  {

    thread_local static Eigen::FFT<float> gEigenFFT;
    thread_local static Eigen::FFT<float> gEigenFFT_dft_1d;      // c2c fwd and inv
    thread_local static Eigen::FFT<float> gEigenFFT_dft_r2c_1d;  // r2c fwd
    thread_local static Eigen::FFT<float> gEigenFFT_dft_c2r_1d;  // c2r inv

    inline void dft_rc(std::complex<float>* out, const float* in, size_t N0, size_t N1, int dim = 0)
    {
      Eigen::Map<const Eigen::ArrayXXf> in_eigen(in, N0, N1);
      
//      auto out_eigen = WireCell::Array::dft_rc(in_eigen, dim);

      const int nrows = in_eigen.rows();
      const int ncols = in_eigen.cols();

      Eigen::MatrixXcf out_eigen(nrows, ncols);

      if (dim == 0) {
          for (int irow = 0; irow < nrows; ++irow) {
              Eigen::VectorXcf fspec(ncols);
              Eigen::VectorXf tmp = in_eigen.row(irow);
              gEigenFFT_dft_r2c_1d.fwd(fspec, tmp);  // r2c
              out_eigen.row(irow) = fspec;
          }
      }
      else if (dim == 1) {
          for (int icol = 0; icol < ncols; ++icol) {
              Eigen::VectorXcf fspec(nrows);
              Eigen::VectorXf tmp = in_eigen.col(icol);
              gEigenFFT_dft_r2c_1d.fwd(fspec, tmp);  // r2c
              out_eigen.col(icol) = fspec;
          }
      }
      memcpy((void*)out, (void*)out_eigen.data(), N0*N1*sizeof(std::complex<float>));
    }

    //FIXME: This should be optimized to be in-place, and test performance diff (both speed and memory)
    //As the out and in can be the same, I remove the const
    inline void dft_cc(std::complex<float>* out, std::complex<float> *in, size_t N0, size_t N1, int dim = 0)
    {
      Eigen::Map<Eigen::ArrayXXcf> in_eigen(in, N0, N1);
//      auto out_eigen = WireCell::Array::dft_cc(in_eigen, dim);

      const int nrows = in_eigen.rows();
      const int ncols = in_eigen.cols();

      Eigen::MatrixXcf out_eigen(nrows, ncols);

      out_eigen = in_eigen.matrix();

      if (dim == 0) {
          for (int irow = 0; irow < nrows; ++irow) {
              Eigen::VectorXcf pspec(ncols);
              gEigenFFT_dft_1d.fwd(pspec, out_eigen.row(irow));  // c2c
              out_eigen.row(irow) = pspec;
          }
      }
      else {
          for (int icol = 0; icol < ncols; ++icol) {
              Eigen::VectorXcf pspec(nrows);
              gEigenFFT_dft_1d.fwd(pspec, out_eigen.col(icol));  // c2c
              out_eigen.col(icol) = pspec;
          }
      }
      memcpy((void*)out, (void*)out_eigen.data(), N0*N1*sizeof(std::complex<float>));
    }

    //Can we do late evaluation for normalization? This takes several ms
    //FIXME: This should be optimized to be in-place like above
    //As the out and in could be the same, I remove const
    inline void idft_cc(std::complex<float>* out, std::complex<float> *in, size_t N0, size_t N1, int dim = 0)
    {
      Eigen::Map<Eigen::ArrayXXcf> in_eigen(in, N0, N1);
//      auto out_eigen = WireCell::Array::idft_cc(in_eigen, dim);

      const int nrows = in_eigen.rows();
      const int ncols = in_eigen.cols();

      // gEigenFFT works on matrices, not arrays, also don't step on const input
      Eigen::MatrixXcf out_eigen(nrows, ncols);
      out_eigen = in_eigen.matrix();

      if (dim == 1) {
          for (int icol = 0; icol < ncols; ++icol) {
              Eigen::VectorXcf pspec(nrows);
              gEigenFFT_dft_1d.inv(pspec, out_eigen.col(icol));  // c2c
              out_eigen.col(icol) = pspec;
          }
      }
      else if (dim == 0) {
          for (int irow = 0; irow < nrows; ++irow) {
              Eigen::VectorXcf pspec(ncols);
              gEigenFFT_dft_1d.inv(pspec, out_eigen.row(irow));  // c2c
              out_eigen.row(irow) = pspec;
          }
      }
      memcpy((void*)out, (void*)out_eigen.data(), N0*N1*sizeof(std::complex<float>));
    }

    inline void idft_cr(float* out, const std::complex<float> *in, size_t N0, size_t N1, int dim = 0)
    {
      Eigen::Map<const Eigen::ArrayXXcf> in_eigen(in, N0, N1);
//      auto out_eigen = WireCell::Array::idft_cr(in_eigen, dim);

      const int nrows = in_eigen.rows();
      const int ncols = in_eigen.cols();

      // gEigenFFT works on matrices, not arrays, also don't step on const input
      Eigen::MatrixXcf partial(nrows, ncols);
      partial = in_eigen.matrix();

      Eigen::ArrayXXf out_eigen(nrows, ncols);

      if (dim == 0) {
          for (int irow = 0; irow < nrows; ++irow) {
              Eigen::VectorXf wave(ncols);                        // back to real-valued time series
              gEigenFFT_dft_c2r_1d.inv(wave, partial.row(irow));  // c2r
              out_eigen.row(irow) = wave;
          }
      }
      else if (dim == 1) {
          for (int icol = 0; icol < ncols; ++icol) {
              Eigen::VectorXf wave(nrows);
              gEigenFFT_dft_c2r_1d.inv(wave, partial.col(icol));  // c2r
              out_eigen.col(icol) = wave;
          }
      }
      memcpy((void*)out, (void*)out_eigen.data(), N0*N1*sizeof(float));
    }

  }  // namespace StdparArray
}  // namespace WireCell

#endif
