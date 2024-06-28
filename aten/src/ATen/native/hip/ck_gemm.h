#pragma once

#include <ATen/OpMathType.h>

namespace at::native {



#define CK_GEMM_ARGTYPES(Dtype)                                                             \
  char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<Dtype> alpha,  \
      const Dtype *a, int64_t lda, const Dtype *b, int64_t ldb, at::opmath_type<Dtype> beta,\
      Dtype *c, int64_t ldc

#define CK_GEMM_ARGS(Dtype) transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc


template <typename Dtype>
inline void gemm_internal_ck(CK_GEMM_ARGTYPES(Dtype)) {
  static_assert(false&&sizeof(Dtype),"at::cuda::blas_gemm_internal_ck: not implemented");
}

template <>
void gemm_internal_ck<double>(CK_GEMM_ARGTYPES(double));
template <>
void gemm_internal_ck<float>(CK_GEMM_ARGTYPES(float));
template <>
void gemm_internal_ck<c10::complex<double>>(CK_GEMM_ARGTYPES(c10::complex<double>));
template <>
void gemm_internal_ck<c10::complex<float>>(CK_GEMM_ARGTYPES(c10::complex<float>));
template <>
void gemm_internal_ck<at::Half>(CK_GEMM_ARGTYPES(at::Half));
template <>
void gemm_internal_ck<at::BFloat16>(CK_GEMM_ARGTYPES(at::BFloat16));



} // namespace at::native
