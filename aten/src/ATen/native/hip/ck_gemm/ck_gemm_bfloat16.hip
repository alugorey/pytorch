#undef __HIP_NO_HALF_CONVERSIONS__

#include <ATen/native/hip/ck_gemm/ck_gemm.h>
#if defined(USE_ROCM_CK_GEMM)
#include <ATen/native/hip/ck_gemm/ck_gemm_template.h>
#include <ATen/native/hip/ck_gemm/ck_gemm_kernels/ck_kernel_collection.h>
#include <ck/utility/sequence.hpp>

#include <unordered_map>
#include <functional>
#include <initializer_list>

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

namespace at::native {


using GemmBaseKernel = std::function<void(CUDABLAS_GEMM_TYPES(at::BFloat16), bool)>;

// For certain high priority shapes, we skip the heuristic and call best kernel directly
static const std::unordered_map<std::tuple<int, int, int>, GemmBaseKernel, IntTupleHash> gemm_base_lookup_dispatch = {
    {{64, 512, 4096},
     ck_gemm_kernel_bf16bf16bf16_128_128x32x32_32x32_2x1_Default_v1}};


void dispatch_bfloat16_gemm(CUDABLAS_GEMM_ARGTYPES(at::BFloat16)) {
  // If any of the shapes cant be tiled, we must use padding.


  // Determine padding and kernel size
  auto [kernel_size, use_padding] = get_kernel_size(m, n, k);
  
  // TODO: need to probably hardcode the padding type, since that is likely going to change frequencly
  //       per kernl
  // First check direct look up
  auto it = gemm_base_lookup_dispatch.find({m, n, k});
  if(it != gemm_base_lookup_dispatch.end()) {
    it->second(CUDABLAS_GEMM_ARGS(at::BFloat16), use_padding);
  } else { // No direct mapping found, go to heuristic based on MNK size

  //std::cout << kernel_size << "" << use_padding << std::endl;
    if (kernel_size == KernelSize::Small) {
      ck_gemm_kernel_bf16bf16bf16_256_128x128x64_32x32_2x2_8x32x1_8x32x1_1x32x1x8_4_Intrawave_v3(
                                            CUDABLAS_GEMM_ARGS(at::BFloat16), use_padding);
    } else if (kernel_size == KernelSize::Large) {
      ck_gemm_kernel_bf16bf16bf16_256_256x128x64_16x16_8x4_8x32x1_8x32x1_1x32x1x8_4_Intrawave_v3(
                                            CUDABLAS_GEMM_ARGS(at::BFloat16), use_padding);
    } else { // kernel_size == KernelSize::Medium
      ck_gemm_kernel_bf16bf16bf16_256_128x128x64_16x16_4x4_8x32x1_8x32x1_1x32x1x8_4_Intrawave_v3(
                                            CUDABLAS_GEMM_ARGS(at::BFloat16), use_padding);
    }
  }
}

void dispatch_bfloat16_gemm_wmma(CUDABLAS_GEMM_ARGTYPES(at::BFloat16)) {
  // If any of the shapes cant be tiled, we must use padding.
  bool use_padding = ((m % 256 != 0) || (n % 128 != 0) || (k % 64 != 0));
  // Dispatch to best implementation.
  // TODO add more configurations. Optimize.

  bool transa_ = std::tolower(transa) != 'n';
  bool transb_ = std::tolower(transb) != 'n';

  if (use_padding) {
      if(transa_ && transb_) { // col , col
          gemm_impl_wmma<
            at::BFloat16,
            256,
            128,
            256,
            64,
            8,
            16,
            16,
            4,
            4,
            S<4, 64, 1>,
            S<0, 2, 1>,
            S<0, 2, 1>,
            1,
            1,
            8,
            true,
            S<4, 64, 1>,
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            1,
            1,
            S<1, 32, 1,  8>,
            8,
            true,
            true,
            true>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else if(transa_ && !transb_) { // row, col
          gemm_impl_wmma<
            at::BFloat16,
            256,
            128,
            256,
            64,
            8,
            16,
            16,
            4,
            4,
            S<4, 64, 1>,
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            S<4, 64, 1>,
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            1,
            1,
            S<1, 32, 1,  8>,
            8,
            true,
            true,
            false>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else if(!transa_ && transb_) { //col, row
          gemm_impl_wmma<
            at::BFloat16,
            256,
            128,
            256,
            64,
            8,
            16,
            16,
            4,
            4,
            S<4, 64, 1>,
            S<0, 2, 1>,
            S<0, 2, 1>,
            1,
            1,
            8,
            true,
            S<4, 64, 1>,
            S<0, 2, 1>,
            S<0, 2, 1>,
            1,
            1,
            8,
            true,
            1,
            1,
            S<1, 32, 1,  8>,
            8,
            true,
            false,
            true>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else if(!transa_ && !transb_) { //row, row
          gemm_impl_wmma<
            at::BFloat16,
            256,
            128,
            256,
            64,
            8,
            16,
            16,
            4,
            4,
            S<4, 64, 1>,
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            S<4, 64, 1>,
            S<0, 2, 1>,
            S<0, 2, 1>,
            1,
            1,
            8,
            true,
            1,
            1,
            S<1, 32, 1,  8>,
            8,
            true,
            false,
            false>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else {
        TORCH_CHECK(false, "unreachable");
      }
  } else {
         if(transa_ && transb_) { // col , col
          gemm_impl_wmma<
            at::BFloat16,
            256,
            128,
            256,
            64,
            8,
            16,
            16,
            4,
            4,
            S<4, 64, 1>,
            S<0, 2, 1>,
            S<0, 2, 1>,
            1,
            1,
            8,
            true,
            S<4, 64, 1>,
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            1,
            1,
            S<1, 32, 1,  8>,
            8,
            false,
            true,
            true>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else if(transa_ && !transb_) { // row, col
          gemm_impl_wmma<
            at::BFloat16,
            256,
            128,
            256,
            64,
            8,
            16,
            16,
            4,
            4,
            S<4, 64, 1>,
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            S<4, 64, 1>,
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            1,
            1,
            S<1, 32, 1,  8>,
            8,
            false,
            true,
            false>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else if(!transa_ && transb_) { //col, row
          gemm_impl_wmma<
            at::BFloat16,
            256,
            128,
            256,
            64,
            8,
            16,
            16,
            4,
            4,
            S<4, 64, 1>,
            S<0, 2, 1>,
            S<0, 2, 1>,
            1,
            1,
            8,
            true,
            S<4, 64, 1>,
            S<0, 2, 1>,
            S<0, 2, 1>,
            1,
            1,
            8,
            true,
            1,
            1,
            S<1, 32, 1,  8>,
            8,
            false,
            false,
            true>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else if(!transa_ && !transb_) { //row, row
          gemm_impl_wmma<
            at::BFloat16,
            256,
            128,
            256,
            64,
            8,
            16,
            16,
            4,
            4,
            S<4, 64, 1>,
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            S<4, 64, 1>,
            S<0, 2, 1>,
            S<0, 2, 1>,
            1,
            1,
            8,
            true,
            1,
            1,
            S<1, 32, 1,  8>, 8,
            false,
            false,
            false>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else {
        TORCH_CHECK(false, "unreachable");
      }
  }
}

template <>
void gemm_internal_ck<at::BFloat16>(CUDABLAS_GEMM_ARGTYPES(at::BFloat16)) {
  auto dprops = at::cuda::getCurrentDeviceProperties();
  std::string_view arch(dprops->gcnArchName);
  if (arch == "gfx1100") {
    dispatch_bfloat16_gemm_wmma(CUDABLAS_GEMM_ARGS(at::BFloat16));
  } else{
    dispatch_bfloat16_gemm(CUDABLAS_GEMM_ARGS(at::BFloat16));
  }
}

} // namespace at::native
#endif // USE_ROCM_CK_GEMM
