#undef __HIP_NO_HALF_CONVERSIONS__

#include <ATen/native/hip/ck_gemm.h>
#include <ATen/native/hip/ck_gemm_template.h>
#include <ck/utility/sequence.hpp>
#include <iostream>

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

namespace at::native {

void dispatch_bfloat16_gemm(CUDABLAS_GEMM_ARGTYPES(at::BFloat16)) {
  // If any of the shapes cant be tiled, we must use padding.
  bool use_padding = ((m % 256 != 0) || (n % 128 != 0) || (k % 64 != 0));
  // Dispatch to best implementation.
  // TODO add more configurations. Optimize.
  bool transa_ = std::tolower(transa) != 'n';
  bool transb_ = std::tolower(transb) != 'n';

  if (use_padding) {
    if (m <= 128) {
      if(transa_ && transb_) {
	     std::cout << "PADDED SMALL TT" << std::endl;
         gemm_impl<
            at::BFloat16,
            256,
            128,
            128,
            64,
            8,
            8,
            32,
            32,
            2,
            2,
            S<8,32,1>,
            S<0,2,1>,
            S<0,2,1>,
            1,
            4,
            8,
            0,
            S<8,32,1>,
            S<1,0,2>,
            S<1,0,2>,
            2,
            8,
            8,
            0,
            1,
            1,
            S<1, 32, 1, 8>,
            S<4>,
            true,
            true,
            true>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else if(transa_ && !transb_) {
	    std::cout << "PADDED SMALL TN" << std::endl;
        gemm_impl<
            at::BFloat16,
            256,
            128,
            128,
            64,
            8,
            8,
            32,
            32,
            2,
            2,
            S<8,32,1>,
            S<1,0,2>,
            S<1,0,2>,
            2,
            8,
            8,
            0,
            S<8,32,1>,
            S<1,0,2>,
            S<1,0,2>,
            2,
            8,
            8,
            0,
            1,
            1,
            S<1, 32, 1, 8>,
            S<4>,
            true,
            true,
            false>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else if(!transa_ && transb_) {
	  	  std::cout << "PADDED SMALL NT" << std::endl;
          gemm_impl<
            at::BFloat16,
            256,
            128,
            128,
            64,
            8,
            8,
            32,
            32,
            2,
            2,
            S<8,32,1>,
            S<0,2,1>,
            S<0,2,1>,
            1,
            4,
            8,
            0,
            S<8,32,1>,
            S<0,2,1>,
            S<0,2,1>,
            1,
            4,
            8,
            0,
            1,
            1,
            S<1, 32, 1, 8>,
            S<4>,
            true,
            false,
            true>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else if(!transa_ && !transb_) {
	      std::cout << "PADDED SMALL NN" << std::endl;
          gemm_impl<
           at::BFloat16,
            256,
            128,
            128,
            64,
            8,
            8,
            32,
            32,
            2,
            2,
            S<8,32,1>,
            S<1,0,2>,
            S<1,0,2>,
            2,
            8,
            8,
            0,
            S<8,32,1>,
            S<0,2,1>,
            S<0,2,1>,
            1,
            4,
            8,
            0,
            1,
            1,
            S<1, 32, 1, 8>,
            S<4>,
            true,
            false,
            false>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else {
        TORCH_CHECK(false, "unreachable");
      }

    } else { // !( m <= 128 )
      if(transa_ && transb_) {
	    std::cout << "PADDED MED TT" << std::endl;
        gemm_impl<
            at::BFloat16,
            256,
            128,
            128,
            64,
            8,
            8,
            32,
            32,
            2,
            2,
            S<8,32,1>,
            S<0,2,1>,
            S<0,2,1>,
            1,
            4,
            8,
            0,
            S<8,32,1>,
            S<1,0,2>,
            S<1,0,2>,
            2,
            8,
            8,
            0,
            1,
            1,
            S<1, 32, 1, 8>,
            S<4>,
            true,
            true,
            true>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else if(transa_ && !transb_) {
	      //std::cout << "PADDED MED TN" << std::endl;
          gemm_impl<
            at::BFloat16,
            256, // changed
            128, // changed
            128, // changed
            64,
            8,
            8,
            16, //changed
            16, //changed
            4,   //changed
            4,   //changed
            S<4,64,1>, //changed
            S<1,0,2>,
            S<1,0,2>,
            2,
            8,
            8,
            0,
            S<4,64,1>,
            S<1,0,2>,
            S<1,0,2>,
            2,
            8,
            8,
            0,
            1,
            2, //changed
            S<1, 32, 1, 8>, // changed
            S<8>, // changed
            true, // PADDING
            true,
            false,
			ck::BlockGemmPipelineScheduler::Intrawave,
			ck::BlockGemmPipelineVersion::v3>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else if(!transa_ && transb_) {
	  //    std::cout << "PADDED MED NT" << std::endl;
          gemm_impl<
            at::BFloat16,
            256,
            128,
            128,
            64,
            8,
            8,
            16,
            16,
            4,
            4,
            S<4,64,1>,
            S<0,2,1>,
            S<0,2,1>,
            1,
            4,
            8,
            0,
            S<4,64,1>,
            S<0,2,1>,
            S<0,2,1>,
            1,
            4,
            8,
            0,
            1,
            1,
            S<1, 32, 1, 8>,
            S<8>,
            true,
            false,
            true,
            ck::BlockGemmPipelineScheduler::Intrawave,
			ck::BlockGemmPipelineVersion::v3>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else if(!transa_ && !transb_) {
	      std::cout << "PADDED MED NN" << std::endl;
          gemm_impl<
            at::BFloat16,
            256,
            128,
            128,
            64,
            8,
            8,
            32,
            32,
            2,
            2,
            S<8,32,1>,
            S<1,0,2>,
            S<1,0,2>,
            2,
            8,
            8,
            0,
            S<8,32,1>,
            S<0,2,1>,
            S<0,2,1>,
            1,
            4,
            8,
            0,
            1,
            1,
            S<1, 32, 1, 8>,
            S<4>,
            true,
            false,
            false>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else {
        TORCH_CHECK(false, "unreachable");
      }
    }
  } else { // !(use_padding)
    {
      if(transa_ && transb_) {
	      std::cout << "UNPADDED TT" << std::endl;
          gemm_impl<
            at::BFloat16,
            256,
            128,
            128,
            64,
            8,
            8,
            32,
            32,
            2,
            2,
            S<8,32,1>,
            S<0,2,1>,
            S<0,2,1>,
            1,
            4,
            8,
            0,
            S<8,32,1>,
            S<1,0,2>,
            S<1,0,2>,
            2,
            8,
            8,
            0,
            1,
            1,
            S<1, 32, 1, 8>,
            S<4>,
            false,
            true,
            true>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else if(transa_ && !transb_) {
	      std::cout << "UNPADDED TN" << std::endl;
          gemm_impl<
            at::BFloat16,
            256,
            128,
            128,
            64,
            8,
            8,
            32,
            32,
            2,
            2,
            S<8,32,1>,
            S<1,0,2>,
            S<1,0,2>,
            2,
            8,
            8,
            0,
            S<8,32,1>,
            S<1,0,2>,
            S<1,0,2>,
            2,
            8,
            8,
            0,
            1,
            1,
            S<1, 32, 1, 8>,
            S<4>,
            false,
            true,
            false>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else if(!transa_ && transb_) {
	      std::cout << "UNPADDED NT" << std::endl;
          gemm_impl<
            at::BFloat16,
            256,
            128,
            128,
            64,
            8,
            8,
            32,
            32,
            2,
            2,
            S<8,32,1>,
            S<0,2,1>,
            S<0,2,1>,
            1,
            4,
            8,
            0,
            S<8,32,1>,
            S<0,2,1>,
            S<0,2,1>,
            1,
            4,
            8,
            0,
            1,
            1,
            S<1, 32, 1, 8>,
            S<4>,
            false,
            false,
            true>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else if(!transa_ && !transb_) {
	      std::cout << "UNPADDED NN" << std::endl;
          gemm_impl<
            at::BFloat16,
            256,
            128,
            128,
            64,
            8,
            8,
            32,
            32,
            2,
            2,
            S<8,32,1>,
            S<1,0,2>,
            S<1,0,2>,
            2,
            8,
            8,
            0,
            S<8,32,1>,
            S<0,2,1>,
            S<0,2,1>,
            1,
            4,
            8,
            0,
            1,
            1,
            S<1, 32, 1, 8>,
            S<4>,
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
}



template <>
void gemm_internal_ck<at::BFloat16>(CUDABLAS_GEMM_ARGTYPES(at::BFloat16)) {
  dispatch_bfloat16_gemm(CUDABLAS_GEMM_ARGS(at::BFloat16));
}

} // namespace at::native
