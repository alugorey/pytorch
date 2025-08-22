#undef __HIP_NO_HALF_CONVERSIONS__

#include <ATen/native/hip/ck_gemm/ck_gemm_template.h>

// Mappings for Sequence order per layout
// TN: A - S<1, 0, 2>, 2, 8 | B - S<1, 0, 2>, 2, 8
// NT: A - S<0, 2, 1>, 1, 4 | B - S<0, 2, 1>, 1, 4
// TT: A - S<0, 2, 1>, 1, 4 | B - S<1, 0, 2>, 2, 8
// NN: A - S<1, 0, 2>, 2, 8 | B - S<0, 2, 1>, 1, 4

namespace at::native {
void
ck_gemm_kernel_bf16bf16bf16_128_128x32x32_32x32_2x1_Default_v1(
        CUDABLAS_GEMM_ARGTYPES(at::BFloat16)) {
  // TODO: Add configs for other layouts
  //TN
  gemm_impl_base<
            at::BFloat16,       // Dtype
            128,                // BLOCK_SIZE
            128,                // MBLOCK
            32,                 // NBLOCK
            32,                 // KBLOCK
            8,                  // AK1
            8,                  // BK1
            32,                 // MPER_XDL
            32,                 // NPER_XDL
            2,                  // MPER_WAVE
            1,                  // NPER_WAVE
            S<4,32,1>,          // ABLOCK_CLUSTER_LENS
            S<1,0,2>,           // ABLOCK_CLUSTER_ORDER
            S<1,0,2>,           // ABLOCK_SRC_ORDER
            2,                  // ABLOCK_VECTOR_DIM
            8,                  // ABLOCK_SCALAR_VEC
            8,                  // ABLOCK_SCALAR_VEC_AK1
            1,                  // ABLOCK_LDS_EXTRAM
            S<4,32,1>,          // BBLOCK_CLUSTER_LENS
            S<1,0,2>,           // BBLOCK_CLUSTER_ORDER
            S<1,0,2>,           // BBLOCK_SRC_ORDER
            2,                  // BBLOCK_VECTOR_DIM
            8,                  // BBLOCK_SCALAR_VEC
            8,                  // BBLOCK_SCALAR_VEC_BK1
            1,                  // BBLOCK_LDS_EXTRAN
            1,                  // CMPER_WAVE
            1,                  // CNPER_WAVE
            S<1, 32, 1, 4>,     // BLOCK_CLUSTER_LENS
            S<8>,               // CDE_SCALAR_VEC
            ck::tensor_operation::device::GemmSpecialization::MNPadding,// PADDING
            true,               // TRANSA
            false,              // TRANSB
            ck::LoopScheduler::Default,
            ck::PipelineVersion::v2>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));

}
}; // namespace at::native
