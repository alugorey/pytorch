#include <ATen/native/transformers/hip/flash_attn/flash_common_hip.hpp>
#include <ATen/native/transformers/hip/flash_attn/ck/me_ck_api.h>

#include <iostream>

namespace pytorch_flash {
std::tuple<
    at::Tensor, // dQ
    at::Tensor, // dK
    at::Tensor, // dV
    at::Tensor> // dBias
mem_eff_backward_ck(
    const at::Tensor &dout,
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &out,
    const at::Tensor &softmax_lse,
    std::optional<at::Tensor> &dq_,
    std::optional<at::Tensor> &dk_,
    std::optional<at::Tensor> &dv_,
    std::optional<at::Tensor> &attn_bias,
    const at::Tensor &cu_seqlens_q,
    const at::Tensor &cu_seqlens_k,
    const int max_seqlen_q,
    const int max_seqlen_k,
    const float p_dropout,
    const float scale,
    const bool is_causal,
    const bool deterministic,
    const bool zero_tensors,
    const at::Tensor philox_seed,
    const at::Tensor philox_offset)
{
// TODO implement wrapper
  std::cout << "HIT MY MEM_EFF ENTRY POINT" << std::endl;
  return std::make_tuple(Tensor{}, Tensor{}, Tensor{}, Tensor{});
}


} // namespace pytorch_flash
