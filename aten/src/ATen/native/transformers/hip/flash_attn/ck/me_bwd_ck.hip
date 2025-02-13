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
    const at::Tensor &dq_,
    const at::Tensor &dk_,
    const at::Tensor &dv_,
    std::optional<at::Tensor> &attn_bias,
    std::optional<at::Tensor> &cu_seqlens_q,
    std::optional<at::Tensor> &cu_seqlens_k,
    int max_seqlen_q,
    int max_seqlen_k,
    float p_dropout,
    float scale,
    bool is_causal,
    bool deterministic,
    bool zero_tensors,
    at::Tensor philox_seed,
    at::Tensor philox_offset)
{
// TODO implement wrapper
  std::cout << "HIT MY MEM_EFF ENTRY POINT" << std::endl;
  return std::make_tuple(at::Tensor{}, at::Tensor{}, at::Tensor{}, at::Tensor{});
}


} // namespace pytorch_flash
