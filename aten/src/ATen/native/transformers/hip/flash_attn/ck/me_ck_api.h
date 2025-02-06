#pragma once
#include <cstddef>

#include <ATen/core/Tensor.h>


namespace pytorch_flash {

std::tuple<
    at::Tensor, // output
    at::Tensor, // q
    at::Tensor, // k
    at::Tensor, // v
    at::Tensor, // lse
    at::Tensor, // seed
    at::Tensor, // offset
    at::Tensor> // dropout randval
mem_eff_forward_ck(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    float p_dropout,
    bool return_dropout_randval,
    std::optional<bool> is_causal,
    std::optional<float> scale,
    const std::optional<at::Tensor>& attn_bias_,
    std::optional<at::Tensor>& out_,
    const std::optional<at::Tensor>& cu_seqlens_q,
    const std::optional<at::Tensor>& cu_seqlens_k,
    const std::optional<at::Tensor>& seqstart_q,
    const std::optional<at::Tensor>& seqstart_k,
    std::optional<at::Generator> gen_,
    std::optional<at::Tensor>& seqused_k_,
    std::optional<at::Tensor>& alibi_slopes_
);


} // namespace pytorch_flash
