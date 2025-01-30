#include <ATen/native/transformers/hip/flash_attn/flash_common_hip.hpp>


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
  const float p_dropout,
  const float softmax_scale,
  const bool return_dropout_randval,
  const std::optional<bool> is_causal,
  const std::optional<double> scale,
  const std::optional<at::Tensor>& attn_bias,
  const std::optional<at::Tensor>& out_,
  const std::optional<at::Tensor>& seqstart_q,
  const std::optional<at::Tensor>& seqstart_k,
  std::optional<at::Generator> gen_) {

  // need to pass attn_bias to both of these
  if(!seqstart_q.has_value()){
    return mha_fwd_ck(
      q,            // q
      k,            // k
      v,            // v
      out_,         // opt(out_)
      std::nullopt, // opt(alibi_slopes)
      p_dropout,    // p_dropout
      scale,        // opt(softmax_scale)
      is_causal,    // opt(is_causal)
      std::nullopt, // window_size_left
      std::nullopt, // window_size_right
      false,        // return_softmax/return_debug_mask
      gen_);        // gen
  } else {
    // max sequence lengths are now at T.size(1) since q,k,v were all transposed
    // in _scaled_dot_product_efficient_attention_cuda
    const int64_t max_seqlen_q = q.size(1);
    const int64_t max_seqlen_k = k.size(1);
    return mha_varlen_fwd_ck(
      q,            // q
      k,            // k
      v,            // v
      out_,         // opt(out)
      std::nullopt, // cu_seqlens_q
      std::nullopt, // cu_seqlens_k
      std::nullopt, // opt(seqused_k)
      std::nullopt, // opt(alibi_slopes)
      max_seqlen_q, // max_seqlen_q
      max_seqlen_k, // max_seqlen_k
      p_dropout     // p_dropout
      scale,        // softmax_scale
      false,        // zero_tensors
      is_causal,    // is_causal
      std::nullopt, // window_size_left
      std::nullopt, // window_size_right
      false,        // return_softmax/return_debug_mask
      gen_);// gen
  }
}

} // namespace pytorch_flash
