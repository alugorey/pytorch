#include <ATen/native/transformers/hip/flash_attn/flash_common_hip.hpp>
#include <ATen/native/transformers/hip/flash_attn/ck/me_ck_api.h>

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
  std::optional<at::Tensor> & alibi_slopes_) {


  // These normally get passed in as std::nullopt so just set to -1
  // Note: See attention.cu ~line 928 and line 729
  const int non_null_window_left  = -1;
  const int non_null_window_right = -1;

  TORCH_CHECK(
    cu_seqlens_q.has_value() == cu_seqlens_k.has_value(),
    "cu_seqlens_q and cu_seqlens_k must be both set or both not set");


  // need to pass attn_bias to both of these
  if(!cu_seqlens_q.has_value()){
    return mha_fwd_ck(
      q,                     // q
      k,                     // k
      v,                     // v
      out_,                  // opt(out_)
      alibi_slopes_,         // opt(alibi_slopes)
      p_dropout,             // p_dropout
      scale.value(),         // opt(softmax_scale)
      is_causal.value(),     // opt(is_causal)
      non_null_window_left,  // window_size_left
      non_null_window_right, // window_size_right
      false,                 // return_softmax/return_debug_mask
      gen_,                  // gen
      attn_bias_);           // attn_bias
  } else {
    // max sequence lengths are now at T.size(1) since q,k,v were all transposed
    // in _scaled_dot_product_efficient_attention_cuda
    const int64_t max_seqlen_q = q.size(1);
    const int64_t max_seqlen_k = k.size(1);

    return mha_varlen_fwd_ck(
      q,                     // q
      k,                     // k
      v,                     // v
      out_,                  // opt(out)
      cu_seqlens_q.value(),  // cu_seqlens_q
      cu_seqlens_k.value(),  // cu_seqlens_k
      seqused_k_,            // opt(seqused_k)
      alibi_slopes_,         // opt(alibi_slopes)
      max_seqlen_q,          // max_seqlen_q
      max_seqlen_k,          // max_seqlen_k
      p_dropout,             // p_dropout
      scale.value(),         // softmax_scale
      false,                 // zero_tensors
      is_causal.value(),     // is_causal
      non_null_window_left,  // window_size_left
      non_null_window_right, // window_size_right
      false,                 // return_softmax/return_debug_mask
      gen_,                  // gen
      attn_bias_);           // attn_bias
  }
}

} // namespace pytorch_flash
