#include "mha_bwd.h"
#include <iostream>

namespace aiter {
mha_bwd_traits get_mha_bwd_traits(int head_size_q,
                                  int head_size_v,
                                  std::string dtype,
                                  bool is_group_mode,
                                  const mask_info &mask,
                                  bias_enum bias_type,
                                  bool has_dbias,
                                  bool has_dropout,
                                  bool is_store_randval,
                                  bool deterministic,
                                  bool use_ext_asm,
                                  bool is_v3_atomic_fp32,
                                  int how_v3_bf16_cvt)
{
    return mha_bwd_traits(head_size_q,
                          head_size_v,
                          dtype,
                          is_group_mode,
                          mask,
                          bias_type,
                          has_dbias,
                          has_dropout,
                          is_store_randval,
                          deterministic,
                          use_ext_asm,
                          is_v3_atomic_fp32,
                          how_v3_bf16_cvt);
}

// share with varlen(group mode) api
float mha_bwd(mha_bwd_args args,
              const ck_tile::stream_config& stream_config,
              std::string q_dtype_str,
              bool is_group_mode,
              mask_info mask,
              bias_enum bias_type,
              bool has_dbias,
              bool is_store_randval,
              bool deterministic,
              bool use_ext_asm,
              bool is_v3_atomic_fp32,
              int how_v3_bf16_cvt)
{
    int head_size_q = args.hdim_q;
    int head_size_v = args.hdim_v;
    bool has_dropout = args.p_drop > 0;
    // bool enable_ailib = args.alibi_slopes_ptr == nullptr;
    auto traits = get_mha_bwd_traits(head_size_q, 
                                     head_size_v,
                                     q_dtype_str,
                                     is_group_mode,
                                     mask,
                                     bias_type,
                                     has_dbias,
                                     has_dropout,
                                     is_store_randval,
                                     deterministic,
                                     use_ext_asm,
                                     is_v3_atomic_fp32,
                                     how_v3_bf16_cvt);
    float t = -1;
    std::cout << "Inside mha_bwd.hip. about to call fmha_bwd or fmha_bwd_v3" << std::endl;
    std::cout << "test bias type?: " << int(bias_type) << std::endl;
    t = fmha_bwd_v3(traits, args, stream_config);
    std::cout << "came back from v3, about to check what happened current t: " << t << std::endl;
    if (t == -1) {
        std::cout << "fmha_bwd_v3 returned a bad value(" << t << ") calling fmha_bwd" << std::endl;
        t = fmha_bwd(traits, args, stream_config);
        std::cout << "Came back from fmha_bwd with a return value of (" << t << ")" << std::endl;
    }

    std::cout << "Coming back from either fmha_bwd or fmha_bwd_v3" << std::endl;
    return t;
}
} // namespace aiter

