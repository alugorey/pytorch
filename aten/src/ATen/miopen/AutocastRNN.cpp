#include <ATen/ATen.h>
#include <ATen/autocast_mode.h>
#include <ATen/cuda/CUDAConfig.h>




namespace at {
namespace autocast {


/**********************************************************************
Autocast wrapper for MIOpen RNNs
**********************************************************************/
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
miopen_rnn(const Tensor & input_r,
		   TensorList weight,
		   int64_t weight_stride0,
		   const Tensor & hx,
		   const c10::optional<Tensor>& cx_opt,
		   int64_t fn_mode,
		   int64_t fn_hidden_size,
		   int64_t fn_num_layers,
		   bool batch_first,
		   double fn_dropout,
		   bool fn_train,
		   bool fn_bidirectional,
		   IntArrayRef fn_batch_sizes,
		   const c10::optional<Tensor>& fn_dropout_state_opt) {




}






// Register Autocast dispatch
namespace {
TORCH_LIBRARY_IMPL(aten, Autocast, m) {
  m.impl("miopen_rnn",
		 TORCH_FN((&at::autocast::miopen_rnn)));
}
} // anonymous namesspace

} // namespace autocast
} // namespace at
