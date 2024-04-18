#include <ATen/TensorGeometry.h>

#include <limits>
#include <cstddef>
#include <iostream>

namespace at {

// See TensorGeometry.h on why this is useful now that we cache is_contiguous.
template <typename T>
bool _geometry_is_contiguous(ArrayRef<T> sizes, ArrayRef<T> strides) {
 // NO NEED TO PRINT ANY OF THIS SINCE THIS FUNCTION AS ONLY EVER RETURNED TRUE
  // sample output
  //_GEOMETRY_IS_CONTIGUOUS
  //sizes  : [2]
  //strides: [1]
  //
  // The value in sizes varies, but strides is always the same i.e. i didn't get anything useful
  // by printing this stuff out
  //std::cout << "_GEOMETRY_IS_CONTIGUOUS" << std::endl;
  //std::cout << "sizes  : " << sizes << std::endl;
  //std::cout << "strides: " << strides << std::endl;



  assert(!overflows<std::int64_t>(sizes.size()));
  auto dim = static_cast<std::int64_t>(sizes.size());
  T expected_stride = 1;
  bool contig_if_nonempty = true;
  for (int64_t i = dim - 1; i >= 0; i--) {
    // IF THE SIZE OF A PARTICULAR DIMENSION IS 0, RETURN TRUE
    if (sizes[i] == 0) {
      return true;
    }
    if (contig_if_nonempty) {
	// if current size does NOT equal 1
	// AND if current stride does not equal expected stride (1 at first)
      if (sizes[i] != 1 && strides[i] != expected_stride) {
	    std::cout << "TENSOR_GEOMERY SETTING CONTIG_IF_NONEMTPY TO FALSE!: " << contig_if_nonempty << std::endl;
		// TENSOR IS NOT CONTIGUOS
        contig_if_nonempty = false;
      }
	  // MULTIPLY EXPECTED STRIDE(Z) BY CURRENT SIZE
      expected_stride *= sizes[i];
    }
  }
  // THIS HAS ONLY EVER BEEN TRUE
  //std::cout << "CONTIG IF NONEMPTY: " << contig_if_nonempty << std::endl;
  if(!contig_if_nonempty)
  {
    std::cout << "_GEOMETRY_IS_CONTIGUOUS: FOUND IS_CONGIG TO BE FALSE: " << contig_if_nonempty << std::endl;
    std::cout << "sizes  : " << sizes << std::endl;
    std::cout << "strides: " << strides << std::endl;

  }
  return contig_if_nonempty;
}

bool geometry_is_contiguous(IntArrayRef sizes, IntArrayRef strides) {
  return _geometry_is_contiguous(sizes, strides);
}

bool TensorGeometry::is_contiguous() const {
  if (numel_ == 0) {
    return true;
  }
  return at::_geometry_is_contiguous<c10::SymInt>(sizes_, strides_);
}

} // namespace at
