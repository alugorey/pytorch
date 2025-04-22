include (CMakePrintHelpers)

# Generate AITER/CK Asm code
execute_process(
    COMMAND python3 ${CMAKE_CURRENT_LIST_DIR}/../../../../../../../../../third_party/aiter/csrc/py_itfs_cu/fmha_bwd_pre_post_kernel.py --receipt 1 --output_dir ${CMAKE_CURRENT_LIST_DIR}
    RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
    message( FATAL_ERROR "Faled to generate FAv3 CK Kernels")
endif()
