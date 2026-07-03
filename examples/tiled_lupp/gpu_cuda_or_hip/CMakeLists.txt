# GPU examples: a large batch of independent systems, one thread per
# system. Single sources in the common CUDA/HIP dialect (see
# examples/common/gpu_runtime.hpp), compiled by whichever of the two
# toolchains is available. Each binary returns 77 when no device is
# present, which ctest reports as a skipped test.
include(CheckLanguage)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(TDLS_GPU_LANGUAGE CUDA)
else()
    check_language(HIP)
    if(CMAKE_HIP_COMPILER)
        enable_language(HIP)
        set(TDLS_GPU_LANGUAGE HIP)
    endif()
endif()
if(TDLS_GPU_LANGUAGE)
    foreach(example implicit_ode_batch_gpu implicit_ode_batch_gpu_soa
                    integral_equation_batch_gpu integral_equation_batch_gpu_soa)
        if(TDLS_GPU_LANGUAGE STREQUAL "HIP")
            set_source_files_properties(${example}.cu PROPERTIES LANGUAGE HIP)
        endif()
        add_executable(tdls_example_tiledlupp_${example} ${example}.cu)
        target_link_libraries(tdls_example_tiledlupp_${example} PRIVATE tdls::tdls)
        target_include_directories(tdls_example_tiledlupp_${example}
                                   PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../common")
        set_target_properties(tdls_example_tiledlupp_${example} PROPERTIES CUDA_STANDARD 17
                                                                           HIP_STANDARD 17)
        add_test(NAME examples.tiledlupp.${example} COMMAND tdls_example_tiledlupp_${example})
        set_tests_properties(examples.tiledlupp.${example} PROPERTIES LABELS "examples"
                                                                      SKIP_RETURN_CODE 77)
    endforeach()
else()
    message(STATUS "tdls: no CUDA or HIP compiler, GPU examples skipped")
endif()
