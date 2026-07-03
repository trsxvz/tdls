# Examples double as living documentation and as tests: each one is
# compiled at the C++17 floor of the library, executed by ctest under
# the "examples" label, and self-checks its result through its exit
# code. The tree is split by solver family, then by execution scale
# (cpu_sequential, cpu_openmp, gpu_cuda_or_hip); examples/common holds
# the pieces shared by every family.
function(tdls_add_example family example)
    add_executable(tdls_example_${family}_${example} ${example}.cpp)
    target_link_libraries(tdls_example_${family}_${example} PRIVATE tdls::tdls)
    target_compile_features(tdls_example_${family}_${example} PRIVATE cxx_std_17)
    if(NOT MSVC)
        target_compile_options(tdls_example_${family}_${example} PRIVATE -Wall -Wextra -Wpedantic
                                                                         -Werror)
    endif()
    add_test(NAME examples.${family}.${example} COMMAND tdls_example_${family}_${example})
    set_tests_properties(examples.${family}.${example} PROPERTIES LABELS "examples")
endfunction()

add_subdirectory(tiled_lupp)
