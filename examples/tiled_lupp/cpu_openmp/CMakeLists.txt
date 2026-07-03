# OpenMP examples: a moderate batch of independent systems, one thread
# iteration per system. Built only when an OpenMP runtime is available.
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    foreach(example implicit_ode_batch_omp integral_equation_batch_omp)
        tdls_add_example(tiledlupp ${example})
        target_link_libraries(tdls_example_tiledlupp_${example} PRIVATE OpenMP::OpenMP_CXX)
    endforeach()
else()
    message(STATUS "tdls: OpenMP not found, OpenMP examples skipped")
endif()
