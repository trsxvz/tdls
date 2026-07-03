# Vanilla test suites of the TiledLUpp solvers: one executable per suite, each
# registered in ctest under the "solvers" label. The suites use the
# homemade harness of tests/common (no external test framework) and may
# use C++20 even though the library floor is C++17.
set(TDLS_TILEDLUPP_TEST_SUITES
    oracle_static
    oracle_dynamic
    static_vs_dynamic
    residencies
    layouts
    entry_points
    inplace_paths
    tile_sizes
    singular
    config_knobs
    adaptors
    cross_full
    dynamic_edges)

foreach(suite IN LISTS TDLS_TILEDLUPP_TEST_SUITES)
    add_executable(tdls_test_tiledlupp_${suite} tiled_lupp/test_${suite}.cpp)
    target_link_libraries(tdls_test_tiledlupp_${suite} PRIVATE tdls::tdls)
    target_include_directories(tdls_test_tiledlupp_${suite} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../common")
    target_compile_features(tdls_test_tiledlupp_${suite} PRIVATE cxx_std_20)
    if(NOT MSVC)
        target_compile_options(tdls_test_tiledlupp_${suite} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(NAME solvers.tiledlupp.${suite} COMMAND tdls_test_tiledlupp_${suite})
    set_tests_properties(solvers.tiledlupp.${suite} PROPERTIES LABELS "solvers")
endforeach()

# Negative compilation tests of the configuration contracts: each
# target must fail to build with the scalar-type diagnostic. ctest
# builds them on demand and passes only when the diagnostic appears,
# so a contract silently dropped turns these tests red.
set(TDLS_TILEDLUPP_REJECT_CASES
    reject_config_float_thresholds
    reject_config_double_thresholds
    reject_config_eps_above_threshold)
set(TDLS_TILEDLUPP_REJECT_DIAGNOSTICS
    "the config thresholds must have the scalar type T"
    "the config thresholds must have the scalar type T"
    "singular_eps must not exceed oot_threshold")

foreach(case diagnostic IN ZIP_LISTS TDLS_TILEDLUPP_REJECT_CASES
                                     TDLS_TILEDLUPP_REJECT_DIAGNOSTICS)
    add_executable(tdls_test_tiledlupp_${case} EXCLUDE_FROM_ALL tiled_lupp/${case}.cpp)
    target_link_libraries(tdls_test_tiledlupp_${case} PRIVATE tdls::tdls)
    target_compile_features(tdls_test_tiledlupp_${case} PRIVATE cxx_std_20)
    add_test(NAME solvers.tiledlupp.${case}
             COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target
                     tdls_test_tiledlupp_${case} --config $<CONFIG>)
    set_tests_properties(solvers.tiledlupp.${case} PROPERTIES LABELS "solvers"
                                                              PASS_REGULAR_EXPRESSION
                                                              "${diagnostic}")
endforeach()
