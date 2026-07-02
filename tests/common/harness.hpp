#ifndef TDLS_TESTS_COMMON_HARNESS_HPP
#define TDLS_TESTS_COMMON_HARNESS_HPP



/// \file
/// \brief Minimal self-contained test harness of the tdls test suites.
/// \author Tristan Chenaille
///
/// Test cases register themselves at static-initialization time through
/// TDLS_TEST_CASE and run from the main() emitted by TDLS_TEST_MAIN. Each
/// case prints one [PASS]/[FAIL] line; failing checks print their
/// expression, values and location. The executable returns the number of
/// failed cases (0 on success), which is what ctest consumes. An optional
/// command-line argument keeps only the cases whose name contains it:
///
///   ./tdls_test_tiledlupp_oracle_static N=12
///
/// The macro vocabulary (TEST_CASE / CHECK) matches mainstream frameworks
/// on purpose, so the suites could migrate with little churn if the
/// project ever adopts one.



#include <cstddef>
#include <cstdio>
#include <cstring>
#include <vector>



namespace tdls_tests {



/// \brief A registered test case: display name and function to run.
struct TestCase {
    const char* name;
    void (*run)();
};

/// \return the test-case registry of the current executable
inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> cases;
    return cases;
}

/// \return the failed-check counter of the case currently running
inline int& current_case_failures() {
    static int failures = 0;
    return failures;
}

/// \brief Registers a test case at static-initialization time.
struct Registrar {
    /// \param[in] name display name of the case
    /// \param[in] run  function holding the checks
    Registrar(const char* name, void (*run)()) {
        registry().push_back({name, run});
    }
};

/// \brief Records a boolean check; prints expression and location on
/// failure.
/// \param[in] ok         check outcome
/// \param[in] expression stringized condition
/// \param[in] file       source file of the check
/// \param[in] line       source line of the check
inline void check(const bool ok, const char* expression, const char* file, const int line) {
    if (!ok) {
        ++current_case_failures();
        std::printf("       check failed: %s  (%s:%d)\n", expression, file, line);
    }
}

/// \brief Records an upper-bound check; prints both sides on failure.
/// \param[in] value      measured value
/// \param[in] bound      inclusive upper bound
/// \param[in] expression stringized comparison
/// \param[in] file       source file of the check
/// \param[in] line       source line of the check
inline void check_le(const double value, const double bound, const char* expression,
                     const char* file, const int line) {
    if (!(value <= bound)) {
        ++current_case_failures();
        std::printf("       check failed: %s, measured %.6e  (%s:%d)\n", expression, value, file,
                    line);
    }
}

/// \brief Records a bitwise-equality check between two element ranges;
/// prints the index of the first differing element on failure.
/// \param[in] a          first range
/// \param[in] b          second range
/// \param[in] count      number of elements compared
/// \param[in] expression stringized comparison
/// \param[in] file       source file of the check
/// \param[in] line       source line of the check
template<typename T>
inline void check_bitwise(const T* a, const T* b, const std::size_t count, const char* expression,
                          const char* file, const int line) {
    if (std::memcmp(a, b, count * sizeof(T)) != 0) {
        ++current_case_failures();
        std::size_t first = 0;
        while (first < count && std::memcmp(a + first, b + first, sizeof(T)) == 0)
            ++first;
        std::printf("       check failed: %s not bitwise equal (first difference at element %zu "
                    "of %zu)  (%s:%d)\n",
                    expression, first, count, file, line);
    }
}

/// \brief Runs every registered case, optionally filtered by substring,
/// and prints one summary line.
/// \param[in] argc argument count of main()
/// \param[in] argv argument vector of main(); argv[1] is the filter
/// \return the number of failed cases
inline int run_all(const int argc, char* const* argv) {
    const char* filter = argc > 1 ? argv[1] : nullptr;
    int ran            = 0;
    int failed         = 0;
    for (const auto& c : registry()) {
        if (filter != nullptr && std::strstr(c.name, filter) == nullptr) continue;
        ++ran;
        current_case_failures() = 0;
        c.run();
        if (current_case_failures() == 0) {
            std::printf("[PASS] %s\n", c.name);
        } else {
            std::printf("[FAIL] %s (%d failed check(s))\n", c.name, current_case_failures());
            ++failed;
        }
    }
    if (filter != nullptr && ran == 0) {
        std::printf("no test case matches the filter '%s'\n", filter);
        return 1;
    }
    std::printf("%d/%d case(s) passed\n", ran - failed, ran);
    return failed;
}



} // namespace tdls_tests



/// \brief Declares and registers a test case; the following block is its
/// body.
#define TDLS_TEST_CASE(name)            TDLS_TEST_CASE_IMPL_A(name, __COUNTER__)
#define TDLS_TEST_CASE_IMPL_A(name, id) TDLS_TEST_CASE_IMPL_B(name, id)
#define TDLS_TEST_CASE_IMPL_B(name, id)                                                            \
    static void tdls_test_function_##id();                                                         \
    static const ::tdls_tests::Registrar tdls_test_registrar_##id(name, &tdls_test_function_##id); \
    static void tdls_test_function_##id()

/// \brief Checks a boolean condition.
#define TDLS_CHECK(condition) ::tdls_tests::check((condition), #condition, __FILE__, __LINE__)

/// \brief Checks value <= bound, reporting the measured value on failure.
#define TDLS_CHECK_LE(value, bound)                                                                \
    ::tdls_tests::check_le((value), (bound), #value " <= " #bound, __FILE__, __LINE__)

/// \brief Checks that count elements of a and b are bitwise identical.
#define TDLS_CHECK_BITWISE(a, b, count)                                                            \
    ::tdls_tests::check_bitwise((a), (b), (count), #a " == " #b, __FILE__, __LINE__)

/// \brief Emits the main() of a test suite executable.
#define TDLS_TEST_MAIN                                                                             \
    int main(int argc, char** argv) {                                                              \
        return ::tdls_tests::run_all(argc, argv);                                                  \
    }



#endif // TDLS_TESTS_COMMON_HARNESS_HPP
