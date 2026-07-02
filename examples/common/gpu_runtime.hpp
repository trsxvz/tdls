/// \file
/// \brief Portability aliases between the CUDA and HIP runtime APIs,
/// letting every GPU example keep a single source compiled as either
/// language.
/// \author Tristan Chenaille
///
/// Device code (kernels, launch syntax, math functions) is identical in
/// CUDA and HIP; only the host-side runtime API names differ. The
/// wrappers below cover the few calls the examples need.

#ifndef TDLS_EXAMPLES_GPU_RUNTIME_HPP
#define TDLS_EXAMPLES_GPU_RUNTIME_HPP

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#if defined(__HIP__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
using gpuError_t                              = hipError_t;
using gpuMemcpyKind                           = hipMemcpyKind;
constexpr gpuError_t gpuSuccess               = hipSuccess;
constexpr gpuMemcpyKind gpuMemcpyHostToDevice = hipMemcpyHostToDevice;
constexpr gpuMemcpyKind gpuMemcpyDeviceToHost = hipMemcpyDeviceToHost;
#define TDLS_EXAMPLES_GPU_API(name) hip##name
#else
#include <cuda_runtime.h>
using gpuError_t                              = cudaError_t;
using gpuMemcpyKind                           = cudaMemcpyKind;
constexpr gpuError_t gpuSuccess               = cudaSuccess;
constexpr gpuMemcpyKind gpuMemcpyHostToDevice = cudaMemcpyHostToDevice;
constexpr gpuMemcpyKind gpuMemcpyDeviceToHost = cudaMemcpyDeviceToHost;
#define TDLS_EXAMPLES_GPU_API(name) cuda##name
#endif

/// \brief Allocates device memory.
/// \tparam T element type
/// \param[out] ptr   receives the device pointer
/// \param[in]  bytes allocation size in bytes
/// \return the runtime error code
template<typename T>
gpuError_t gpuMalloc(T** ptr, const std::size_t bytes) {
    return TDLS_EXAMPLES_GPU_API(Malloc)(reinterpret_cast<void**>(ptr), bytes);
}

/// \brief Copies memory between host and device.
/// \param[out] dst   destination pointer
/// \param[in]  src   source pointer
/// \param[in]  bytes copy size in bytes
/// \param[in]  kind  copy direction
/// \return the runtime error code
inline gpuError_t gpuMemcpy(void* dst, const void* src, const std::size_t bytes,
                            const gpuMemcpyKind kind) {
    return TDLS_EXAMPLES_GPU_API(Memcpy)(dst, src, bytes, kind);
}

/// \brief Releases device memory.
/// \param[in] ptr device pointer to release
/// \return the runtime error code
inline gpuError_t gpuFree(void* ptr) {
    return TDLS_EXAMPLES_GPU_API(Free)(ptr);
}

/// \brief Waits for every pending device operation.
/// \return the runtime error code
inline gpuError_t gpuDeviceSynchronize() {
    return TDLS_EXAMPLES_GPU_API(DeviceSynchronize)();
}

/// \brief Returns the last error raised by the runtime.
/// \return the runtime error code
inline gpuError_t gpuGetLastError() {
    return TDLS_EXAMPLES_GPU_API(GetLastError)();
}

/// \brief Aborts with a message when a runtime call failed.
/// \param[in] err  error code returned by the call
/// \param[in] file source file of the call
/// \param[in] line source line of the call
inline void gpu_check(const gpuError_t err, const char* file, const int line) {
    if (err != gpuSuccess) {
        std::fprintf(stderr, "%s:%d: %s\n", file, line, TDLS_EXAMPLES_GPU_API(GetErrorString)(err));
        std::exit(1);
    }
}

/// \brief Wraps a runtime call with an abort-on-error check.
#define GPU_CHECK(call) gpu_check((call), __FILE__, __LINE__)

/// \brief Exit code registered in ctest as a skipped test, returned when
/// no device is present.
constexpr int gpu_skip_code = 77;

/// \brief Tells whether at least one device is usable.
/// \return true when a device can be selected
inline bool gpu_device_available() {
    int count = 0;
    return TDLS_EXAMPLES_GPU_API(GetDeviceCount)(&count) == gpuSuccess && count > 0;
}

#endif // TDLS_EXAMPLES_GPU_RUNTIME_HPP
