#ifndef CUDA_CUDA_BUFFER_H
#define CUDA_CUDA_BUFFER_H

#include <memory>

template <typename T>
struct CudaDeviceMemDeleter
{
    void operator()(T * ptr) const;
};

template <typename T>
struct CudaHostMemDeleter
{
    void operator()(T * ptr) const;
};

template <typename T>
using DeviceBuffer = std::unique_ptr<T, CudaDeviceMemDeleter<T>>;

template <typename T>
using HostBuffer = std::unique_ptr<T, CudaHostMemDeleter<T>>;

#endif // CUDA_CUDA_BUFFER_H
