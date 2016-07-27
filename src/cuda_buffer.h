#ifndef CUDA_CUDA_BUFFER_H
#define CUDA_CUDA_BUFFER_H

#include <memory>

template <typename T>
struct CudaDeleter 
{
    void operator()(T * ptr) const;
};

template <typename T>
using DeviceBuffer = std::unique_ptr<T, CudaDeleter<T>>;

#endif // CUDA_CUDA_BUFFER_H
