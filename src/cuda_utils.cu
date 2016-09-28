#include "cuda_utils.h"

template <typename T>
void CudaDeviceMemDeleter<T>::operator()(T * ptr) const
{
    if (ptr) {
        cuda_utils.free(ptr);
    }
}

template <typename T>
void CudaHostMemDeleter<T>::operator()(T * ptr) const
{
    if (ptr) {
        cuda_utils.hostFree(ptr);
    }
}

template class CudaDeviceMemDeleter<int>;
template class CudaDeviceMemDeleter<float>;

template class CudaHostMemDeleter<int>;
template class CudaHostMemDeleter<float>;
