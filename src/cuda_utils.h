#ifndef CUDA_CUDA_UTILS_H
#define CUDA_CUDA_UTILS_H

// contains various CUDA helper functions related to memory management, etc.

#include "cuda_buffer.h"

#include <cstdio>
#include <cmath>
#include <cassert>
#include <algorithm>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define ENABLE_ASSERTIONS 0

#if ENABLE_ASSERTIONS
#define CUDA_ASSERT_FIN(x) \
    do {\
        assert(!isnan(x));\
        assert(isfinite(x));\
    } while (false)
#define CUDA_ASSERT assert
#else
#define CUDA_ASSERT(x) do {} while (0)
#define CUDA_ASSERT_FIN(x) do {} while (0)
#endif

#define CHECK_ERR(code) __CHECK_ERR(code, __FILE__, __LINE__, __PRETTY_FUNCTION__)

#define __CHECK_ERR(code, file, line, func) do \
{ \
    cudaError_t err = code; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s [%s:%d in %s]\n", cudaGetErrorString(err),  file, line, func); \
        exit(33); \
    } \
} while(0) 

inline int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

#define cuda_utils __cuda_utils(__FILE__, __LINE__, __PRETTY_FUNCTION__)

struct __cuda_utils
{
    const char * file;
    const int line;
    const char * func;

    inline __cuda_utils(const char * file, const int line, const char * func)
            : file(file),
              line(line),
              func(func)
    {
    }

    template <typename T>
    inline DeviceBuffer<T> malloc(int64_t numElems)
    {
        T *result;
        __CHECK_ERR(cudaMalloc((void ** )&result, numElems * sizeof(T)), file, line, func);
        return DeviceBuffer<T>(result);
    }

    template <typename T>
    inline void free(T * ptr)
    {
        __CHECK_ERR(cudaFree(ptr), file, line, func);
    }

    template <typename T>
    inline HostBuffer<T> hostMalloc(int64_t numElems)
    {
        T *result;
        __CHECK_ERR(cudaMallocHost((void ** )&result, numElems * sizeof(T)), file, line, func);
        return HostBuffer<T>(result);
    }

    template <typename T>
    inline T * hostMallocMapped(int64_t numElems)
    {
        T * result;
        T * result_map;
        __CHECK_ERR(cudaHostAlloc(&result, numElems * sizeof(T), cudaHostAllocMapped), file, line, func);
        __CHECK_ERR(cudaHostGetDevicePointer(&result_map, result, 0), file, line, func);
        assert(result_map == result); // this should be always true because of UVA

        return result;
    }

    template <typename T>
    inline T * hostMallocManaged(int64_t numElems)
    {
        T * result;
        __CHECK_ERR(cudaMallocManaged(&result, numElems * sizeof(T)), file, line, func);
        return result;
    }

    template <typename T>
    inline void hostFree(T * ptr)
    {
        __CHECK_ERR(cudaFreeHost(ptr), file, line, func);
    }

    template <typename T>
    inline void memcpy(DeviceBuffer<T> & dst, T const * src, int64_t numElems)
    {
        __CHECK_ERR(cudaMemcpy(dst.get(), src, numElems * sizeof(T), cudaMemcpyHostToDevice), file, line, func);
    }

    template <typename T>
    inline void memcpy(DeviceBuffer<T> & dst, HostBuffer<T> const & src, int64_t numElems)
    {
        __CHECK_ERR(cudaMemcpy(dst.get(), src.get(), numElems * sizeof(T), cudaMemcpyHostToDevice), file, line, func);
    }

    template <typename T>
    inline void memcpy(T * dst, DeviceBuffer<T> const & src, int64_t numElems)
    {
        __CHECK_ERR(cudaMemcpy(dst, src.get(), numElems * sizeof(T), cudaMemcpyDeviceToHost), file, line, func);
    }

    template <typename T, typename U>
    inline void memcpyToSymbol(const T & dst, const U src)
    {
        __CHECK_ERR(cudaMemcpyToSymbol(dst, &src, sizeof(U)), file, line, func);
    }

    template <typename T, typename U>
    inline void memcpyToSymbol2(const T & dst, const U src1, const U src2)
    {
        U tmp[2] = { src1, src2 };
        __CHECK_ERR(cudaMemcpyToSymbol(dst, &tmp, sizeof(U) * 2), file, line, func);
    }

    template <typename T>
    inline void memset(DeviceBuffer<T> & dst, char byte, int64_t numElems)
    {
        __CHECK_ERR(cudaMemset(dst.get(), byte, numElems * sizeof(T)), file, line, func);
    }
};

#endif // CUDA_UTILS_H
