#include "constants.h"
#include "ffm_predictor.h"
#include "cuda_utils.h"
#include "model.h"

__constant__ float cNormalizationFactor[1];       // 1.0 / numFields;
__constant__ int cRowSize[1];                     // numFields * FactorSize

// Partially computes t (each thread block computes the inner sum for one field) - for prediction, batch mode
__global__ void batchFfmInnerSumKernel(const float *__restrict__ weights, const int *__restrict__ input, float *__restrict__ fieldSums,
                                       const int numFields)
{
    const int fieldIdx1 = threadIdx.x;
    const int fieldIdx2 = blockIdx.x;

    const int batchIdx = blockIdx.y;
    const int batchInputOffset = (numFields + 1) * batchIdx;
    const int rowSize = *cRowSize;

    CUDA_ASSERT(fieldIdx1 < numFields);
    CUDA_ASSERT(fieldIdx2 < numFields);

    float sum = 0.0f;

    if (fieldIdx2 > fieldIdx1) {
        const int j1 = input[batchInputOffset + fieldIdx1];
        const int j2 = input[batchInputOffset + fieldIdx2];

        const int offset1 = j1 * rowSize + fieldIdx2 * FactorSize;
        const int offset2 = j2 * rowSize + fieldIdx1 * FactorSize;

        const float4 W1 = cub::ThreadLoad<cub::LOAD_DEFAULT>((float4 *) (weights + offset1));
        const float4 W2 = cub::ThreadLoad<cub::LOAD_DEFAULT>((float4 *) (weights + offset2));

        sum += W1.x * W2.x;
        sum += W1.y * W2.y;
        sum += W1.z * W2.z;
        sum += W1.w * W2.w;
    }

    typedef cub::BlockReduce<float, MaxPredictBlockSize> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    float aggregate = BlockReduce(tempStorage).Sum(sum);

    if (threadIdx.x == 0) {
        fieldSums[batchIdx * numFields + blockIdx.x] = aggregate * *cNormalizationFactor;
    }
}

// Computes outer sum (t) and and applies logit function - for prediction
__global__ void batchSigmoidKernel(const float *__restrict__ fieldSums, float *__restrict__ predictionResults, const int numFields)
{
    CUDA_ASSERT(threadIdx.x < numFields);

    typedef cub::BlockReduce<float, MaxPredictBlockSize> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    float t = BlockReduce(tempStorage).Sum(fieldSums[blockIdx.x * numFields + threadIdx.x]);

    if (threadIdx.x == 0) {
        const float p = 1.0f / (1.0f + expf(-t));
        CUDA_ASSERT_FIN(p);
        CUDA_ASSERT(p <= 1.0);
        CUDA_ASSERT(p >= 0.0);
        predictionResults[blockIdx.x] = p;
    }
}

FFMPredictor::FFMPredictor(DeviceBuffer<float> const * dWeights, int maxBatchSize, Model const & model)
        : dWeights(dWeights),
          maxPredictBatchSize(maxBatchSize),
          numFields(model.numFields)
{
    dXYPredictInputBuffer = cuda_utils.malloc<int>(maxPredictBatchSize * (numFields + 1));
    dPredictFieldSums = cuda_utils.malloc<float>(maxPredictBatchSize * numFields);
    dPredictResultsBuffer = cuda_utils.malloc<float>(maxPredictBatchSize * (numFields + 1));

    CHECK_ERR(cudaGetLastError());
    CHECK_ERR(cudaDeviceSynchronize());

    cuda_utils.memset(dPredictFieldSums, 0, maxPredictBatchSize * numFields);
    cuda_utils.memcpyToSymbol(cNormalizationFactor, model.normalizationFactor);
    cuda_utils.memcpyToSymbol(cRowSize, numFields * FactorSize);
}

void FFMPredictor::predict(HostBuffer<int> const & hXYBatchBuffer, int batchSize, HostBuffer<float> & hPredictResults)
{
    assert(batchSize <= maxPredictBatchSize);

    cuda_utils.memcpy(dXYPredictInputBuffer, hXYBatchBuffer.get(), batchSize * (numFields + 1));

    batchFfmInnerSumKernel<<<dim3(numFields, batchSize, 1), numFields>>>(dWeights->get(), dXYPredictInputBuffer.get(),
                                                                         dPredictFieldSums.get(), numFields);
    batchSigmoidKernel<<<batchSize, numFields>>>(dPredictFieldSums.get(), dPredictResultsBuffer.get(), numFields);

    cuda_utils.memcpy(hPredictResults.get(), dPredictResultsBuffer, batchSize);
}
