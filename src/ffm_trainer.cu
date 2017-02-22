#include "constants.h"
#include "ffm_trainer.h"
#include "ffm_predictor.h"
#include "cuda_utils.h"

__constant__ float cLearningRate[1];
__constant__ float cL2Reg[1];
__constant__ float cNormalizationFactor[1];       // 1.0 / numFields;
__constant__ float cScaledNormalizationFactor[2]; // samplingFactor * normalizationFactor, idx 0 for y = -1, idx 1 for y = 1
__constant__ int cRowSize[1];                     // numFields * FactorSize

// Fills matrix with constant value
__global__ void fillKernel(float *__restrict__ matrix, const float value)
{
    matrix[blockIdx.x * blockDim.x + threadIdx.x] = value;
}

// Updates FFM weights
//
// Loss function and gradients:
//
//    y          := 1 or -1
//    t          := sum_j1(sum_j2(w[j1, f2] * w[j2, f1]))
//    p(y=1|x)   := 1 / (1 + exp(-t))
//
//    L          := p(y=1|x)^[y == 1] * (1 - p(y=1|x))^[y == -1] // likelihood
//               := 1 / (1 + exp(-yt))
//
//    LL         := log(L) // log-likelihood
//               := -log(1 + exp(-yt))
//
//    reg        := 1/2 * learningRate * sum_i(w_i^2) // regularization
//
//    loss       := LL + reg
//
//    -grad[w]   := -dLoss / dw == (dLL / dt * dt / dw) + dReg / dw
//    kappa      := -dLL / dt == -y / (1 + exp(yt)) ==  -y * exp(-yt) / (1 + exp(-yt))
//    dt / dw1   := w2
//    dReg / dw1 := learningRate * w1
//
//    -grad[w1]  := kappa * w2 + learningRate * w1
//
// AdaGrad update:
//
//    update[w, i]  := -grad[w] * learningRate / sqrt(sum(j=1..i-1, grad[w, j]^2))
//
__global__ void updateKernel(const float *__restrict__ fieldSums, float *__restrict__ weights, float *__restrict__ squaredGradsSum,
                             const int *__restrict__ input, const float y, int numFields)
{
    __shared__ float _normalizedKappa;

    const int fieldIdx1 = threadIdx.x / 4;
    const int d = threadIdx.x % 4;
    const int fieldIdx2 = blockIdx.x;
    const int rowSize = *cRowSize;

    // sum partially computed t (outer sum)

    typedef cub::BlockReduce<float, MaxUpdateBlockSize> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    const float t = BlockReduce(tempStorage).Sum(fieldSums[threadIdx.x], numFields);

    // compute kappa

    if (threadIdx.x == 0) {
        const float expYT = expf(y * t);
        const float kappa = -y / (1.0f + expYT);
        const float normalizationFactor = cScaledNormalizationFactor[y > 0];
        CUDA_ASSERT_FIN(expYT);
        CUDA_ASSERT_FIN(kappa);
        _normalizedKappa = kappa * normalizationFactor;
    }

    __syncthreads();

    // update weights

    if (fieldIdx2 > fieldIdx1) {
        const float normalizedKappa = _normalizedKappa;
        const float l2Reg = *cL2Reg;
        const float learningRate = *cLearningRate;

        CUDA_ASSERT_FIN(normalizedKappa);

        const int j1 = input[fieldIdx1];
        const int j2 = input[fieldIdx2];

        const int offset1 = j1 * rowSize + fieldIdx2 * FactorSize;
        const int offset2 = j2 * rowSize + fieldIdx1 * FactorSize;

        float *weight1Start = weights + offset1;
        float *weight2Start = weights + offset2;

        float *squaredGradsSum1Start = squaredGradsSum + offset1;
        float *squaredGradsSum2Start = squaredGradsSum + offset2;

        float weight1 = weight1Start[d];
        float weight2 = weight2Start[d];

        float prevSquaredGrads1Sum = squaredGradsSum1Start[d];
        float prevSquaredGrads2Sum = squaredGradsSum2Start[d];

        float regTerm1 = weight1 * l2Reg;
        float regTerm2 = weight2 * l2Reg;

        const float grad1 = regTerm1 + normalizedKappa * weight2;
        const float grad2 = regTerm2 + normalizedKappa * weight1;

        prevSquaredGrads1Sum += grad1 * grad1;
        prevSquaredGrads2Sum += grad2 * grad2;

        CUDA_ASSERT_FIN(prevSquaredGrads1Sum);
        CUDA_ASSERT_FIN(prevSquaredGrads2Sum);

        weight1 = weight1 - learningRate * grad1 * rsqrtf(prevSquaredGrads1Sum);
        weight2 = weight2 - learningRate * grad2 * rsqrtf(prevSquaredGrads2Sum);

        weight1Start[d] = weight1;
        weight2Start[d] = weight2;

        squaredGradsSum1Start[d] = prevSquaredGrads1Sum;
        squaredGradsSum2Start[d] = prevSquaredGrads2Sum;
    }
}

// Partially computes t (each thread block computes the inner sum for one field) - for updates
__global__ void ffmInnerSumKernel(const float *__restrict__ weights, const int *__restrict__ input, float *__restrict__ fieldSums)
{
    const int fieldIdx1 = threadIdx.x;
    const int fieldIdx2 = blockIdx.x;

    CUDA_ASSERT(fieldIdx1 < numFields);
    CUDA_ASSERT(fieldIdx2 < numFields);

    const int rowSize = *cRowSize;

    float sum = 0.0f;

    if (fieldIdx2 > fieldIdx1) {
        const int j1 = input[fieldIdx1];
        const int j2 = input[fieldIdx2];

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
        fieldSums[blockIdx.x] = aggregate * *cNormalizationFactor;
    }
}

FFMTrainer::FFMTrainer(Model const & model, float samplingFactor, int maxBatchSize, float l2Reg, float learningRate)
        : numFields(model.numFields),
          samplingFactor(samplingFactor),
          maxTrainBatchSize(maxBatchSize),
          weightsSize(HashSpaceSize * model.numFields * FactorSize)
{
    const int MaxNumFields = 1024;
    dWeights = cuda_utils.malloc<float>(weightsSize);
    dSquaredGradsSum = cuda_utils.malloc<float>(weightsSize);

    dXYTrainInputBuffer = cuda_utils.malloc<int>(maxTrainBatchSize * (numFields + 1));
    dLearnFieldSums = cuda_utils.malloc<float>(MaxNumFields);

    CHECK_ERR(cudaGetLastError());
    CHECK_ERR(cudaDeviceSynchronize());

    cuda_utils.memcpy(dWeights, model.weights.data(), weightsSize);

    cuda_utils.memcpyToSymbol(cL2Reg, l2Reg);
    cuda_utils.memcpyToSymbol(cLearningRate, learningRate);
    cuda_utils.memcpyToSymbol(cNormalizationFactor, model.normalizationFactor);
    cuda_utils.memcpyToSymbol(cRowSize, numFields * FactorSize);
    cuda_utils.memcpyToSymbol2(cScaledNormalizationFactor, model.normalizationFactor * samplingFactor, model.normalizationFactor);

    cuda_utils.memset(dLearnFieldSums, 0, MaxNumFields);
    fillKernel<<<HashSpaceSize, numFields * FactorSize>>>(dSquaredGradsSum.get(), 1.0);
}

void FFMTrainer::learn(HostBuffer<int> const & hXYBatchBuffer, int batchSize)
{
    assert(batchSize <= maxTrainBatchSize);

    cuda_utils.memcpy(dXYTrainInputBuffer, hXYBatchBuffer, batchSize * (numFields + 1));
    const int row_len = numFields + 1;

    for (int i = 0; i < batchSize; ++i) {
        int y = hXYBatchBuffer.get()[i * row_len + numFields];
        const int *xy = dXYTrainInputBuffer.get() + i * row_len;
        ffmInnerSumKernel<<<numFields, numFields>>>(dWeights.get(), xy, dLearnFieldSums.get());
        updateKernel<<<numFields, numFields * 4>>>(dLearnFieldSums.get(), dWeights.get(), dSquaredGradsSum.get(), xy, y, numFields);

        CHECK_ERR(cudaGetLastError());
    }
    CHECK_ERR(cudaDeviceSynchronize());
}

FFMPredictor FFMTrainer::createPredictor() const
{
    return FFMPredictor(&dWeights, maxTrainBatchSize * 5, numFields);
}

template <typename T>
HostBuffer<T> FFMStatic::createHostBuffer(int size)
{
    return cuda_utils.hostMalloc<T>(size);
}

void FFMTrainer::copyWeightsToHost(float *hWeights)
{
    cuda_utils.memcpy(hWeights, dWeights, weightsSize);
}

void FFMTrainer::copyGradsToHost(float *hGrads)
{
    cuda_utils.memcpy(hGrads, dSquaredGradsSum, weightsSize);
}

void FFMStatic::init()
{
    CHECK_ERR(cudaSetDeviceFlags(cudaDeviceMapHost));
    CHECK_ERR(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
    //CHECK_ERR(cudaSetDeviceFlags(cudaDeviceScheduleYield));
    //CHECK_ERR(cudaSetDeviceFlags(cudaDeviceBlockingSync));

#if __CUDA_ARCH__ > 500
    CHECK_ERR(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 0));
    CHECK_ERR(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 0));
#endif // CUDA 5.0
    CHECK_ERR(cudaDeviceSetLimit(cudaLimitStackSize, 0));
    CHECK_ERR(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 0));
    CHECK_ERR(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
}

void FFMStatic::destroy()
{
    CHECK_ERR(cudaDeviceReset());
}

template HostBuffer<int> FFMStatic::createHostBuffer<int>(int);

template HostBuffer<float> FFMStatic::createHostBuffer<float>(int);

