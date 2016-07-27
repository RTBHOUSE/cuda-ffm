#ifndef CUDA_FFM_TRAINER_H
#define CUDA_FFM_TRAINER_H

#include "model.h"
#include "cuda_buffer.h"

struct FFMTrainerStatic
{
    static void init();
    static void destroy();
};

// Trains FFM model on CUDA using AdaGrad.
struct FFMTrainer
{
    const int numFields;
    const float samplingFactor; // 100 - for 1%  sampling, 10 for 10 % sampling
    const int maxBatchSize;

    // state
    DeviceBuffer<float> dWeights;
    DeviceBuffer<float> dSquaredGradsSum;

    // temporary buffers for learning
    DeviceBuffer<float> dLearnFieldSums;
    DeviceBuffer<int> dXYLearnInputBuffer;

    // temporary buffers for prediction
    DeviceBuffer<int> dXYPredictInputBuffer;
    DeviceBuffer<float> dPredictResultsBuffer;
    DeviceBuffer<float> dPredictFieldSums;

    FFMTrainer(Model const & model, float samplingFactor, int maxBatchSize, float l2Reg, float learningRate);
    ~FFMTrainer();

    template <typename T>
    T * createHostBuffer(int size);

    template <typename T>
    void destroyHostBuffer(T * hBuffer);

    void copyWeightsToHost(float * hWeights);
    void copyGradsToHost(float * hGrads);

    // learn using given batch of samples
    void learn(int const * hXYBatchBuffer, int batchSize);

    // predict given batch of samples, store prediction results inside predictResults
    void predict(int const * hXYBatchBuffer, int batchSize, float * predictResults);

private:
    int weightsSize;
};

#endif // CUDA_FFM_TRAINER_H
