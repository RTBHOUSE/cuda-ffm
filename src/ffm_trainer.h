#ifndef CUDA_FFM_TRAINER_H
#define CUDA_FFM_TRAINER_H

#include "model.h"
#include "cuda_buffer.h"

struct FFMStatic
{
    static void init();
    static void destroy();

    template <typename T>
    static HostBuffer<T> createHostBuffer(int size);
};

struct FFMPredictor;

// Trains FFM model on CUDA using AdaGrad.
struct FFMTrainer
{
    const int numFields;
    const float samplingFactor; // 100 - for 1%  sampling, 10 for 10 % sampling
    const int maxTrainBatchSize;

    // state
    DeviceBuffer<float> dWeights;
    DeviceBuffer<float> dSquaredGradsSum;

    // temporary buffers for training
    DeviceBuffer<float> dLearnFieldSums;
    DeviceBuffer<int> dXYTrainInputBuffer;

    FFMTrainer(Model const & model, float samplingFactor, int maxBatchSize, float l2Reg, float learningRate);

    void copyWeightsToHost(float * hWeights);
    void copyGradsToHost(float * hGrads);
    FFMPredictor createPredictor() const;

    // learn using given batch of samples
    void learn(HostBuffer<int> const & hXYBatchBuffer, int batchSize);

private:
    int weightsSize;
};


#endif // CUDA_FFM_TRAINER_H
