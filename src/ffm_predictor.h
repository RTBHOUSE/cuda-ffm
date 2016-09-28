#ifndef CUDA_FFM_PREDICTOR_H
#define CUDA_FFM_PREDICTOR_H

#include "cuda_buffer.h"

struct Model;

struct FFMPredictor
{
    const int numFields;
    const int maxPredictBatchSize;

    DeviceBuffer<float> const * dWeights; // owned by FFMTrainer

    // temporary buffers for prediction
    DeviceBuffer<int> dXYPredictInputBuffer;
    DeviceBuffer<float> dPredictResultsBuffer;
    DeviceBuffer<float> dPredictFieldSums;

    FFMPredictor(DeviceBuffer<float> const * dWeights, int maxBatchSize, Model const & model);

    // predict given batch of samples, store prediction results inside predictResults
    void predict(HostBuffer<int> const & hXYBatchBuffer, int batchSize, HostBuffer<float> & hPredictResults);
};

#endif // CUDA_FFM_PREDICTOR_H
