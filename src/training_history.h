#ifndef CUDA_TRAINING_HISTORY_H
#define CUDA_TRAINING_HISTORY_H

#include "training_options.h"
#include "utils.h"

#include <vector>

// TrainingHistory:
//  - tracks logLoss achieved every training step.
//  - computes early-stopping condition
//  - writes history to CSV file
struct TrainingHistory
{
    TrainingHistory(int64_t numSamplesTotal, Options const & options);
    ~TrainingHistory();

    bool isLastBatchInStep(int64_t batchIdx) const;
    void updateHistory(int epoch, int64_t batchIdx, double logLoss);
    float getStepIdx(int epoch, int64_t batchIdx) const;
    bool stopCondition() const;
    Option<double> bestLogLoss() const;

private:
    std::vector<double> logLossValues;
    std::vector<double> logLossDiffs;

    int64_t numSamplesPerStep;
    int64_t numSamplesTotal;
    Options const & options;
    FILE * trainingHistoryFile;
};

#endif // CUDA_TRAINING_HISTORY_H
