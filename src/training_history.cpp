#include "training_history.h"
#include "utils.h"

#include <cstdio>
#include <numeric>
#include <limits>

TrainingHistory::TrainingHistory(int64_t numSamplesTotal, Options const & options)
        : numSamplesTotal(numSamplesTotal),
          options(options),
          trainingHistoryFile(::fopen((options.outputModelFilePath + ".csv").c_str(), "w"))
{
    assert(numSamplesTotal > options.numStepsPerEpoch);
    assert(numSamplesTotal / options.maxBatchSize > options.numStepsPerEpoch);

    numSamplesPerStep = numSamplesTotal / options.numStepsPerEpoch;
    ::fprintf(trainingHistoryFile, "epoch,log_loss,diff\n");
}

TrainingHistory::~TrainingHistory()
{
    ::fclose(trainingHistoryFile);
}

bool TrainingHistory::isLastBatchInStep(int64_t batchIdx) const
{
    auto curStep = batchIdx * options.maxBatchSize / numSamplesPerStep;
    auto nextStep = (batchIdx + 1) * options.maxBatchSize / numSamplesPerStep;
    return curStep != nextStep;
}

void TrainingHistory::updateHistory(int epoch, int64_t batchIdx, double logLoss)
{
    double diff = logLossValues.empty() ? 0 : logLoss - logLossValues.back();
    logLossValues.push_back(logLoss);
    logLossDiffs.push_back(diff);

    float step = getStepIdx(epoch, batchIdx);
    ::fprintf(trainingHistoryFile, "%5lf,%16.7lf,%16.7lf\n", step, logLoss, diff);
    ::fflush(trainingHistoryFile);
}

float TrainingHistory::getStepIdx(int epoch, int64_t batchIdx) const
{
    return epoch - 1 + batchIdx * options.maxBatchSize / static_cast<double>(numSamplesTotal);
}

bool TrainingHistory::stopCondition() const
{
    const size_t windowSize = options.numStepsPerEpoch;

    if (logLossDiffs.size() < windowSize) {
        return false;
    }

    double rolling_mean = std::accumulate(logLossDiffs.end() - windowSize, logLossDiffs.end(), 0.0) / windowSize;
    return rolling_mean > -1e-7;
}

Option<double> TrainingHistory::bestLogLoss() const
{
    if (logLossValues.empty()) {
        return Option<double>();
    }

    return Option<double>(
            std::accumulate(logLossValues.begin(), logLossValues.end(), std::numeric_limits<double>::max(),
                            [](double a, double b) {return std::min(a, b);}));
}
