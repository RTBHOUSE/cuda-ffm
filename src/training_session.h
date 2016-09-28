#ifndef CUDA_TRAINING_SESSION_H
#define CUDA_TRAINING_SESSION_H

#include <memory>

#include "ffm_trainer.h"
#include "ffm_predictor.h"
#include "dataset.h"
#include "model.h"
#include "training_options.h"
#include "training_history.h"

// Learns FFM model.
// see README.md
class TrainingSession
{
    Options const options;

    Dataset trainingDataset;
    Dataset testingDataset;

    Model model; // with current weights
    Model bestModel; // with best weights so far
    FFMTrainer trainer;
    FFMPredictor predictor;
    HostBuffer<int> hXYTrainBatchBuffer;
    TrainingHistory trainingHistory;

public:
    TrainingSession(Options const & options);

    void trainModel();
    void exportModel() const;

private:
    void learnOneEpoch(int epoch);
    float calcLogLoss(Dataset & dataset); // calculates log-loss metric for a given dataset
    void printSessionInfo() const;
    int createTrainingBatch();
    void updateTrainingHistory(int epoch, int64_t batchIdx);
    void binarySerializeModel(std::string const & suffix = "") const;
};

#endif // CUDA_TRAINING_SESSION_H
