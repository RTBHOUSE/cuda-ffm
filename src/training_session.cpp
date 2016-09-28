#include "training_session.h"
#include "log_loss_calculator.h"
#include "timer.h"
#include "ffm_predictor.h"

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

TrainingSession::TrainingSession(Options const & options)
        : options(options),
          trainingDataset(options.trainingDatasetPath),
          testingDataset(options.testingDatasetPath),
          model(std::move(
                  options.inputModelFilePath.isEmpty() ?
                          Model(trainingDataset.numFields, options.seed) :
                          Model(trainingDataset.numFields, options.inputModelFilePath.cget()))),
          bestModel(trainingDataset.numFields),
          trainer(model, options.samplingFactor, options.maxBatchSize, options.l2Reg, options.learningRate),
          predictor(trainer.createPredictor()),
          hXYTrainBatchBuffer(FFMStatic::createHostBuffer<int>(trainer.maxTrainBatchSize * (model.numFields + 1))),
          trainingHistory(trainingDataset.numSamples, options)
{
}

float TrainingSession::calcLogLoss(Dataset & dataset)
{
    LogLossCalculator logLossCalculator(options.samplingFactor);

    HostBuffer<int> hXYPredictBatchBuffer = FFMStatic::createHostBuffer<int>(predictor.maxPredictBatchSize * (model.numFields + 1));
    HostBuffer<float> hPredictResultsBuffer = FFMStatic::createHostBuffer<float>(predictor.maxPredictBatchSize);

    dataset.rewind();
    while (dataset.hasNext()) {
        int sampleInBatchIdx = 0;
        for (; sampleInBatchIdx < predictor.maxPredictBatchSize && dataset.hasNext(); ++sampleInBatchIdx) {
            Dataset::Sample const & xyBuffer = dataset.next();
            std::copy(xyBuffer.begin(), xyBuffer.end(), hXYPredictBatchBuffer.get() + sampleInBatchIdx * (model.numFields + 1));
        }
        predictor.predict(hXYPredictBatchBuffer, sampleInBatchIdx, hPredictResultsBuffer);

        for (int resultIdx = 0; resultIdx < sampleInBatchIdx; ++resultIdx) {
            const int y = hXYPredictBatchBuffer.get()[resultIdx * (model.numFields + 1) + model.numFields];
            const float t = hPredictResultsBuffer.get()[resultIdx];
            logLossCalculator.update(t, y);
        }
    }

    return logLossCalculator.get();
}

void TrainingSession::printSessionInfo() const
{
    LOG("CUDA FFM version: %s", CUDA_FFM_VERSION);
    LOG("Number of samples in training dataset: %ld", trainingDataset.numSamples);
    LOG("Number of samples in testing dataset : %ld", testingDataset.numSamples);
    LOG("Number of fields: %d, hash space size: %d", model.numFields, HashSpaceSize);
    LOG("L2 regularization: %.8f, learning rate: %.8f", options.l2Reg, options.learningRate);
    LOG("Sampling factor: %f, seed: %d", options.samplingFactor, options.seed);
    LOG("Max number of epochs: %d, number of steps per epoch: %d", options.maxNumEpochs, options.numStepsPerEpoch);
    LOG("");
}

void TrainingSession::trainModel()
{
    LOG("Training model");
    printSessionInfo();

    const float logLoss = calcLogLoss(testingDataset);
    LOG("Initial testing dataset log-loss: %f", logLoss);

    for (int epoch = 1; epoch <= options.maxNumEpochs; ++epoch) {
        const float duration = Timer::getDuration([&]() {
            learnOneEpoch(epoch);
        });

        LOG("> Epoch finished: epoch: %d; duration: %.3f s", epoch, duration);

        if (trainingHistory.stopCondition()) {
            break;
        }
    }

    LOG("Best model log-loss: %16.7lf", trainingHistory.bestLogLoss().get());
    LOG("Last model log-loss: %16.7lf", calcLogLoss(testingDataset));

    trainer.copyWeightsToHost(model.weights.data());
}

int TrainingSession::createTrainingBatch()
{
    int sampleInBatchIdx = 0;

    for (; sampleInBatchIdx < trainer.maxTrainBatchSize && trainingDataset.hasNext(); ++sampleInBatchIdx) {
        const Dataset::Sample& xyBuffer = trainingDataset.next();
        int offset = sampleInBatchIdx * (model.numFields + 1);
        std::copy(xyBuffer.begin(), xyBuffer.end(), hXYTrainBatchBuffer.get() + offset);
    }

    return sampleInBatchIdx;
}

void TrainingSession::updateTrainingHistory(int epoch, int64_t batchIdx)
{
    if (trainingHistory.isLastBatchInStep(batchIdx)) {
        const float logLoss = calcLogLoss(testingDataset);
        const Option<double> prevBestLogLoss = trainingHistory.bestLogLoss();
        trainingHistory.updateHistory(epoch, batchIdx, logLoss);

        if (prevBestLogLoss.isEmpty() || prevBestLogLoss.cget() > logLoss) {
            LOG("Better model found. Step: %f, logLoss: %f", trainingHistory.getStepIdx(epoch, batchIdx), logLoss);
            trainer.copyWeightsToHost(bestModel.weights.data());
        }
    }
}

void TrainingSession::learnOneEpoch(int epoch)
{
    for (int64_t batchIdx = 0; trainingDataset.hasNext(); ++batchIdx) {
        const int batchSize = createTrainingBatch();
        trainer.learn(hXYTrainBatchBuffer, batchSize);
        updateTrainingHistory(epoch, batchIdx);

        if (trainingHistory.stopCondition()) {
            LOG("Early stopping. Step: %5.3f", trainingHistory.getStepIdx(epoch, batchIdx));
            break;
        }
    }

    trainingDataset.rewind();
}

void TrainingSession::exportModel() const
{
    LOG("Exporting model to file: %s", options.outputModelFilePath.c_str());

    std::string const tmpPath = options.outputModelFilePath + ".tmp";
    bestModel.exportModel(tmpPath);
    model.exportModel(options.outputModelFilePath + ".last");
    ::rename(tmpPath.c_str(), options.outputModelFilePath.c_str());
}

void TrainingSession::binarySerializeModel(std::string const & suffix) const
{
    const std::string filePath = options.outputModelFilePath + suffix;
    LOG("Saving model to file: %s (binary mode)", filePath.c_str());
    model.binarySerialize(filePath);
}
