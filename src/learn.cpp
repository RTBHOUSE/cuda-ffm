#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "ffm_trainer.h"
#include "dataset.h"
#include "log_loss_calculator.h"
#include "model.h"
#include "learn_options.h"
#include "timer.h"

// calculates log-loss metric for a dataset

float calcLogLoss(FFMTrainer & fm, Model const & model, Dataset & dataset, Options const & options)
{
    LogLossCalculator logLossCalculator(options.samplingFactor);
    dataset.rewind();

    int * hXYBatchBuffer = fm.createHostBuffer<int>(options.maxBatchSize * (model.numFields + 1));
    float * hPredictResultsBuffer = fm.createHostBuffer<float>(options.maxBatchSize);

    while (dataset.hasNext()) {
        int sampleInBatchIdx = 0;
        for (; sampleInBatchIdx < options.maxBatchSize && dataset.hasNext(); ++sampleInBatchIdx) {
            Dataset::Sample const & xyBuffer = dataset.next();
            std::copy(xyBuffer.begin(), xyBuffer.end(), hXYBatchBuffer + sampleInBatchIdx * (model.numFields + 1));
        }
        fm.predict(hXYBatchBuffer, sampleInBatchIdx, hPredictResultsBuffer);

        for (int resultIdx = 0; resultIdx < sampleInBatchIdx; ++resultIdx) {
            const int y = hXYBatchBuffer[resultIdx * (model.numFields + 1) + model.numFields];
            const float t = hPredictResultsBuffer[resultIdx];
            logLossCalculator.update(t, y);
        }
    }

    fm.destroyHostBuffer(hXYBatchBuffer);
    fm.destroyHostBuffer(hPredictResultsBuffer);
    return logLossCalculator.get();
}

// Learns FFM model.
// see README.md
void learn(Model & model, Dataset & trainingDataset, Dataset & testingDataset, const Options & options)
{
    printf("Number of samples in training dataset: %ld\n", trainingDataset.numSamples);
    printf("Number of samples in testing dataset : %ld\n", testingDataset.numSamples);
    printf("Number of fields: %d, hash space size: %d\n", model.numFields, HashSpaceSize);
    printf("L2 regularization: %f, learning rate: %f\n", options.l2Reg, options.learningRate);
    printf("Sampling factor: %f, seed: %d\n", options.samplingFactor, options.seed);
    printf("Max number of epochs: %d\n\n", options.maxNumEpochs);

    FFMTrainer fm(model, options.samplingFactor, options.maxBatchSize, options.l2Reg, options.learningRate);

    float logLoss = calcLogLoss(fm, model, testingDataset, options);
    printf("Initial testing dataset log-loss: %f\n", logLoss);

    Timer timer;
    printf("\n%5s %14s %16s\n", "epoch", "epoch_duration", "testing_log_loss");

    int * hXYBatchBuffer = fm.createHostBuffer<int>(options.maxBatchSize * (model.numFields + 1));

    for (int epoch = 1; epoch <= options.maxNumEpochs; ++epoch) {
        timer.start();

        while (trainingDataset.hasNext()) {
            int sampleInBatchIdx = 0;
            for (; sampleInBatchIdx < options.maxBatchSize && trainingDataset.hasNext(); ++sampleInBatchIdx) {
                Dataset::Sample const & xyBuffer = trainingDataset.next();
                std::copy(xyBuffer.begin(), xyBuffer.end(), hXYBatchBuffer + sampleInBatchIdx * (model.numFields + 1));
            }
            fm.learn(hXYBatchBuffer, sampleInBatchIdx);
        }

        trainingDataset.rewind();
        float logLoss = calcLogLoss(fm, model, testingDataset, options);
        float duration = timer.get();
        printf("%5d %14.3f %16.7f\n", epoch, duration, logLoss);
    }

    fm.copyWeightsToHost(model.weights.data());
    fm.destroyHostBuffer(hXYBatchBuffer);
}

int main(int const argc, const char ** argv)
{
    Options options = parseOptions(argc, argv);
    FFMTrainerStatic::init();

    Dataset trainingDataset(options.trainingDatasetPath);
    Dataset testingDataset(options.testingDatasetPath);

    std::cout << "Initializing model" << std::endl;

    Model model(trainingDataset.numFields);
    if (!options.inputModelFilePath.isEmpty()) {
        model.deserialize(options.inputModelFilePath.get());
    } else {
        model.randomInit(options.seed);
    }

    std::cout << "Learning model" << std::endl;

    learn(model, trainingDataset, testingDataset, options);

    std::cout << "Saving model to file: " << options.outputModelFilePath << std::endl;
    model.serialize(options.outputModelFilePath);

    FFMTrainerStatic::destroy();
    return EXIT_SUCCESS;
}
