#ifndef CUDA_LEARN_OPTIONS_H
#define CUDA_LEARN_OPTIONS_H

#include <string>
#include "utils.h"

// Learning options
struct Options
{
    Options()
            : learningRate(0.1),
              l2Reg(0.00002),
              samplingFactor(1.0),
              maxNumEpochs(15),
              maxBatchSize(200),
              seed(123)
    {
    }

    float learningRate;
    float l2Reg;

    float samplingFactor; // 100 for 1% sampling
    int maxNumEpochs;
    int maxBatchSize;
    int seed; // for random number generator, used during model initialization

    Option<std::string> inputModelFilePath;
    std::string outputModelFilePath;
    std::string trainingDatasetPath;
    std::string testingDatasetPath;
};

Options parseOptions(int argc, const char ** argv);

#endif // CUDA_LEARN_OPTIONS_H
