#include "constants.h"
#include "cli.h"
#include "model.h"
#include "dataset.h"
#include "log_loss_calculator.h"
#include "FactorizationMachineNativeOps.h"

#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <string>

struct Options {
    Options() : samplingFactor(1.0) {}

    std::string datasetPath;
    std::string textModelPath;
    std::string binModelPath;
    float samplingFactor;
};

Options parseOptions(std::vector<std::string> const & args)
{
    Options options;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--samplingFactor") {
            options.samplingFactor = std::stof(args[++i]);
        } else if (args[i] == "--datasetPath") {
            options.datasetPath = args[++i];
        } else if (args[i] == "--binModelPath") {
            options.binModelPath = args[++i];
        } else if (args[i] == "--textModelPath") {
            options.textModelPath = args[++i];
        } else {
            throw std::invalid_argument("Invalid command line parameter: " + args[i]);
        }
    }

    return options;
}

// Calculates log-loss of a dataset using ffm-native-ops C++ library (without GPU).
//
// Arguments:
//  --datasetPath <path> (--binModelPath | --textModelPath) <path> [--samplingFactor <float>] 
int main(int argc, const char ** argv)
{
    Options const & options = parseOptions(argvToArgs(argc, argv));

    Dataset dataset(options.datasetPath);
    Model model(dataset.numFields);
    if (!options.binModelPath.empty()) {
        model.binaryDeserialize(options.binModelPath);
    } else {
        model.importModel(options.textModelPath);
    }

    LogLossCalculator logLossCalc(options.samplingFactor);

    for (int64_t sampleIdx = 0; dataset.hasNext(); ++sampleIdx) {
        Dataset::Sample const & sample = dataset.next();
        float t = ffmPredict(model.weights.data(), dataset.numFields, sample.data());
        int y = sample.back() > 0 ? 1 : -1;
        logLossCalc.update(t, y);
        std::cout << sampleIdx << " " << y << " " << t << std::endl;
    }

    std::cout << "Log-loss: " << logLossCalc.get() << std::endl;

    return 0;
}
