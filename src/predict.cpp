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

// Calculates log-loss of a dataset using ffm-native-ops C++ library (without GPU).
//
// Arguments:
//   - argv[1] - model file path
//   - argv[2] - dataset file path
//   - argv[3] - sampling factor
int main(int argc, const char ** argv)
{
    auto args = argvToArgs(argc, argv);

    Dataset dataset(args[1]);
    Model model(dataset.numFields);
    model.deserialize(args[0]);

    double samplingFactor(std::stod(args[2]));

    LogLossCalculator logLossCalc(samplingFactor);

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
