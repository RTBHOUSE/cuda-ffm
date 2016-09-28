#include "training_options.h"
#include "cli.h"

#include <vector>
#include <stdexcept>

std::string train_help()
{
    return std::string("Usage: learn [<options>]\n"
                       "Options are:\n"
                       "\n"
                       "   --l2reg <float>                : L2-regularization penalty\n"
                       "   --maxNumEpochs <int>           : number of epochs\n"
                       "   --numStepsPerEpoch <int>       : number of steps per epoch\n"
                       "   --learningRate <float>         : training rate\n"
                       "   --seed <int>                   : random number generator seed\n"
                       "   --maxBatchSize <int>           : performance related, use 200\n"
                       "   --samplingFactor <float>       : how many negative data were subsampled\n"
                       "   --inputModelFilePath <string>  : path to the input model\n"
                       "   --outputModelFilePath <string> : path to the output model\n"
                       "   --trainingDatasetPath <string> : path to the training dataset\n"
                       "   --testingDatasetPath <string>  : path to the testing dataset\n");
}

Options parseOptions(std::vector<std::string> const & args)
{
    uint32_t const argc = static_cast<uint32_t>(args.size());

    if (argc == 0) {
        throw std::invalid_argument(train_help());
    }

    Options opt;

    uint32_t i = 0;
    for (; i < argc; ++i) {
        if (i >= argc - 1) throw std::invalid_argument("Invalid number of command line arguments");

        if (args[i] == "--maxNumEpochs") {
            opt.maxNumEpochs = std::stoi(args[++i]);
            assert(opt.maxNumEpochs > 0);
        } else if (args[i] == "--numStepsPerEpoch") {
            opt.numStepsPerEpoch = std::stoi(args[++i]);
            assert(opt.numStepsPerEpoch > 1);
        } else if (args[i] == "--learningRate") {
            opt.learningRate = std::stof(args[++i]);
            assert(opt.learningRate > 0.0);
        } else if (args[i] == "--l2reg") {
            opt.l2Reg = std::stof(args[++i]);
            assert(opt.l2Reg >= 0.0);
        } else if (args[i] == "--samplingFactor") {
            opt.samplingFactor = std::stof(args[++i]);
        } else if (args[i] == "--maxBatchSize") {
            opt.maxBatchSize = std::stoi(args[++i]);
        } else if (args[i] == "--seed") {
            opt.seed = std::stoi(args[++i]);
        } else if (args[i] == "--inputModelFilePath") {
            opt.inputModelFilePath = args[++i];
        } else if (args[i] == "--outputModelFilePath") {
            opt.outputModelFilePath = args[++i];
        } else if (args[i] == "--trainingDatasetPath") {
            opt.trainingDatasetPath = args[++i];
        } else if (args[i] == "--testingDatasetPath") {
            opt.testingDatasetPath = args[++i];
        } else {
            throw std::invalid_argument("Invalid command line parameter: " + args[i]);
        }
    }

    return opt;
}

Options parseOptions(int argc, const char ** argv)
{
    return parseOptions(argvToArgs(argc, argv));
}
