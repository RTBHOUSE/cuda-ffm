#include "dataset.h"
#include "dataset_writer.h"
#include "cli.h"

// Splits dataset into two.
//
//   Reads samples from file pointed by argv[1] one by one:
//     - With probability argv[2] writes sample to file pointed by argv[3]
//     - otherwise writes sample to file pointed by argv[4]
int main(int argc, const char ** argv)
{
    ::srand48(::time(NULL));

    assert(argc == 5);
    auto const & args = argvToArgs(argc, argv);

    Dataset inputDataset(args[0]);
    double dataset1Prob = std::stod(args[1]);
    DatasetWriter outputWriter1(args[2]);
    DatasetWriter outputWriter2(args[3]);

    while (inputDataset.hasNext()) {
        auto const & sample = inputDataset.next();
        if (::drand48() <= dataset1Prob) {
            outputWriter1.writeSample(sample);
        } else {
            outputWriter2.writeSample(sample);
        }
    }

    outputWriter1.writeFooter();
    outputWriter2.writeFooter();

    return 0;
}
