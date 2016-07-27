#include "constants.h"
#include "cli.h"
#include "dataset.h"
#include "dataset_writer.h"

#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>

#define MAX_MISS_COUNT 5 // change it to 2 << 30 to disable approximate shuffling

struct RandomNumberGenerator
{
    std::vector<double> randomDoubles;
    std::vector<int> randomInts;

    std::vector<double>::const_iterator doubleIt;
    std::vector<int>::const_iterator intIt;

    RandomNumberGenerator(int maxInt)
            : randomDoubles(10000),
              randomInts(8000),
              doubleIt(randomDoubles.begin()),
              intIt(randomInts.begin())
    {
        ::srand(::time(NULL));
        std::generate(randomDoubles.begin(), randomDoubles.end(), [] {return static_cast<double>(::rand()) / RAND_MAX;});
        std::generate(randomInts.begin(), randomInts.end(), [maxInt] {return ::rand() % maxInt;});
    }

    double nextDouble()
    {
        if (doubleIt == randomDoubles.end()) {
            doubleIt = randomDoubles.begin();
        }
        return *doubleIt++;
    }

    int nextInt()
    {
        if (intIt == randomInts.end()) {
            intIt = randomInts.begin();
        }
        return *intIt++;
    }
};

// Reads data from multiple datasets, and outputs single shuffled dataset to stdout.
//
// This is very naive implementation with two dirty hacks to fully utilize SSD/CPU.
//
// Assumptions:
//   - input files are shuffled
//   - input files have similar number of samples
//      - otherwise change MAX_MISS_COUNT
//
int main(int argc, const char ** argv)
{
    auto paths = argvToArgs(argc, argv);
    std::vector<Dataset> datasets;

    int64_t numSamplesLeftTotal = 0;
    int numFiles = paths.size();

    for (auto const & path : paths) {
        auto dataset = Dataset(path);
        datasets.push_back(dataset);
        numSamplesLeftTotal += dataset.numSamplesLeft();
    }

    DatasetWriter datasetWriter(stdout);
    RandomNumberGenerator rng(numFiles);
    int missCount = 0;

    while (numSamplesLeftTotal > 0) {
        int datasetIdx = rng.nextInt();
        auto & dataset = datasets.at(datasetIdx);

        if (!dataset.hasNext()) {
            continue;
        }

        double datasetProb = static_cast<double>(dataset.numSamplesLeft()) / numSamplesLeftTotal;

        if (rng.nextDouble() <= datasetProb || missCount >= MAX_MISS_COUNT) {
            auto const & sample = dataset.next();
            datasetWriter.writeSample(sample);
            numSamplesLeftTotal--;
            missCount = 0;
        } else {
            missCount++;
        }
    }

    datasetWriter.writeFooter();

    return 0;
}
