#include "model.h"

#include <stdexcept>
#include <cstring>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>
#include <cassert>

void Model::randomInit(int seed)
{
    double const coef = static_cast<double>(1.0 / sqrt(static_cast<float>(FactorSize)));

    ::srand48(seed);

    float *w = weights.data();
    for (int j = 0; j < HashSpaceSize; ++j) {
        for (int f = 0; f < numFields; ++f) {
            for (int d = 0; d < FactorSize; ++d, ++w) {
                *w = coef * static_cast<double>(::drand48());
            }
        }
    }
}

void Model::deserialize(std::string const & modelFilePath)
{
    std::ifstream file;
    file.open(modelFilePath);

    int tmpFactorSize;
    int tmpNumFields;
    int tmpHashSpaceSize;
    int64_t numWeights;

    file >> tmpHashSpaceSize >> tmpFactorSize >> tmpNumFields >> numWeights;
    assert(tmpFactorSize == FactorSize);
    assert(tmpNumFields == numFields);
    assert(tmpHashSpaceSize == HashSpaceSize);

    for (int64_t weightIdx = 0; weightIdx < numWeights; ++weightIdx) {
        file >> weights[weightIdx];
    }

    file.close();
}

void Model::serialize(std::string const & modelFilePath)
{
    std::ofstream file;
    file.open(modelFilePath);

    file << HashSpaceSize << " " << FactorSize << " " << numFields << " " << weights.size() << "\n";
    file << std::setprecision(9);

    for (auto f : weights) {
        ASSERT_FIN(f);
        file << f << "\n";
    }

    file.close();
}
