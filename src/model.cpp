#include "model.h"

#include <stdexcept>
#include <cstring>
#include <cstdio>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>
#include <cassert>

void Model::randomInit(int seed)
{
    LOG("Initializing model with random weights");

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

void Model::exportModel(std::string const & modelFilePath) const
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

void Model::importModel(std::string const & modelFilePath)
{
    LOG("Initializing model from %s", modelFilePath.c_str());

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

void Model::binarySerialize(std::string const & modelFilePath) const
{
    const int EndiannessCheck = 123456;

    FILE * file = ::fopen(modelFilePath.c_str(), "wb");

    ::fprintf(file, "ffm_model_v1 %d %d %d\n", FactorSize, numFields, HashSpaceSize);
    ::fwrite(&EndiannessCheck, sizeof(int), 1, file);

    for (int featureIdx = 0; featureIdx < HashSpaceSize; ++featureIdx) {
        size_t const rowSize = FactorSize * numFields;
        int64_t const offset = featureIdx * rowSize;
        size_t numWritten = ::fwrite(weights.data() + offset, sizeof(float), rowSize, file);
        assert(numWritten == rowSize);
    }

    ::fclose(file);
}

void Model::binaryDeserialize(std::string const & modelFilePath)
{
    LOG("Initializing model from %s (binary mode)", modelFilePath.c_str());

    FILE * file = ::fopen(modelFilePath.c_str(), "rb");

    int rFactorSize, rNumFields, rHashSpaceSize;
    int read = fscanf(file, "ffm_model_v1 %d %d %d\n", &rFactorSize, &rNumFields, &rHashSpaceSize);
    assert(read == 3);
    assert(rFactorSize == FactorSize);
    assert(rHashSpaceSize == HashSpaceSize);
    assert(rNumFields == numFields);

    int endiannessCheck;
    read = ::fread(&endiannessCheck, sizeof(int), 1, file);
    assert(read == 1);
    assert(endiannessCheck == 123456);

    int64_t const rowSize = FactorSize * numFields;
    int64_t const numWeights = rowSize * HashSpaceSize;
    for (int64_t weightIdx = 0; weightIdx < numWeights; weightIdx += rowSize) {
        int read = ::fread(weights.data() + weightIdx, sizeof(float), rowSize, file);
        assert(read == numFields + 1);
    }

    ::fclose(file);
}
