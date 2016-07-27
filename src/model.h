#ifndef CUDA_MODEL_H
#define CUDA_MODEL_H

#include <string>
#include <fstream>
#include <vector>

#include "utils.h"
#include "constants.h"

// Contains FFM data (weights, params).
// Can serialize and deserialize itself to/from a file.
struct Model
{
    const int numFields;
    const float normalizationFactor;
    std::vector<float> weights;

    Model(int const numFields)
            : numFields(numFields),
              normalizationFactor(1.0 / static_cast<float>(numFields)),
              weights(HashSpaceSize * numFields * FactorSize)
    {
    }

    void randomInit(int seed);
    void deserialize(std::string const & modelFilePath);
    void serialize(std::string const & modelFilePath);
};

#endif // CUDA_MODEL_H
