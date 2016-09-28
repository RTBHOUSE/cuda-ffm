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

    Model(int const numFields, std::string const & inputModelFilePath)
            : Model(numFields)
    {
        binaryDeserialize(inputModelFilePath);
    }

    Model(int const numFields, int seed)
            : Model(numFields)
    {
        randomInit(seed);
    }

    Model(int const numFields)
            : numFields(numFields),
              normalizationFactor(1.0 / static_cast<float>(numFields)),
              weights(HashSpaceSize * numFields * FactorSize)
    {
    }

    Model(Model && other)
            : numFields(other.numFields),
              normalizationFactor(other.normalizationFactor),
              weights(std::move(other.weights))
    {
    }

    void randomInit(int seed);

    void exportModel(std::string const & modelFilePath) const;
    void importModel(std::string const & modelFilePath);

    void binarySerialize(std::string const & modelFilePath) const;
    void binaryDeserialize(std::string const & modelFilePath);
};

#endif // CUDA_MODEL_H
