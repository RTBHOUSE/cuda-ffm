#ifndef CUDA_DATASET_H
#define CUDA_DATASET_H

#include "utils.h"
#include "constants.h"

#include <vector>
#include <cstdio>

namespace dataset {
static const char * CANARY = "\ncanary\n";
const int EndiannessCheck = 123456;
}

// Dataset instance points to a file with binary samples and provides iterator interface to it
struct Dataset
{
    // f1:x1, f2:x2, ..., fn:xn, y
    // where y is -1 or 1
    typedef std::vector<int> Sample;

    int hashSpaceSize;
    int numFields;
    int64_t numSamples;

    Dataset(std::string const & filePath)
            : filePath(filePath),
              file(NULL)
    {
        rewind();
    }

    void rewind()
    {
        if (file) {
            ::fclose(file);
        }

        file = ::fopen(filePath.c_str(), "rb");
        assert(file);

        readHeader();
    }

    inline bool hasNext() const
    {
        return currentSampleIdx < numSamples;
    }

    inline int64_t numSamplesLeft() const
    {
        return numSamples - currentSampleIdx;
    }

    Sample const & next()
    {
        int read = ::fread(xyBuffer.data(), sizeof(int), numFields + 1, file);
        assert(read == numFields + 1);

        if (currentSampleIdx == numSamples - 1) {
            closeFile();
        }

        ++currentSampleIdx;
        return xyBuffer;
    }

private:

    void readHeader()
    {
        ::fseek(file, 0, SEEK_END);
        int64_t fileLen = ::ftell(file);
        ::rewind(file);

        int rFactorSize;
        int read = fscanf(file, "ffm_features_v1 %d %d %d\n", &rFactorSize, &numFields, &hashSpaceSize);
        assert(read == 3);
        assert(rFactorSize == FactorSize);
        assert(hashSpaceSize == HashSpaceSize);

        int endiannessCheck;
        read = ::fread(&endiannessCheck, sizeof(int), 1, file);
        assert(read == 1);
        assert(endiannessCheck == dataset::EndiannessCheck);

        int64_t currentPos = ::ftell(file);

        numSamples = fileLen - currentPos - sizeof(dataset::CANARY);
        const int rowSize = sizeof(int) * (numFields + 1);
        ASSERT(numSamples % rowSize == 0, "numSamples: %ld rowSize: %d", numSamples, rowSize);
        numSamples /= rowSize;

        currentSampleIdx = 0;
        xyBuffer.resize(numFields + 1);
    }

    void closeFile()
    {
        char buf[sizeof(dataset::CANARY) + 1];
        size_t numRead = ::fread(buf, 1, sizeof(dataset::CANARY), file);
        buf[sizeof(buf) - 1] = '\0';

        assert(numRead == sizeof(dataset::CANARY));
        assert(std::string(buf) == dataset::CANARY);

        ::fclose(file);
        file = NULL;
    }

    const std::string filePath;
    int currentSampleIdx;
    Sample xyBuffer; // x1, ..., xn, y ; y is -1 or 1
    FILE * file;
};

#endif //CUDA_DATASET_H
