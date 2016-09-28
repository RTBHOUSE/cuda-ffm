#ifndef CUDA_DATASET_WRITER_H
#define CUDA_DATASET_WRITER_H

#include "utils.h"
#include "dataset.h"

#include <string>
#include <cassert>

// DatasetWriter can write a dataset to file using binary format
struct DatasetWriter
{
    typedef Dataset::Sample Sample;

    DatasetWriter(FILE * file)
            : file(file),
              headerWritten(false)
    {
    }

    DatasetWriter(std::string const & filePath)
            : file(::fopen(filePath.c_str(), "wb")),
              headerWritten(false)
    {
    }

    void writeSample(Sample const & sample)
    {
        if (!headerWritten) {
            writeHeader(sample.size() - 1);
        }
        size_t numWritten = ::fwrite(sample.data(), sizeof(int), sample.size(), file);
        assert(numWritten == sample.size());
    }

    void writeFooter()
    {
        assert(headerWritten);

        ::fputs(dataset::CANARY, file);
        ::fclose(file);
    }

private:

    void writeHeader(int numFields)
    {
        assert(!headerWritten);

        ::fprintf(file, "ffm_features_v1 %d %d %d\n", FactorSize, numFields, HashSpaceSize);
        ::fwrite(&dataset::EndiannessCheck, sizeof(int), 1, file);
        headerWritten = true;
    }

    FILE * file;
    bool headerWritten;
};

#endif //CUDA_DATASET_WRITER_H
