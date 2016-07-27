#ifndef CUDA_CONSTANTS_H
#define CUDA_CONSTANTS_H

static const int FactorSize = 4;
static const int MaxPredictBlockSize = 74; // maxNumFields
static const int MaxUpdateBlockSize = MaxPredictBlockSize * FactorSize;
static const int HashSpaceSize = 1000000;

#endif // CUDA_CONSTANTS_H
