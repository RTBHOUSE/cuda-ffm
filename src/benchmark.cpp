#include "constants.h"
#include "timer.h"
#include "ffm_trainer.h"
#include "ffm_predictor.h"
#include "cuda_buffer.h"
#include "log_loss_calculator.h"
#include "FactorizationMachineNativeOps.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <ctime>

// simple FFM trainer and predictor benchmark

static const int NumRuns = 10;
static const int NumFields = 74;

struct TrainBenchmark
{
    static const int NumIters = 1000 * 100;
    static const int BatchSize = 200;
    static const int XYsize = (NumFields + 1) * sizeof(int) * BatchSize;

    void gpuLearnBenchmark()
    {
        LOG("Starting training benchmark");
        Model model(NumFields, 123);
        FFMTrainer fm(model, 10, BatchSize, 0.0001, 0.2);

        HostBuffer<int> h_xy = FFMStatic::createHostBuffer<int>(BatchSize * (NumFields + 1));

        Timer timer;
        timer.start();

        for (int iter = 0; iter < NumIters; iter += BatchSize) {
            createRandomBatch(iter, h_xy.get());
            fm.learn(h_xy, BatchSize);
        }

        float duration = timer.get();
        LOG("duration: %f s; ops/s: %f", duration, NumIters / duration);
    }

    void createRandomBatch(int iter, int* h_xy)
    {
        for (int batchIdx = 0; batchIdx < BatchSize; ++batchIdx) {
            for (int i = 0; i < NumFields; i++) {
                h_xy[batchIdx * (NumFields + 1) + i] = abs((iter + batchIdx + i) * 1582177L + 1923979) % HashSpaceSize;
            }
            h_xy[(NumFields + 1) * batchIdx + NumFields] = batchIdx % 2 - 1;
        }
    }
};

struct PredictBenchmark
{
    static const int NumIters = 1000 * 1000;
    static const int BatchSize = 200;

    void createSample(int * sampleBuffer, int sampleIdx)
    {
        for (int fieldIdx = 0; fieldIdx < NumFields; fieldIdx++) {
            sampleBuffer[fieldIdx] = abs(sampleIdx * 1582177L + 1923979) % HashSpaceSize;
        }
    }

    void cpuPredictBenchmark()
    {
        LOG("Starting training benchmark (CPU)");

        Model model(NumFields, 123);
        LOG("Weights initialized");

        LogLossCalculator logLossCalc(1.0);

        Timer timer;
        timer.start();
        for (int64_t sampleIdx = 0; sampleIdx < NumIters; ++sampleIdx) {
            int sample[NumFields];
            createSample(sample, sampleIdx);
            int y = sampleIdx % 2 > 0 ? 1 : -1;

            float t = ffmPredict(model.weights.data(), NumFields, sample);
            logLossCalc.update(t, y);
        }
        float duration = timer.get();

        LOG("Log-loss: %14.8f", logLossCalc.get());
        LOG("duration: %f s; ops/s: %f", duration, NumIters / duration);
    }

    void gpuPredictBenchmark()
    {
        LOG("Starting prediction benchmark (GPU)");

        Model model(NumFields, 123);
        LOG("Weights initialized");
        FFMTrainer trainer(model, 10, BatchSize, 0.0001, 0.2);
        FFMPredictor predictor = trainer.createPredictor();

        HostBuffer<int> batch = FFMStatic::createHostBuffer<int>(BatchSize * (NumFields + 1));
        HostBuffer<float> hPredictResultsBuffer = FFMStatic::createHostBuffer<float>(BatchSize);

        LogLossCalculator logLossCalc(1.0);

        Timer timer;
        timer.start();

        for (int64_t sampleIdx = 0; sampleIdx < NumIters; sampleIdx += BatchSize) {
            for (int64_t batchIdx = 0; batchIdx < BatchSize; batchIdx++) {
                createSample(batch.get() + (NumFields + 1) * batchIdx, sampleIdx);
                batch.get()[(NumFields + 1) * batchIdx + NumFields] = sampleIdx % 2 > 0 ? 1 : -1;
            }

            predictor.predict(batch, BatchSize, hPredictResultsBuffer);
            for (int resultIdx = 0; resultIdx < BatchSize; ++resultIdx) {
                const int y = batch.get()[resultIdx * (model.numFields + 1) + model.numFields];
                const float t = hPredictResultsBuffer.get()[resultIdx];
                logLossCalc.update(t, y);
            }
        }
        float duration = timer.get();

        LOG("Log-loss: %14.8f", logLossCalc.get());
        LOG("duration: %f s; ops/s: %f", duration, NumIters / duration);
    }
};

int main()
{
    FFMStatic::init();
    for (int runIdx = 0; runIdx < NumRuns; ++runIdx) {
        TrainBenchmark().gpuLearnBenchmark();
        PredictBenchmark().cpuPredictBenchmark();
        PredictBenchmark().gpuPredictBenchmark();
    }
    FFMStatic::destroy();
}
