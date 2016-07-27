#include "constants.h"
#include "timer.h"
#include "ffm_trainer.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <ctime>

// simple FFM trainer benchmark

static const int NumRuns = 10;

struct TrainBenchmark
{
    static const int NumFields = 74;
    static const int BatchSize = 200;
    static const int NumIters = 1000 * 100;
    static const int XYsize = (NumFields + 1) * sizeof(int) * BatchSize;

    FFMTrainer * fm;

    TrainBenchmark()
    {
        Model model(NumFields);
        model.randomInit(123);
        printf("Weights initialized\n");

        fm = new FFMTrainer(model, 10, BatchSize, 0.0001, 0.2);
    }

    ~TrainBenchmark()
    {
        delete fm;
    }

    void run()
    {
        int * h_xy = fm->createHostBuffer<int>(BatchSize * (NumFields + 1));

        for (int iter = 0; iter < NumIters; iter += BatchSize) {
            createRandomBatch(iter, h_xy);
            fm->learn(h_xy, BatchSize);
        }
        fm->destroyHostBuffer(h_xy);
    }

private:
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

void benchmark()
{
    TrainBenchmark benchmark;

    printf("Start\n");
    for (int i = 0; i < NumRuns; ++i) {
        Timer timer;
        timer.start();
        benchmark.run();
        float duration = timer.get();
        printf("duration=%10.5f\n", duration);
    }
}

int main()
{
    FFMTrainerStatic::init();
    benchmark();
    FFMTrainerStatic::destroy();
}
