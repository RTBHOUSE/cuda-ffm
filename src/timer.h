#ifndef CUDA_TIMER_H
#define CUDA_TIMER_H

#include <chrono>

// Very simple stop watch.
struct Timer
{
    inline void start()
    {
        begin = std::chrono::high_resolution_clock::now();
    }

    inline float get()
    {
        auto now = std::chrono::high_resolution_clock::now();
        return (float) std::chrono::duration_cast<std::chrono::milliseconds>(now - begin).count() / 1000;
    }

private:
    std::chrono::high_resolution_clock::time_point begin;
};

#endif // CUDA_TIMER_H
