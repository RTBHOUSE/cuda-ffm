#ifndef CUDA_LOG_LOSS_CALCULATOR_H
#define CUDA_LOG_LOSS_CALCULATOR_H

// LogLossCalculator helps to calculate log-loss metric
struct LogLossCalculator
{
    const double samplingFactor;

    double logLikelihood = 0;
    int64_t scaledNumSamples = 0;

    LogLossCalculator(const double samplingFactor)
            : samplingFactor(samplingFactor)
    {
    }

    void update(double t, double y)
    {
        const double multi = y > 0 ? 1 : samplingFactor;
        const double ll = calcLogLikelihood(t, y);
        ASSERT_FIN2(ll, "ll=%f, t=%f, y=%f", ll, t, y);
        logLikelihood += ll * multi;
        ASSERT_FIN2(logLikelihood, "ll=%f, t=%f, y=%f, multi=%f, logLikelihood: %f", ll, t, y, multi, logLikelihood);
        scaledNumSamples += multi;
    }

    float get() const
    {
        return logLikelihood / scaledNumSamples;
    }

private:

    double calcLogLikelihood(double t, double y) const
    {
        ASSERT(y == 1 || y == -1, "y: %f", y);
        ASSERT(t >= 0 && t <= 1, "t: %f", t);
        ASSERT_FIN(t);

        const double eps = 1e-9;

        t = std::max(std::min(1.0 - eps, t), eps);
        ASSERT_FIN(log(1.0 - t));
        y = (y + 1.0) / 2.0;

        return -((1.0 - y) * log(1.0 - t) + y * log(t));
    }
};

#endif //CUDA_LOG_LOSS_CALCULATOR_H
