#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

// Contains few helper functions/data structures.

#include <cassert>
#include <memory>
#include <cmath>

namespace asserts {
static double const inf = 1e30;
}

#define ASSERT(x, fmt, args...) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "ASSERTION FAILED. Message: " fmt "\n", args); \
            assert(x); \
        }\
        assert(x); /* for fast-math optimiztions */ \
    } while(false)

#define ASSERT_FIN(x) \
    do {\
        assert(!std::isnan(x)); \
        assert(std::isfinite(x));\
        assert(x < asserts::inf);\
        assert(x > -asserts::inf);\
    } while (false)

#define ASSERT_FIN2(x, fmt, args...) \
    do {\
        assert(!std::isnan(x)); \
        ASSERT(std::isfinite(x), fmt, args); \
        ASSERT(x < asserts::inf, fmt, args); \
        ASSERT(x > -asserts::inf, fmt, args); \
    } while (false)

#define LOG(fmt, ...) ::fprintf(stdout, fmt "\n", ## __VA_ARGS__)

template <typename T>
struct Option
{
    Option(const T & value)
            : value(new T(value))
    {
    }

    Option()
    {
    }

    Option & operator=(T const & newValue)
    {
        value.reset(new T(newValue));
        return *this;
    }

    bool isEmpty() const
    {
        return value.get() == NULL;
    }

    T & get()
    {
        return *value.get();
    }

    T const & cget() const
    {
        return *value.get();
    }

private:
    std::shared_ptr<T> value;
};

#endif //CUDA_UTILS_H
