#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

// Contains few helper functions/data structures.

#include <cassert>
#include <memory>
#include <cmath>

namespace asserts {
static float const inf = 1e12;
}

#define ASSERT(x, fmt, args...) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "ASSERTION FAILED. Message: " fmt "\n", args); \
            assert(x); \
        }\
    } while(false)

#define ASSERT_FIN(x) \
    do {\
        assert(std::isfinite(x));\
        assert(x < asserts::inf);\
        assert(x > -asserts::inf);\
    } while (false)

#define ASSERT_FIN2(x, fmt, args...) \
    do {\
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
