#include <cstdlib>
#include <cmath>
#include <inttypes.h>
#include <pmmintrin.h>
#include <immintrin.h>

static uint32_t const FACTOR_SIZE = 4;

// Horizontal sum of 8 floats
inline float horizontal_add8(__m256 x) {
    __m128 const hi256 = _mm256_extractf128_ps(x, 1);
    __m128 const lo256 = _mm256_castps256_ps128(x);
    __m128 const sum2 = _mm_add_ps(lo256, hi256); // x3 + x7, x2 + x6, x1 + x5, x0 + x4

    __m128 const lo128 = sum2;
    __m128 const hi128 = _mm_movehl_ps(sum2, sum2);
    __m128 const sum4 = _mm_add_ps(lo128, hi128); //  _, _, (x3 + x7) + (x1 + x5), (x0 + x4) + (x2 + x6)

    __m128 const lo = sum4;
    __m128 const hi = _mm_shuffle_ps(sum4, sum4, 0x1);
    __m128 const sum8 = _mm_add_ss(lo, hi); // _, _, _, [ (x3 + x7) + (x1 + x5) ]  +  [ (x0 + x4) + (x2 + x6) ]

    return _mm_cvtss_f32(sum8);
}

// Horizontal sum of 4 floats
inline float horizontal_add4(__m128 x) {
    __m128 const sum2 = _mm_hadd_ps(x, x); // x3 + x2, x1 + x0, _, _
    __m128 const sum4 = _mm_hadd_ps(sum2, sum2); // x3 + x2  +  x1 + x0, _, _

    return _mm_cvtss_f32(sum4);
}

// Performs:
//
//  [ sum ] += [ w1 ] * [ w2lo, w2hi ]
//
// in a vectorized manner.
//
// Where:
//   - sum points to 8 floats
//   - w1 points to 8 floats
//   - w2lo points to 4 floats
//   - w2hi points to 4 floats
inline void mul_add8(const float * const w1, const float * const w2lo, const float * const w2hi, __m256 & sum) {
    __m256 YMMw1 = _mm256_loadu_ps(w1);
    __m256 YMMw2 = _mm256_setzero_ps();

    __m128 const XMMw2lo = _mm_load_ps(w2lo);
    __m128 const XMMw2hi = _mm_load_ps(w2hi);
    YMMw2 = _mm256_insertf128_ps(YMMw2, XMMw2lo, 0);
    YMMw2 = _mm256_insertf128_ps(YMMw2, XMMw2hi, 1);

    // GCC will translate it to fused multiply–add (_mm256_fmadd_ps) anyway if -mfma switch is enabled
    sum = _mm256_add_ps(sum, _mm256_mul_ps(YMMw1, YMMw2));
}

// Performs:
//
//   [ sum ] += [ w1 ] * [ w2 ]
//
// in a vectorized manner.
//
// Where:
//   - sum points to 4 floats
//   - w1 points to 4 floats
//   - w2 points to 4 floats
inline void mul_add4(const float * const w1, const float * const w2, __m128 & sum) {
    __m128 const XMMw1 = _mm_load_ps(w1);
    __m128 const XMMw2 = _mm_load_ps(w2);
    sum = _mm_add_ps(sum, _mm_mul_ps(XMMw1, XMMw2));
}

// Performs:
//
//  [ sum ] += dot(weights[feature1, field2], weights[feature2, field1])
//
// in a vectorized manner.
inline void dot_factors2(uint32_t numFields, const float * const weights,
        uint32_t const feature1, uint32_t const feature2,
        uint32_t const field1, uint32_t const field2,
        __m128 & sum
) {
    uint64_t const align0 = (uint64_t) FACTOR_SIZE;
    uint64_t const align1 = (uint64_t) numFields * align0;

    const float * const w1a = weights + feature1 * align1 + field2 * align0;
    const float * const w2a = weights + feature2 * align1 + field1 * align0;

    mul_add4(w1a, w2a, sum);
}

// Performs:
//
//  [ sum ] += dot(weights[feature1, field2], weights[feature2, field1]) +
//             dot(weights[feature1, field2 + 1], weights[feature3, field1])
//
// in a vectorized manner.
inline void dot_factors4(uint32_t numFields, const float * const weights,
        uint32_t const feature1, uint32_t const feature2, uint32_t feature3,
        uint32_t const field1, uint32_t const field2,
        __m256 & sum
) {
    uint64_t const align0 = (uint64_t) FACTOR_SIZE;
    uint64_t const align1 = (uint64_t) numFields * align0;

    const float * const w1  =  weights + feature1 * align1 + field2 * align0;
    const float * const w2lo = weights + feature2 * align1 + field1 * align0;
    const float * const w2hi = weights + feature3 * align1 + field1 * align0;

    mul_add8(w1, w2lo, w2hi, sum);
}

// Computes sigmoid (logit) function
inline float sigmoid(float x) {
    return 1.0 / (1.0 + expf(-x));
}

//
// Performs Field-aware Factorization Machine prediction in a vectorized manner.
//
// PART 1
// ------
//
// Dot two adjacent factors using AVX _mm256_mul_ps (two factors, 8 floats a time).
//
// Same letters below do not mean same factors, but that they will be processed at once
// e.g. all A's will be multiplied at once, than all B's and so on.
//
//     * A A B B
//     A * C C _
//     A C * D D
//     B C D * _
//     B _ D _ *
//
//
// PART 2
// ------
//
// Dot leftovers using SSE _mm_mul_ps (one factor, 4 floats a time).
//
//     * _ _ _ _
//     _ * _ _ E
//     _ _ * _ _
//     _ _ _ * F
//     _ E _ F *
//
// Two E's will be multiplied at once, and then two F's.
float ffmPredict(const float * weights, uint32_t numFields, const int32_t * features) {
    uint32_t leftoverStartIdx = numFields % 2;

    __m256 mainSum = _mm256_setzero_ps();
    __m128 leftoversSum = _mm_setzero_ps();

    for(uint32_t field1 = 0; field1 < numFields; field1++)
    {
        uint32_t const feature1 = features[field1];

        for(uint32_t field2 = field1 + 1; field2 < numFields - 1; field2 += 2)
        {
            dot_factors4(numFields, weights, feature1, features[field2], features[field2 + 1], field1, field2, mainSum);
        }
    }

    // process leftovers
    for(uint32_t field1 = leftoverStartIdx; field1 < numFields; field1 += 2) {
        dot_factors2(numFields, weights, features[field1], features[numFields-1], field1, numFields-1, leftoversSum);
    }

    float const v = 1.0 / static_cast<float>(numFields);
    float const t = (horizontal_add8(mainSum) + horizontal_add4(leftoversSum)) * v;

    return sigmoid(t);
}

