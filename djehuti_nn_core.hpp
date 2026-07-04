// djehuti_nn_core.hpp – minimal dependencies for the NN engine
//
// Copyright © Co Bros & Alex Ramanaidou (Webuildcodes). All rights reserved.
// Licensed under the Non‑Exclusive Source License – see LICENSE file.
#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <fstream>
#include <string>
#include <cstring>
#include <omp.h>
#include <immintrin.h>
#include <Eigen/Dense>

namespace djehuti { namespace nn {

using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// ---------- SIMD macros (copied from djehuti_core.hpp) ----------
#ifndef L1D_CACHE_SIZE
#define L1D_CACHE_SIZE 32768
#endif
#ifndef L2_CACHE_SIZE
#define L2_CACHE_SIZE 524288
#endif
#ifndef L3_CACHE_SIZE
#define L3_CACHE_SIZE 16777216
#endif
#ifndef CACHE_LINE_SIZE_MACRO
#define CACHE_LINE_SIZE_MACRO 64
#endif
#ifndef PREFETCH_DISTANCE
#define PREFETCH_DISTANCE 32
#endif

#ifdef __AVX512F__
#define VECTOR_TYPE __m512d
#define VECTOR_LOAD _mm512_load_pd
#define VECTOR_LOADU _mm512_loadu_pd
#define VECTOR_STORE _mm512_store_pd
#define VECTOR_ADD _mm512_add_pd
#define VECTOR_SUB _mm512_sub_pd
#define VECTOR_MUL _mm512_mul_pd
#define VECTOR_DIV _mm512_div_pd
#define VECTOR_FMA _mm512_fmadd_pd
#define VECTOR_SET1 _mm512_set1_pd
#define VECTOR_ZERO _mm512_setzero_pd
#define VECTOR_SQRT _mm512_sqrt_pd
#define VECTOR_CMP_LT(x, y) _mm512_cmp_pd_mask(x, y, _CMP_LT_OQ)
#define VECTOR_BLENDV(a, b, mask) _mm512_mask_blend_pd(mask, a, b)
static constexpr size_t VECTOR_SIZE = 8;
#else
#define VECTOR_TYPE __m256d
#define VECTOR_LOAD  _mm256_load_pd
#define VECTOR_LOADU _mm256_loadu_pd
#define VECTOR_STORE _mm256_store_pd
#define VECTOR_ADD _mm256_add_pd
#define VECTOR_SUB _mm256_sub_pd
#define VECTOR_MUL _mm256_mul_pd
#define VECTOR_DIV _mm256_div_pd
#define VECTOR_FMA _mm256_fmadd_pd
#define VECTOR_SET1 _mm256_set1_pd
#define VECTOR_ZERO _mm256_setzero_pd
#define VECTOR_SQRT _mm256_sqrt_pd
#define VECTOR_CMP_LT(x, y) _mm256_cmp_pd(x, y, _CMP_LT_OQ)
#define VECTOR_BLENDV(a, b, mask) _mm256_blendv_pd(a, b, mask)
static constexpr size_t VECTOR_SIZE = 4;
#endif

// ---------- SIMD engine (used by expT, logT, softmax, etc.) ----------
class UltimateSIMD {
public:
    static inline VECTOR_TYPE fast_exp(VECTOR_TYPE x) {
#ifdef DJEHUTI_EXACT_MATH
        alignas(64) double _b[VECTOR_SIZE];
        VECTOR_STORE(_b, x);
        for (size_t _i = 0; _i < VECTOR_SIZE; ++_i) _b[_i] = std::exp(_b[_i]);
        return VECTOR_LOAD(_b);
#else
        const double ln2 = 0.6931471805599453, inv_ln2 = 1.4426950408889634;
        alignas(64) double xn[VECTOR_SIZE], kb[VECTOR_SIZE], p2b[VECTOR_SIZE];
        VECTOR_STORE(xn, VECTOR_MUL(x, VECTOR_SET1(inv_ln2)));
        for (size_t _i = 0; _i < VECTOR_SIZE; ++_i) {
            double kf = std::nearbyint(xn[_i]);
            kb[_i] = kf;
            int64_t ki = (int64_t)kf;
            uint64_t bits = (uint64_t)(ki + 1023) << 52;
            std::memcpy(&p2b[_i], &bits, sizeof(double));
        }
        VECTOR_TYPE kvec  = VECTOR_LOAD(kb);
        VECTOR_TYPE pow2k = VECTOR_LOAD(p2b);
        VECTOR_TYPE r = VECTOR_SUB(x, VECTOR_MUL(kvec, VECTOR_SET1(ln2)));
        VECTOR_TYPE y = VECTOR_MUL(r, VECTOR_SET1(0.5));
        VECTOR_TYPE p2 = VECTOR_SET1(0.5);
        VECTOR_TYPE p3 = VECTOR_SET1(0.16666666666666666);
        VECTOR_TYPE p4 = VECTOR_SET1(0.041666666666666664);
        VECTOR_TYPE y2 = VECTOR_MUL(y, y);
        VECTOR_TYPE y3 = VECTOR_MUL(y2, y);
        VECTOR_TYPE y4 = VECTOR_MUL(y2, y2);
        VECTOR_TYPE numerator = VECTOR_SET1(1.0);
        numerator = VECTOR_ADD(numerator, y);
        numerator = VECTOR_ADD(numerator, VECTOR_MUL(p2, y2));
        numerator = VECTOR_ADD(numerator, VECTOR_MUL(p3, y3));
        numerator = VECTOR_ADD(numerator, VECTOR_MUL(p4, y4));
        VECTOR_TYPE denominator = VECTOR_SET1(1.0);
        denominator = VECTOR_SUB(denominator, y);
        denominator = VECTOR_ADD(denominator, VECTOR_MUL(p2, y2));
        denominator = VECTOR_SUB(denominator, VECTOR_MUL(p3, y3));
        denominator = VECTOR_ADD(denominator, VECTOR_MUL(p4, y4));
        return VECTOR_MUL(pow2k, VECTOR_DIV(numerator, denominator));
#endif
    }

    static inline VECTOR_TYPE fast_erf(VECTOR_TYPE x) {
#ifdef DJEHUTI_EXACT_MATH
        alignas(64) double _b[VECTOR_SIZE];
        VECTOR_STORE(_b, x);
        for (size_t _i = 0; _i < VECTOR_SIZE; ++_i) _b[_i] = std::erf(_b[_i]);
        return VECTOR_LOAD(_b);
#else
        VECTOR_TYPE a1 = VECTOR_SET1(0.254829592);
        VECTOR_TYPE a2 = VECTOR_SET1(-0.284496736);
        VECTOR_TYPE a3 = VECTOR_SET1(1.421413741);
        VECTOR_TYPE a4 = VECTOR_SET1(-1.453152027);
        VECTOR_TYPE a5 = VECTOR_SET1(1.061405429);
        VECTOR_TYPE p  = VECTOR_SET1(0.3275911);
        VECTOR_TYPE neg_x = VECTOR_SUB(VECTOR_SET1(0.0), x);
        VECTOR_TYPE abs_x = VECTOR_BLENDV(x, neg_x, VECTOR_CMP_LT(x, VECTOR_SET1(0.0)));
        VECTOR_TYPE t  = VECTOR_DIV(VECTOR_SET1(1.0),
                         VECTOR_ADD(VECTOR_SET1(1.0), VECTOR_MUL(p, abs_x)));
        VECTOR_TYPE poly = VECTOR_FMA(a5, t, a4);
        poly = VECTOR_FMA(poly, t, a3);
        poly = VECTOR_FMA(poly, t, a2);
        poly = VECTOR_FMA(poly, t, a1);
        poly = VECTOR_MUL(poly, t);
        VECTOR_TYPE exp_neg_x2 = fast_exp(VECTOR_SUB(VECTOR_SET1(0.0), VECTOR_MUL(x, x)));
        poly = VECTOR_MUL(poly, exp_neg_x2);
        VECTOR_TYPE result = VECTOR_SUB(VECTOR_SET1(1.0), poly);
        VECTOR_TYPE sign = VECTOR_BLENDV(VECTOR_SET1(1.0), VECTOR_SET1(-1.0),
                                         VECTOR_CMP_LT(x, VECTOR_SET1(0.0)));
        return VECTOR_MUL(sign, result);
#endif
    }

    static inline VECTOR_TYPE fast_norm_cdf(VECTOR_TYPE x) {
        VECTOR_TYPE sqrt2 = VECTOR_SET1(1.41421356237);
        return VECTOR_MUL(VECTOR_SET1(0.5), VECTOR_ADD(VECTOR_SET1(1.0), fast_erf(VECTOR_DIV(x, sqrt2))));
    }

    static inline __m256d fast_log(__m256d x) {
#ifdef DJEHUTI_EXACT_MATH
        alignas(32) double _b[4];
        _mm256_store_pd(_b, x);
        for (int _i = 0; _i < 4; ++_i) _b[_i] = std::log(_b[_i]);
        return _mm256_load_pd(_b);
#else
        __m256i xi = _mm256_castpd_si256(x);
        __m256i exp_bits = _mm256_srli_epi64(
            _mm256_and_si256(xi, _mm256_set1_epi64x(0x7FF0000000000000LL)), 52);
        __m256i exp_unbiased = _mm256_sub_epi64(exp_bits, _mm256_set1_epi64x(1023));
        double e0 = (double)(int64_t)_mm256_extract_epi64(exp_unbiased, 0);
        double e1 = (double)(int64_t)_mm256_extract_epi64(exp_unbiased, 1);
        double e2 = (double)(int64_t)_mm256_extract_epi64(exp_unbiased, 2);
        double e3 = (double)(int64_t)_mm256_extract_epi64(exp_unbiased, 3);
        __m256d exp_d = _mm256_set_pd(e3, e2, e1, e0);
        __m256i mantissa_bits = _mm256_or_si256(
            _mm256_and_si256(xi, _mm256_set1_epi64x(0x000FFFFFFFFFFFFFLL)),
            _mm256_set1_epi64x(0x3FF0000000000000LL));
        __m256d m = _mm256_castsi256_pd(mantissa_bits);
        __m256d one = _mm256_set1_pd(1.0);
        __m256d s  = _mm256_div_pd(_mm256_sub_pd(m, one), _mm256_add_pd(m, one));
        __m256d s2 = _mm256_mul_pd(s, s);
        __m256d poly = _mm256_set1_pd(2.0/9);
        poly = _mm256_fmadd_pd(poly, s2, _mm256_set1_pd(2.0/7));
        poly = _mm256_fmadd_pd(poly, s2, _mm256_set1_pd(2.0/5));
        poly = _mm256_fmadd_pd(poly, s2, _mm256_set1_pd(2.0/3));
        poly = _mm256_fmadd_pd(poly, s2, _mm256_set1_pd(2.0));
        poly = _mm256_mul_pd(poly, s);
        __m256d ln2 = _mm256_set1_pd(0.6931471805599453);
        return _mm256_fmadd_pd(ln2, exp_d, poly);
#endif
    }

#ifdef __AVX512F__
    static inline __m512d fast_log(__m512d x) {
#ifdef DJEHUTI_EXACT_MATH
        alignas(64) double _b[8];
        _mm512_store_pd(_b, x);
        for (int _i = 0; _i < 8; ++_i) _b[_i] = std::log(_b[_i]);
        return _mm512_load_pd(_b);
#else
        __m512i xi = _mm512_castpd_si512(x);
        __m512i exp_bits = _mm512_srli_epi64(
            _mm512_and_si512(xi, _mm512_set1_epi64(0x7FF0000000000000LL)), 52);
        __m512i exp_unbiased = _mm512_sub_epi64(exp_bits, _mm512_set1_epi64(1023));
        __m512d exp_d = _mm512_cvtepi64_pd(exp_unbiased);
        __m512i mantissa_bits = _mm512_or_si512(
            _mm512_and_si512(xi, _mm512_set1_epi64(0x000FFFFFFFFFFFFFLL)),
            _mm512_set1_epi64(0x3FF0000000000000LL));
        __m512d m = _mm512_castsi512_pd(mantissa_bits);
        __m512d one = _mm512_set1_pd(1.0);
        __m512d s  = _mm512_div_pd(_mm512_sub_pd(m, one), _mm512_add_pd(m, one));
        __m512d s2 = _mm512_mul_pd(s, s);
        __m512d poly = _mm512_set1_pd(2.0/9);
        poly = _mm512_fmadd_pd(poly, s2, _mm512_set1_pd(2.0/7));
        poly = _mm512_fmadd_pd(poly, s2, _mm512_set1_pd(2.0/5));
        poly = _mm512_fmadd_pd(poly, s2, _mm512_set1_pd(2.0/3));
        poly = _mm512_fmadd_pd(poly, s2, _mm512_set1_pd(2.0));
        poly = _mm512_mul_pd(poly, s);
        __m512d ln2 = _mm512_set1_pd(0.6931471805599453);
        return _mm512_fmadd_pd(ln2, exp_d, poly);
#endif
    }
#endif

    static inline double dot_product(const double* __restrict__ a, const double* __restrict__ b, size_t n) {
        VECTOR_TYPE sum = VECTOR_ZERO();
        size_t i = 0;
        const size_t step = VECTOR_SIZE;
        for (; i + step <= n; i += step) {
            VECTOR_TYPE va = VECTOR_LOADU(a + i);   // unaligned: a/b are arbitrary caller pointers
            VECTOR_TYPE vb = VECTOR_LOADU(b + i);
            sum = VECTOR_FMA(va, vb, sum);
        }
        alignas(64) double result[VECTOR_SIZE];
        VECTOR_STORE(result, sum);
        double total = 0;
        for (size_t j = 0; j < VECTOR_SIZE; ++j) total += result[j];
        for (; i < n; ++i) total += a[i] * b[i];
        return total;
    }
};

}} // namespace djehuti::nn