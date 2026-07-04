// =====================================================================
// P5.6 self-test — fast vs. exact transcendentals.
//   default build: fast SIMD approximations are ACCURATE (and correct — the
//                  old fast_exp silently computed exp(2x); fixed here).
//   -DDJEHUTI_EXACT_MATH: every fast_* upgrades to std:: (research-grade).
// Same source, compiled both ways; thresholds switch on the flag.
//
// Copyright © Co Bros & Alex Ramanaidou (Webuildcodes). All rights reserved.
// Licensed under the Non‑Exclusive Source License – see LICENSE file.
//
// =====================================================================
#include "djehuti_nn_core.hpp"
#include <cstdio>
#include <cmath>

using namespace djehuti::nn;   // makes UltimateSIMD and VECTOR_SIZE visible

static int fails = 0;
static void check(bool ok, const char* msg, double v) {
    std::printf("  [%s] %s (%.2e)\n", ok ? "PASS" : "FAIL", msg, v);
    if (!ok) fails++;
}
static double L0e(double v){ alignas(64) double b[VECTOR_SIZE]; VECTOR_STORE(b, UltimateSIMD::fast_exp(VECTOR_SET1(v))); return b[0]; }
static double L0g(double v){ alignas(64) double b[VECTOR_SIZE]; VECTOR_STORE(b, UltimateSIMD::fast_log(VECTOR_SET1(v))); return b[0]; }
static double L0f(double v){ alignas(64) double b[VECTOR_SIZE]; VECTOR_STORE(b, UltimateSIMD::fast_erf(VECTOR_SET1(v))); return b[0]; }

int main() {
#ifdef DJEHUTI_EXACT_MATH
    const char* mode = "EXACT (std::)"; double tol_exp = 1e-12, tol_log = 1e-12, tol_erf = 1e-12;
#else
    const char* mode = "FAST (SIMD approx)"; double tol_exp = 1e-5, tol_log = 1e-5, tol_erf = 1e-5;
#endif
    std::printf("Djehuti SDK — transcendental accuracy self-test (P5.6)  [%s]\n", mode);

    double me = 0, mg = 0, mf = 0;
    for (double x = -20; x <= 20; x += 0.01) { double b = std::exp(x); me = std::max(me, std::abs(L0e(x) - b) / b); }
    for (double x = 0.05; x <= 100; x += 0.02)                          mg = std::max(mg, std::abs(L0g(x) - std::log(x)));
    for (double x = -3;  x <= 3;  x += 0.005)                           mf = std::max(mf, std::abs(L0f(x) - std::erf(x)));

    std::printf("\n");
    check(me < tol_exp, "fast_exp relative error", me);
    check(mg < tol_log, "fast_log absolute error", mg);
    check(mf < tol_erf, "fast_erf absolute error", mf);

    // regression guard: fast_exp must return exp(x), NOT the old exp(2x)
    double e1 = L0e(1.0);
    check(std::abs(e1 - std::exp(1.0)) < 1e-3, "fast_exp(1)=e, not e^2", e1);

    std::printf("\n%s  (%d failure%s)\n", fails ? "FAILED" : "ALL PASS",
                fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
