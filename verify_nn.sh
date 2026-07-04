#!/bin/bash
# =====================================================================
# Djehuti NN — one-command verification for the free NN edition.
# Builds nn_recovery (full autograd & learning anchors), nn_llm_demo (GPT),
# and exact_math_test (SIMD transcendentals), then runs each and asserts
# they pass.
#   bash verify_nn.sh
# =====================================================================
set -u
cd "$(dirname "$0")"

# ---------- compiler ----------
if   command -v g++-15 &>/dev/null; then CXX=g++-15
elif command -v g++-14 &>/dev/null; then CXX=g++-14
elif command -v g++-13 &>/dev/null; then CXX=g++-13
else CXX=g++; fi

# ---------- Eigen ----------
EIGEN=""
for p in "${EIGEN_INCLUDE:-}" /usr/include/eigen3 /usr/local/include/eigen3; do
    [ -n "$p" ] && [ -d "$p" ] && EIGEN="$p" && break
done
if [ -z "$EIGEN" ]; then
    echo "ERROR: Eigen not found. Install libeigen3-dev or set EIGEN_INCLUDE=/path/to/eigen3" >&2
    exit 1
fi

# ---------- build flags (same as compile-native, but without cache tuning for simplicity) ----------
BASE="-std=c++17 -O3 -march=native -mtune=native -fopenmp -funroll-loops -ftree-vectorize -fprefetch-loop-arrays -I. -I$EIGEN"
FAST="$BASE -ffast-math -fno-math-errno -funsafe-math-optimizations -fno-trapping-math -ffinite-math-only -freciprocal-math"
STRICT="$BASE -fno-fast-math"

mkdir -p build_nn
PASS=0; FAIL=0

ok() { echo "  [PASS] $1"; PASS=$((PASS+1)); }
no() { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }

# ---------- helper: compile & run a test, require exit 0 ----------
run_test() {
    local name="$1"
    local src="$2"
    local flags="$3"
    local exe="build_nn/${name}"
    shift 3
    if $CXX $flags "$src" -o "$exe" -lpthread -lm 2>/tmp/dj_${name}.log; then
        if ./"$exe" >/tmp/dj_${name}.out 2>&1; then
            ok "$name (exit 0)"
        else
            no "$name (nonzero exit)"; tail -5 /tmp/dj_${name}.out
        fi
    else
        no "$name (build failed)"; tail -10 /tmp/dj_${name}.log
    fi
}

echo "== Building & running NN recovery (gradient‑checks + learning anchors) =="
run_test "nn_recovery" "examples/nn_recovery.cpp" "$STRICT"

echo "== Building & running GPT demo (char‑level Transformer) =="
run_test "nn_llm_demo" "examples/nn_llm_demo.cpp" "$STRICT"

# ---------- optional: exact‑math test (fast approximations vs std) ----------
echo "== Building & running exact‑math test (SIMD transcendentals) =="
if $CXX $FAST examples/exact_math_test.cpp -o build_nn/exact_math_test -lpthread -lm 2>/tmp/dj_em.log; then
    if ./build_nn/exact_math_test >/tmp/dj_em.out 2>&1; then
        ok "exact_math_test (FAST profile)"
    else
        no "exact_math_test (FAST profile)"; tail -5 /tmp/dj_em.out
    fi
else
    no "exact_math_test build"; tail -10 /tmp/dj_em.log
fi

# ---------- also test the EXACT=1 build (std:: transcendentals) ----------
if $CXX $FAST -DDJEHUTI_EXACT_MATH examples/exact_math_test.cpp -o build_nn/exact_math_test_exact -lpthread -lm 2>/tmp/dj_eme.log; then
    if ./build_nn/exact_math_test_exact >/tmp/dj_eme.out 2>&1; then
        ok "exact_math_test (EXACT=1, std:: transcendentals)"
    else
        no "exact_math_test (EXACT=1)"; tail -5 /tmp/dj_eme.out
    fi
else
    no "exact_math_test (EXACT=1) build"; tail -10 /tmp/dj_eme.log
fi

# ----------- Fraud demo test ------------------------------------------------------------
echo "== Building & running fraud detection demo (synthetic data) =="
run_test "fraud_demo" "examples/fraud_demo.cpp" "$STRICT"

# ---------- final summary ----------
echo
echo "=================================================="
echo "  NN VERIFICATION RESULT: $PASS passed, $FAIL failed"
if [ "$FAIL" -eq 0 ]; then
    echo "  ALL TESTS PASSED — the NN engine is working correctly on this machine."
else
    echo "  SOME TESTS FAILED — see above for details."
fi
echo "=================================================="
[ "$FAIL" -eq 0 ]