#!/bin/bash
# =====================================================================
# Djehuti NN — portable build for the free NN edition.
# Auto-detects CPU cache, AVX‑512, and sets optimal flags.
#
#   ./compile-native.sh                 # builds examples/nn_demo.cpp (default)
#   DEMO=nn_llm_demo ./compile-native.sh # builds the GPT demo
#   DEMO=nn_recovery
#   DEMO=fraud_demo    # builds the fraud detection demo ./compile-native.sh # builds the full verification suite
#   STRICT=1 ./compile-native.sh        # IEEE-strict math
#   EXACT=1  ./compile-native.sh        # exact std::transcendentals
#   EIGEN_INCLUDE=/path ./compile-native.sh
# =====================================================================
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"

# ---- which demo to build (default: nn_demo) ----
DEMO="${DEMO:-nn_demo}"

# ---- compiler ----
if   command -v g++-15 &>/dev/null; then CXX=g++-15
elif command -v g++-14 &>/dev/null; then CXX=g++-14
elif command -v g++-13 &>/dev/null; then CXX=g++-13
else CXX=g++; fi

# ---- Eigen ----
EIG=""
for p in "$EIGEN_INCLUDE" /usr/include/eigen3 /usr/local/include/eigen3; do
    [ -n "$p" ] && [ -d "$p" ] && EIG="-I$p" && break
done
if [ -z "$EIG" ]; then
    echo "ERROR: Eigen not found. Install libeigen3-dev or set EIGEN_INCLUDE=/path/to/eigen3" >&2
    exit 1
fi

# ---- auto‑tune cache ----
CACHE=""
L1=$(getconf LEVEL1_DCACHE_SIZE 2>/dev/null || echo 0)
L2=$(getconf LEVEL2_CACHE_SIZE 2>/dev/null || echo 0)
L3=$(getconf LEVEL3_CACHE_SIZE 2>/dev/null || echo 0)
[ "${L1:-0}" -gt 0 ] 2>/dev/null && CACHE="$CACHE -DL1D_CACHE_SIZE=$L1"
[ "${L2:-0}" -gt 0 ] 2>/dev/null && CACHE="$CACHE -DL2_CACHE_SIZE=$L2"
[ "${L3:-0}" -gt 0 ] 2>/dev/null && CACHE="$CACHE -DL3_CACHE_SIZE=$L3"
TH=$(nproc 2>/dev/null || echo 4)
CACHE="$CACHE -DMAX_THREADS=$TH"

# ---- math profile ----
if [ "${STRICT:-0}" = "1" ]; then
    MATH="-fno-fast-math"
    echo "math profile : STRICT (IEEE-reproducible)"
else
    MATH="-ffast-math -fno-math-errno -funsafe-math-optimizations -fno-trapping-math -ffinite-math-only -freciprocal-math"
    echo "math profile : FAST (default — set STRICT=1 for IEEE reproducibility)"
fi
if [ "${EXACT:-0}" = "1" ]; then
    MATH="$MATH -DDJEHUTI_EXACT_MATH"
    echo "transcendental: EXACT (fast_*->std::, full precision)"
fi

OUT="$HERE/$DEMO"
SRC="$HERE/examples/${DEMO}.cpp"

if [ ! -f "$SRC" ]; then
    echo "ERROR: source file not found: $SRC" >&2
    exit 1
fi

echo "compiler     : $CXX"
echo "eigen        : $EIG"
echo "cpu tuning   : -march=native  cache[$CACHE ]"
echo "demo         : $DEMO"
echo "building     : $OUT"

$CXX -std=c++17 -O3 -march=native -mtune=native -flto -fopenmp \
     -funroll-loops -ftree-vectorize -fprefetch-loop-arrays \
     $MATH $CACHE $EIG -I"$HERE" \
     "$SRC" -o "$OUT" \
     -lpthread -lm

echo "OK: $OUT"
if objdump -d "$OUT" 2>/dev/null | grep -qm1 -E 'zmm[0-9]'; then
    echo "simd         : AVX-512 (zmm) path active on this CPU"
else
    echo "simd         : AVX2 path (no AVX-512 on this CPU)"
fi