// Djehuti SDK NN / Deep-Learning / RL suite (non-exclusive source). Depends only on core.
//
// Copyright © Co Bros & Alex Ramanaidou (Webuildcodes). All rights reserved.
// Licensed under the Non‑Exclusive Source License – see LICENSE file.
//
// A standalone, self-hostable AI product: a tensor-level REVERSE-MODE AUTOGRAD engine
// (PyTorch/tinygrad-shaped) + a Module/Layer API on top. Every layer composes primitive ops
// whose backward is automatic, so any architecture (MLP / CNN / RNN-LSTM-GRU / Transformer)
// trains with no hand-derived gradients — and every op is gradient-checkable (autograd vs
// finite-difference), which is the "we don't sell a toy" proof.
//
// This header (built incrementally, each piece gradient-checked before the next):
//   [1] Autograd core: Tensor (flat N-d value+grad), ops, reverse tape, backward().   <-- HERE
//   later: layers, optimizers, losses, training, PCA, anomaly, RL, the self-hosted LLM.
//
// Header-only, Eigen + the core perf layer (AVX2 floor + auto AVX-512, OpenMP). double precision.
#pragma once
#include "djehuti_nn_core.hpp"
#include <vector>
#include <memory>
#include <functional>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
#include <unordered_set>
#include <fstream>
#include <string>

namespace djehuti { namespace nn {

using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// ----------------------------------------------------------------------------------
// Autograd node. A Tensor holds a flat row-major value buffer + its gradient, an N-d
// shape, the parents it was computed from, and a backward closure that pushes this
// node's gradient into its parents'. The graph is a DAG (parents only point backward),
// so shared_ptr ownership is cycle-free; backward closures capture parents by shared_ptr
// and the OUTPUT by raw pointer (capturing the output by shared_ptr would make a cycle).
// ----------------------------------------------------------------------------------
struct TensorImpl;
using TensorPtr = std::shared_ptr<TensorImpl>;

struct TensorImpl {
    std::vector<int> shape;
    Eigen::VectorXd v;                       // values, flat row-major
    Eigen::VectorXd g;                       // gradients, same layout
    bool requires_grad = false;
    std::vector<TensorPtr> parents;
    std::function<void()> backward_fn;       // accumulate this->g into parents' g

    int size() const { return (int)v.size(); }
    int dim(int i) const { return shape[i]; }
    int rank() const { return (int)shape.size(); }
};

class Tensor {
public:
    TensorPtr p;
    Tensor() = default;
    explicit Tensor(TensorPtr ptr) : p(std::move(ptr)) {}

    // ---- factories ----
    static Tensor zeros(const std::vector<int>& shape, bool requires_grad = false) {
        auto t = std::make_shared<TensorImpl>();
        t->shape = shape;
        int n = numel(shape);
        t->v = Eigen::VectorXd::Zero(n);
        t->g = Eigen::VectorXd::Zero(n);
        t->requires_grad = requires_grad;
        return Tensor(t);
    }
    static Tensor from(const std::vector<int>& shape, const Eigen::VectorXd& vals, bool requires_grad = false) {
        Tensor t = zeros(shape, requires_grad);
        t.p->v = vals;
        return t;
    }
    static Tensor scalar(double x, bool requires_grad = false) {
        Tensor t = zeros({1}, requires_grad); t.p->v(0) = x; return t;
    }
    // Kaiming/He-flavoured normal init (good default for ReLU nets).
    static Tensor randn(const std::vector<int>& shape, double std_dev, uint64_t seed, bool requires_grad = true) {
        Tensor t = zeros(shape, requires_grad);
        std::mt19937_64 g(seed); std::normal_distribution<double> nd(0.0, std_dev);
        for (int i = 0; i < t.p->size(); i++) t.p->v(i) = nd(g);
        return t;
    }

    // ---- accessors ----
    const std::vector<int>& shape() const { return p->shape; }
    int size() const { return p->size(); }
    int rank() const { return p->rank(); }
    Eigen::VectorXd& value() { return p->v; }
    Eigen::VectorXd& grad()  { return p->g; }
    double item() const { return p->v(0); }
    void zeroGrad() { p->g.setZero(); }

    // view a 2-D tensor as a row-major Eigen matrix (no copy)
    Eigen::Map<RowMat> mat() const {
        return Eigen::Map<RowMat>(p->v.data(), p->shape[0], p->shape[1]);
    }
    Eigen::Map<RowMat> gmat() const {
        return Eigen::Map<RowMat>(p->g.data(), p->shape[0], p->shape[1]);
    }

    // ---- reverse-mode backward from a scalar root ----
    void backward() {
        std::vector<TensorImpl*> topo;
        std::unordered_set<TensorImpl*> seen;
        std::function<void(TensorImpl*)> build = [&](TensorImpl* n) {
            if (seen.count(n)) return; seen.insert(n);
            for (auto& par : n->parents) build(par.get());
            topo.push_back(n);
        };
        build(p.get());
        p->g.setOnes();                                  // d(root)/d(root) = 1 (root is scalar)
        for (auto it = topo.rbegin(); it != topo.rend(); ++it)
            if ((*it)->backward_fn) (*it)->backward_fn();
    }

    static int numel(const std::vector<int>& s) {
        int n = 1; for (int d : s) n *= d; return n;
    }
};

// helper: make an op-result tensor wired to its parents + a backward closure
namespace detail {
    inline Tensor result(const std::vector<int>& shape, std::vector<TensorPtr> parents) {
        Tensor out = Tensor::zeros(shape, false);
        out.p->parents = std::move(parents);
        return out;
    }
}

// ===================================================================================
// Primitive differentiable ops. Each: compute value, then set out's backward closure
// that accumulates out.g into the parents' g via the op's local rule.
// ===================================================================================

// elementwise add (same shape)
inline Tensor operator+(const Tensor& a, const Tensor& b) {
    Tensor out = detail::result(a.shape(), {a.p, b.p});
    out.p->v = a.p->v + b.p->v;
    TensorImpl* o = out.p.get(); TensorImpl* ap = a.p.get(); TensorImpl* bp = b.p.get();
    out.p->backward_fn = [o, ap, bp]() { ap->g += o->g; bp->g += o->g; };
    return out;
}
inline Tensor operator-(const Tensor& a, const Tensor& b) {
    Tensor out = detail::result(a.shape(), {a.p, b.p});
    out.p->v = a.p->v - b.p->v;
    TensorImpl* o = out.p.get(); TensorImpl* ap = a.p.get(); TensorImpl* bp = b.p.get();
    out.p->backward_fn = [o, ap, bp]() { ap->g += o->g; bp->g -= o->g; };
    return out;
}
// elementwise (Hadamard) product (same shape)
inline Tensor operator*(const Tensor& a, const Tensor& b) {
    Tensor out = detail::result(a.shape(), {a.p, b.p});
    out.p->v = a.p->v.cwiseProduct(b.p->v);
    TensorImpl* o = out.p.get(); TensorImpl* ap = a.p.get(); TensorImpl* bp = b.p.get();
    out.p->backward_fn = [o, ap, bp]() {
        ap->g += o->g.cwiseProduct(bp->v);
        bp->g += o->g.cwiseProduct(ap->v);
    };
    return out;
}
// scalar multiply
inline Tensor operator*(const Tensor& a, double s) {
    Tensor out = detail::result(a.shape(), {a.p});
    out.p->v = a.p->v * s;
    TensorImpl* o = out.p.get(); TensorImpl* ap = a.p.get();
    out.p->backward_fn = [o, ap, s]() { ap->g += o->g * s; };
    return out;
}
inline Tensor operator*(double s, const Tensor& a) { return a * s; }

// matrix multiply: a (N,K) x b (K,M) -> (N,M).  dA = dC Bᵀ ; dB = Aᵀ dC.
inline Tensor matmul(const Tensor& a, const Tensor& b) {
    const int N = a.shape()[0], K = a.shape()[1], M = b.shape()[1];
    Tensor out = detail::result({N, M}, {a.p, b.p});
    out.mat().noalias() = a.mat() * b.mat();
    TensorImpl* o = out.p.get(); TensorImpl* ap = a.p.get(); TensorImpl* bp = b.p.get();
    out.p->backward_fn = [o, ap, bp, N, K, M]() {
        Eigen::Map<RowMat> dC(o->g.data(), N, M);
        Eigen::Map<RowMat> A(ap->v.data(), N, K), B(bp->v.data(), K, M);
        Eigen::Map<RowMat> dA(ap->g.data(), N, K), dB(bp->g.data(), K, M);
        dA.noalias() += dC * B.transpose();
        dB.noalias() += A.transpose() * dC;
    };
    return out;
}

// add a per-column bias: x (N,M) + b (M) broadcast over rows.  db = colsum(dY).
inline Tensor addBias(const Tensor& x, const Tensor& b) {
    const int N = x.shape()[0], M = x.shape()[1];
    Tensor out = detail::result({N, M}, {x.p, b.p});
    Eigen::Map<RowMat> X(x.p->v.data(), N, M), Y(out.p->v.data(), N, M);
    Y = X; for (int i = 0; i < N; i++) Y.row(i) += b.p->v.transpose();
    TensorImpl* o = out.p.get(); TensorImpl* xp = x.p.get(); TensorImpl* bp = b.p.get();
    out.p->backward_fn = [o, xp, bp, N, M]() {
        Eigen::Map<RowMat> dY(o->g.data(), N, M);
        xp->g += o->g;                                   // dX = dY
        for (int i = 0; i < N; i++) bp->g += dY.row(i).transpose();   // db = Σ_rows dY
    };
    return out;
}

// ---- activations ----
inline Tensor relu(const Tensor& a) {
    Tensor out = detail::result(a.shape(), {a.p});
    out.p->v = a.p->v.cwiseMax(0.0);
    TensorImpl* o = out.p.get(); TensorImpl* ap = a.p.get();
    out.p->backward_fn = [o, ap]() {
        for (int i = 0; i < ap->size(); i++) ap->g(i) += (ap->v(i) > 0.0 ? o->g(i) : 0.0);
    };
    return out;
}
inline Tensor sigmoidT(const Tensor& a) {
    Tensor out = detail::result(a.shape(), {a.p});
    for (int i = 0; i < a.size(); i++) out.p->v(i) = 1.0 / (1.0 + std::exp(-a.p->v(i)));
    TensorImpl* o = out.p.get();
    out.p->backward_fn = [o, ap = a.p.get()]() {
        for (int i = 0; i < ap->size(); i++) { double s = o->v(i); ap->g(i) += o->g(i) * s * (1.0 - s); }
    };
    return out;
}
inline Tensor tanhT(const Tensor& a) {
    Tensor out = detail::result(a.shape(), {a.p});
    for (int i = 0; i < a.size(); i++) out.p->v(i) = std::tanh(a.p->v(i));
    TensorImpl* o = out.p.get();
    out.p->backward_fn = [o, ap = a.p.get()]() {
        for (int i = 0; i < ap->size(); i++) { double t = o->v(i); ap->g(i) += o->g(i) * (1.0 - t * t); }
    };
    return out;
}

// ---- reductions (to a scalar) ----
inline Tensor sum(const Tensor& a) {
    Tensor out = detail::result({1}, {a.p});
    out.p->v(0) = a.p->v.sum();
    TensorImpl* o = out.p.get();
    out.p->backward_fn = [o, ap = a.p.get()]() { ap->g.array() += o->g(0); };
    return out;
}
inline Tensor mean(const Tensor& a) {
    Tensor out = detail::result({1}, {a.p});
    const double inv = 1.0 / a.size();
    out.p->v(0) = a.p->v.mean();
    TensorImpl* o = out.p.get();
    out.p->backward_fn = [o, ap = a.p.get(), inv]() { ap->g.array() += o->g(0) * inv; };
    return out;
}

// ---- losses ----
// mean-squared error between pred and (constant) target, returns a scalar.
inline Tensor mseLoss(const Tensor& pred, const Tensor& target) {
    Tensor diff = pred - target;
    return mean(diff * diff);
}

// ---- more activations (modern; each gradient-checkable) ----
inline Tensor leakyRelu(const Tensor& a, double alpha = 0.01) {
    Tensor out = detail::result(a.shape(), {a.p});
    for (int i = 0; i < a.size(); i++) { double x = a.p->v(i); out.p->v(i) = x > 0 ? x : alpha * x; }
    TensorImpl* o = out.p.get();
    out.p->backward_fn = [o, ap = a.p.get(), alpha]() {
        for (int i = 0; i < ap->size(); i++) ap->g(i) += o->g(i) * (ap->v(i) > 0 ? 1.0 : alpha);
    };
    return out;
}
// exact GELU: x·Φ(x);  d/dx = Φ(x) + x·φ(x).
inline Tensor gelu(const Tensor& a) {
    const double inv_sqrt2 = 0.70710678118654752440, inv_sqrt2pi = 0.39894228040143267794;
    Tensor out = detail::result(a.shape(), {a.p});
    for (int i = 0; i < a.size(); i++) { double x = a.p->v(i); out.p->v(i) = 0.5 * x * (1.0 + std::erf(x * inv_sqrt2)); }
    TensorImpl* o = out.p.get();
    out.p->backward_fn = [o, ap = a.p.get(), inv_sqrt2, inv_sqrt2pi]() {
        for (int i = 0; i < ap->size(); i++) {
            double x = ap->v(i), cdf = 0.5 * (1.0 + std::erf(x * inv_sqrt2)), pdf = inv_sqrt2pi * std::exp(-0.5 * x * x);
            ap->g(i) += o->g(i) * (cdf + x * pdf);
        }
    };
    return out;
}
// SiLU / Swish: x·σ(x);  d/dx = σ(x)(1 + x(1−σ(x))).
inline Tensor silu(const Tensor& a) {
    Tensor out = detail::result(a.shape(), {a.p});
    for (int i = 0; i < a.size(); i++) { double x = a.p->v(i), s = 1.0 / (1.0 + std::exp(-x)); out.p->v(i) = x * s; }
    TensorImpl* o = out.p.get();
    out.p->backward_fn = [o, ap = a.p.get()]() {
        for (int i = 0; i < ap->size(); i++) { double x = ap->v(i), s = 1.0 / (1.0 + std::exp(-x)); ap->g(i) += o->g(i) * (s + x * s * (1.0 - s)); }
    };
    return out;
}
// elementwise exp:  d/dx e^x = e^x = out.
inline Tensor expT(const Tensor& a) {
    Tensor out = detail::result(a.shape(), {a.p});
    out.p->v = a.p->v.array().exp();
    TensorImpl* o = out.p.get();
    out.p->backward_fn = [o, ap = a.p.get()]() { for (int i = 0; i < ap->size(); i++) ap->g(i) += o->g(i) * o->v(i); };
    return out;
}

// row-wise softmax of a (N,C) tensor (numerically stable). Jacobian-vector backward.
inline Tensor softmaxRow(const Tensor& a) {
    const int N = a.shape()[0], C = a.shape()[1];
    Tensor out = detail::result({N, C}, {a.p});
    Eigen::Map<RowMat> X(a.p->v.data(), N, C), S(out.p->v.data(), N, C);
    for (int i = 0; i < N; i++) { double m = X.row(i).maxCoeff();
        Eigen::ArrayXd e = (X.row(i).array() - m).exp(); S.row(i) = (e / e.sum()).matrix(); }
    TensorImpl* o = out.p.get();
    out.p->backward_fn = [o, ap = a.p.get(), N, C]() {
        Eigen::Map<RowMat> S(o->v.data(), N, C), dS(o->g.data(), N, C), dX(ap->g.data(), N, C);
        for (int i = 0; i < N; i++) { double dot = (dS.row(i).array() * S.row(i).array()).sum();
            dX.row(i).array() += S.row(i).array() * (dS.row(i).array() - dot); }
    };
    return out;
}

// ---- classification losses (from LOGITS; numerically stable; fused backward) ----
// softmax cross-entropy: targets = integer class index per row.  dZ = (softmax − onehot)/N.
inline Tensor crossEntropyLoss(const Tensor& logits, const std::vector<int>& targets) {
    const int N = logits.shape()[0], C = logits.shape()[1];
    Tensor out = detail::result({1}, {logits.p});
    Eigen::Map<RowMat> Z(logits.p->v.data(), N, C);
    auto soft = std::make_shared<Eigen::MatrixXd>(N, C);
    double loss = 0;
    for (int i = 0; i < N; i++) {
        double m = Z.row(i).maxCoeff();
        Eigen::ArrayXd e = (Z.row(i).array() - m).exp(); double s = e.sum();
        soft->row(i) = (e / s).matrix();
        loss += -std::log(std::max((*soft)(i, targets[i]), 1e-300));
    }
    out.p->v(0) = loss / N;
    TensorImpl* o = out.p.get();
    out.p->backward_fn = [o, lp = logits.p.get(), soft, targets, N, C]() {
        Eigen::Map<RowMat> dZ(lp->g.data(), N, C);
        double g = o->g(0) / N;
        for (int i = 0; i < N; i++)
            for (int c = 0; c < C; c++) dZ(i, c) += g * ((*soft)(i, c) - (c == targets[i] ? 1.0 : 0.0));
    };
    return out;
}
// binary cross-entropy with logits (stable):  L = max(z,0) − z·y + log(1+e^−|z|);  dz = (σ(z) − y)/n.
inline Tensor bceWithLogits(const Tensor& logits, const Tensor& targets) {
    const int n = logits.size();
    Tensor out = detail::result({1}, {logits.p});
    double loss = 0;
    for (int i = 0; i < n; i++) { double z = logits.p->v(i), y = targets.p->v(i);
        loss += std::max(z, 0.0) - z * y + std::log1p(std::exp(-std::fabs(z))); }
    out.p->v(0) = loss / n;
    TensorImpl* o = out.p.get();
    out.p->backward_fn = [o, lp = logits.p.get(), tp = targets.p.get(), n]() {
        double g = o->g(0) / n;
        for (int i = 0; i < n; i++) { double s = 1.0 / (1.0 + std::exp(-lp->v(i))); lp->g(i) += g * (s - tp->v(i)); }
    };
    return out;
}

// elementwise natural log:  y = ln(x);  dy/dx = 1/x  (clamped for safety).
inline Tensor logT(const Tensor& a) {
    Tensor out = detail::result(a.shape(), {a.p});
    out.p->v = a.p->v.array().max(1e-300).log();
    TensorImpl* o = out.p.get();
    out.p->backward_fn = [o, ap = a.p.get()]() {
        for (int i = 0; i < ap->size(); i++) ap->g(i) += o->g(i) / std::max(ap->v(i), 1e-300);
    };
    return out;
}

// row-wise log-softmax of a (N,C) tensor (stable).  softmax = exp(out);
// backward:  dz = g − softmax(z)·rowsum(g).  (Used by REINFORCE/PPO log-probs.)
inline Tensor logSoftmaxRow(const Tensor& a) {
    const int N = a.shape()[0], C = a.shape()[1];
    Tensor out = detail::result({N, C}, {a.p});
    Eigen::Map<RowMat> X(a.p->v.data(), N, C), Y(out.p->v.data(), N, C);
    for (int i = 0; i < N; i++) {
        double m = X.row(i).maxCoeff();
        Eigen::ArrayXd e = (X.row(i).array() - m).exp();
        double lse = m + std::log(e.sum());
        Y.row(i) = (X.row(i).array() - lse).matrix();
    }
    TensorImpl* o = out.p.get();
    out.p->backward_fn = [o, ap = a.p.get(), N, C]() {
        Eigen::Map<RowMat> Y(o->v.data(), N, C), dY(o->g.data(), N, C), dX(ap->g.data(), N, C);
        for (int i = 0; i < N; i++) {
            double gs = dY.row(i).sum();
            dX.row(i).array() += dY.row(i).array() - Y.row(i).array().exp() * gs;
        }
    };
    return out;
}

// gather one column per row: out(i,0) = X(i, idx[i]).  Returns (N,1).
// Built from constant one-hot ⊙ X then row-sum via matmul-ones, so the gradient
// flows back to ONLY the selected entries (the Q/log-prob the agent acted on).
inline Tensor gatherColumns(const Tensor& X, const std::vector<int>& idx) {
    const int N = X.shape()[0], A = X.shape()[1];
    Tensor onehot = Tensor::zeros({N, A}, false);
    for (int i = 0; i < N; i++) onehot.p->v(i * A + idx[i]) = 1.0;
    Tensor masked = X * onehot;
    Tensor ones = Tensor::zeros({A, 1}, false); ones.p->v.setOnes();
    return matmul(masked, ones);
}

// elementwise clamp to [lo,hi]; gradient passes through only where unclamped (PPO ratio clip).
inline Tensor clampT(const Tensor& a, double lo, double hi) {
    Tensor out = detail::result(a.shape(), {a.p});
    for (int i = 0; i < a.size(); i++) out.p->v(i) = std::min(hi, std::max(lo, a.p->v(i)));
    TensorImpl* o = out.p.get();
    out.p->backward_fn = [o, ap = a.p.get(), lo, hi]() {
        for (int i = 0; i < ap->size(); i++) { double v = ap->v(i); if (v > lo && v < hi) ap->g(i) += o->g(i); }
    };
    return out;
}

// elementwise minimum of two same-shape tensors; gradient routes to the smaller side
// (the PPO pessimistic surrogate min(ratio·A, clip(ratio)·A)).
inline Tensor minT(const Tensor& a, const Tensor& b) {
    Tensor out = detail::result(a.shape(), {a.p, b.p});
    for (int i = 0; i < a.size(); i++) out.p->v(i) = std::min(a.p->v(i), b.p->v(i));
    TensorImpl* o = out.p.get();
    out.p->backward_fn = [o, ap = a.p.get(), bp = b.p.get()]() {
        for (int i = 0; i < ap->size(); i++) (ap->v(i) <= bp->v(i) ? ap->g(i) : bp->g(i)) += o->g(i);
    };
    return out;
}

// ---- shape ops (for attention / transformers) ----
inline Tensor transpose2d(const Tensor& a) {
    const int N = a.shape()[0], M = a.shape()[1];
    Tensor out = detail::result({M, N}, {a.p});
    out.mat() = a.mat().transpose();
    TensorImpl* o = out.p.get();
    out.p->backward_fn = [o, ap = a.p.get(), N, M]() {
        Eigen::Map<RowMat> dO(o->g.data(), M, N), dA(ap->g.data(), N, M);
        dA.noalias() += dO.transpose();
    };
    return out;
}
// slice a contiguous column range [c0,c1) of a (N,M) tensor -> (N, c1-c0). backward scatters.
inline Tensor sliceCols(const Tensor& a, int c0, int c1) {
    const int N = a.shape()[0], M = a.shape()[1], W = c1 - c0;
    Tensor out = detail::result({N, W}, {a.p});
    Eigen::Map<RowMat> A(a.p->v.data(), N, M), Y(out.p->v.data(), N, W);
    Y = A.block(0, c0, N, W);
    TensorImpl* o = out.p.get();
    out.p->backward_fn = [o, ap = a.p.get(), N, M, c0, W]() {
        Eigen::Map<RowMat> dY(o->g.data(), N, W), dA(ap->g.data(), N, M);
        dA.block(0, c0, N, W) += dY;
    };
    return out;
}
// concatenate two (N,*) tensors along columns. backward splits.
inline Tensor concatCols(const Tensor& a, const Tensor& b) {
    const int N = a.shape()[0], Ma = a.shape()[1], Mb = b.shape()[1];
    Tensor out = detail::result({N, Ma + Mb}, {a.p, b.p});
    Eigen::Map<RowMat> A(a.p->v.data(), N, Ma), B(b.p->v.data(), N, Mb), Y(out.p->v.data(), N, Ma + Mb);
    Y.block(0, 0, N, Ma) = A; Y.block(0, Ma, N, Mb) = B;
    TensorImpl* o = out.p.get();
    out.p->backward_fn = [o, ap = a.p.get(), bp = b.p.get(), N, Ma, Mb]() {
        Eigen::Map<RowMat> dY(o->g.data(), N, Ma + Mb), dA(ap->g.data(), N, Ma), dB(bp->g.data(), N, Mb);
        dA += dY.block(0, 0, N, Ma); dB += dY.block(0, Ma, N, Mb);
    };
    return out;
}
// slice a contiguous row range [r0,r1) of a (N,M) tensor -> (r1-r0, M). backward scatters.
inline Tensor sliceRows(const Tensor& a, int r0, int r1) {
    const int N = a.shape()[0], M = a.shape()[1], R = r1 - r0;
    Tensor out = detail::result({R, M}, {a.p});
    Eigen::Map<RowMat> A(a.p->v.data(), N, M), Y(out.p->v.data(), R, M);
    Y = A.block(r0, 0, R, M);
    TensorImpl* o = out.p.get();
    out.p->backward_fn = [o, ap = a.p.get(), N, M, r0, R]() {
        Eigen::Map<RowMat> dY(o->g.data(), R, M), dA(ap->g.data(), N, M);
        dA.block(r0, 0, R, M) += dY;
    };
    return out;
}
// concatenate two (*,M) tensors along rows. backward splits.
inline Tensor concatRows(const Tensor& a, const Tensor& b) {
    const int Na = a.shape()[0], Nb = b.shape()[0], M = a.shape()[1];
    Tensor out = detail::result({Na + Nb, M}, {a.p, b.p});
    Eigen::Map<RowMat> A(a.p->v.data(), Na, M), B(b.p->v.data(), Nb, M), Y(out.p->v.data(), Na + Nb, M);
    Y.block(0, 0, Na, M) = A; Y.block(Na, 0, Nb, M) = B;
    TensorImpl* o = out.p.get();
    out.p->backward_fn = [o, ap = a.p.get(), bp = b.p.get(), Na, Nb, M]() {
        Eigen::Map<RowMat> dY(o->g.data(), Na + Nb, M), dA(ap->g.data(), Na, M), dB(bp->g.data(), Nb, M);
        dA += dY.block(0, 0, Na, M); dB += dY.block(Na, 0, Nb, M);
    };
    return out;
}

// ---- LayerNorm over the last dim:  y = γ ⊙ (x−μ)/√(σ²+ε) + β  (per row). ----
inline Tensor layerNorm(const Tensor& x, const Tensor& gamma, const Tensor& beta, double eps = 1e-5) {
    const int N = x.shape()[0], D = x.shape()[1];
    Tensor out = detail::result({N, D}, {x.p, gamma.p, beta.p});
    Eigen::Map<RowMat> X(x.p->v.data(), N, D), Y(out.p->v.data(), N, D);
    auto xhat = std::make_shared<Eigen::MatrixXd>(N, D);
    auto invs = std::make_shared<Eigen::VectorXd>(N);
    for (int i = 0; i < N; i++) {
        double mu = X.row(i).mean();
        Eigen::RowVectorXd c = X.row(i).array() - mu;
        double var = c.array().square().sum() / D, is = 1.0 / std::sqrt(var + eps);
        (*invs)(i) = is; xhat->row(i) = c * is;
        for (int j = 0; j < D; j++) Y(i, j) = (*xhat)(i, j) * gamma.p->v(j) + beta.p->v(j);
    }
    TensorImpl* o = out.p.get();
    out.p->backward_fn = [o, xp = x.p.get(), gp = gamma.p.get(), bp = beta.p.get(), xhat, invs, N, D]() {
        Eigen::Map<RowMat> dY(o->g.data(), N, D), dX(xp->g.data(), N, D);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < D; j++) { gp->g(j) += dY(i, j) * (*xhat)(i, j); bp->g(j) += dY(i, j); }
        for (int i = 0; i < N; i++) {
            double mean_dxhat = 0, mean_dxhat_xhat = 0;
            for (int j = 0; j < D; j++) { double dxh = dY(i, j) * gp->v(j); mean_dxhat += dxh; mean_dxhat_xhat += dxh * (*xhat)(i, j); }
            mean_dxhat /= D; mean_dxhat_xhat /= D; double is = (*invs)(i);
            for (int j = 0; j < D; j++) { double dxh = dY(i, j) * gp->v(j);
                dX(i, j) += is * (dxh - mean_dxhat - (*xhat)(i, j) * mean_dxhat_xhat); }
        }
    };
    return out;
}

// ---- Embedding lookup: rows of W (vocab,D) at integer indices -> (L,D). backward scatter-adds. ----
inline Tensor embeddingLookup(const Tensor& W, const std::vector<int>& idx) {
    const int V = W.shape()[0], D = W.shape()[1], L = (int)idx.size();
    Tensor out = detail::result({L, D}, {W.p});
    Eigen::Map<RowMat> Wm(W.p->v.data(), V, D), Y(out.p->v.data(), L, D);
    for (int i = 0; i < L; i++) Y.row(i) = Wm.row(idx[i]);
    TensorImpl* o = out.p.get();
    out.p->backward_fn = [o, wp = W.p.get(), idx, V, D, L]() {
        Eigen::Map<RowMat> dW(wp->g.data(), V, D), dY(o->g.data(), L, D);
        for (int i = 0; i < L; i++) dW.row(idx[i]) += dY.row(i);
    };
    return out;
}

// ---- reshape (same element count; identity on the flat buffer) ----
inline Tensor reshape(const Tensor& a, const std::vector<int>& newshape) {
    Tensor out = detail::result(newshape, {a.p});
    out.p->v = a.p->v;
    TensorImpl* o = out.p.get();
    out.p->backward_fn = [o, ap = a.p.get()]() { ap->g += o->g; };
    return out;
}

// ---- Conv2D (single image): x (Cin,H,W), W (Cout·Cin·kh·kw flat), b (Cout) -> (Cout,Hout,Wout).
// Explicit cross-correlation; backward accumulates dX/dW/db by the same index walk (gradient-checked).
inline Tensor conv2d(const Tensor& x, const Tensor& W, const Tensor& b,
                     int Cin, int H, int Wd, int Cout, int kh, int kw, int stride = 1, int pad = 0) {
    const int Hout = (H + 2*pad - kh) / stride + 1, Wout = (Wd + 2*pad - kw) / stride + 1;
    Tensor out = detail::result({Cout, Hout, Wout}, {x.p, W.p, b.p});
    auto XI = [=](int c,int i,int j){ return c*H*Wd + i*Wd + j; };
    auto WI = [=](int co,int ci,int a,int c){ return co*Cin*kh*kw + ci*kh*kw + a*kw + c; };
    auto OI = [=](int c,int i,int j){ return c*Hout*Wout + i*Wout + j; };
    const double* xv = x.p->v.data(); const double* wv = W.p->v.data(); const double* bv = b.p->v.data();
    double* ov = out.p->v.data();
    for (int co = 0; co < Cout; co++)
        for (int i = 0; i < Hout; i++) for (int j = 0; j < Wout; j++) {
            double acc = bv[co];
            for (int ci = 0; ci < Cin; ci++) for (int a = 0; a < kh; a++) for (int c = 0; c < kw; c++) {
                int ii = i*stride - pad + a, jj = j*stride - pad + c;
                if (ii >= 0 && ii < H && jj >= 0 && jj < Wd) acc += xv[XI(ci,ii,jj)] * wv[WI(co,ci,a,c)];
            }
            ov[OI(co,i,j)] = acc;
        }
    TensorImpl* o = out.p.get();
    out.p->backward_fn = [o, xp=x.p.get(), wp=W.p.get(), bp=b.p.get(), Cin,H,Wd,Cout,kh,kw,stride,pad,Hout,Wout]() {
        auto XI = [=](int c,int i,int j){ return c*H*Wd + i*Wd + j; };
        auto WI = [=](int co,int ci,int a,int c){ return co*Cin*kh*kw + ci*kh*kw + a*kw + c; };
        auto OI = [=](int c,int i,int j){ return c*Hout*Wout + i*Wout + j; };
        const double* xv = xp->v.data(); const double* wv = wp->v.data();
        double* dx = xp->g.data(); double* dw = wp->g.data(); double* db = bp->g.data(); const double* dov = o->g.data();
        for (int co = 0; co < Cout; co++)
            for (int i = 0; i < Hout; i++) for (int j = 0; j < Wout; j++) {
                double go = dov[OI(co,i,j)]; db[co] += go;
                for (int ci = 0; ci < Cin; ci++) for (int a = 0; a < kh; a++) for (int c = 0; c < kw; c++) {
                    int ii = i*stride - pad + a, jj = j*stride - pad + c;
                    if (ii >= 0 && ii < H && jj >= 0 && jj < Wd) {
                        dw[WI(co,ci,a,c)] += go * xv[XI(ci,ii,jj)];
                        dx[XI(ci,ii,jj)]  += go * wv[WI(co,ci,a,c)];
                    }
                }
            }
    };
    return out;
}

// ---- 2-D max / average pooling on (C,H,W) -> (C,Hout,Wout) ----
inline Tensor maxPool2d(const Tensor& x, int C, int H, int Wd, int k, int stride) {
    const int Hout = (H - k)/stride + 1, Wout = (Wd - k)/stride + 1;
    Tensor out = detail::result({C, Hout, Wout}, {x.p});
    auto am = std::make_shared<std::vector<int>>(C*Hout*Wout);
    const double* xv = x.p->v.data(); double* ov = out.p->v.data();
    auto XI = [=](int c,int i,int j){ return c*H*Wd + i*Wd + j; };
    auto OI = [=](int c,int i,int j){ return c*Hout*Wout + i*Wout + j; };
    for (int c = 0; c < C; c++) for (int i = 0; i < Hout; i++) for (int j = 0; j < Wout; j++) {
        double best = -1e300; int bidx = 0;
        for (int a = 0; a < k; a++) for (int d = 0; d < k; d++) {
            int idx = XI(c, i*stride+a, j*stride+d); if (xv[idx] > best) { best = xv[idx]; bidx = idx; }
        }
        ov[OI(c,i,j)] = best; (*am)[OI(c,i,j)] = bidx;
    }
    TensorImpl* o = out.p.get();
    out.p->backward_fn = [o, xp=x.p.get(), am]() {
        double* dx = xp->g.data(); const double* dov = o->g.data();
        for (size_t i = 0; i < am->size(); i++) dx[(*am)[i]] += dov[i];
    };
    return out;
}
inline Tensor avgPool2d(const Tensor& x, int C, int H, int Wd, int k, int stride) {
    const int Hout = (H - k)/stride + 1, Wout = (Wd - k)/stride + 1; const double inv = 1.0/(k*k);
    Tensor out = detail::result({C, Hout, Wout}, {x.p});
    const double* xv = x.p->v.data(); double* ov = out.p->v.data();
    auto XI = [=](int c,int i,int j){ return c*H*Wd + i*Wd + j; };
    auto OI = [=](int c,int i,int j){ return c*Hout*Wout + i*Wout + j; };
    for (int c = 0; c < C; c++) for (int i = 0; i < Hout; i++) for (int j = 0; j < Wout; j++) {
        double acc = 0; for (int a = 0; a < k; a++) for (int d = 0; d < k; d++) acc += xv[XI(c, i*stride+a, j*stride+d)];
        ov[OI(c,i,j)] = acc * inv;
    }
    TensorImpl* o = out.p.get();
    out.p->backward_fn = [o, xp=x.p.get(), C,H,Wd,k,stride,Hout,Wout,inv]() {
        auto XI = [=](int c,int i,int j){ return c*H*Wd + i*Wd + j; };
        auto OI = [=](int c,int i,int j){ return c*Hout*Wout + i*Wout + j; };
        double* dx = xp->g.data(); const double* dov = o->g.data();
        for (int c = 0; c < C; c++) for (int i = 0; i < Hout; i++) for (int j = 0; j < Wout; j++) {
            double go = dov[OI(c,i,j)] * inv;
            for (int a = 0; a < k; a++) for (int d = 0; d < k; d++) dx[XI(c, i*stride+a, j*stride+d)] += go;
        }
    };
    return out;
}

// ===================================================================================
// [2] Module / Layer API (PyTorch-shaped): layers hold Parameter tensors and COMPOSE
// the ops above — no hand-derived backward. parameters() feeds the optimizer.
// ===================================================================================
struct Module {
    virtual ~Module() = default;
    virtual Tensor forward(const Tensor& x) = 0;
    virtual std::vector<Tensor> parameters() { return {}; }
    Tensor operator()(const Tensor& x) { return forward(x); }
};
using ModulePtr = std::shared_ptr<Module>;

// Fully-connected layer: y = x·W + b.  He-normal init by default (good for ReLU nets).
class Linear : public Module {
public:
    Tensor W, b;
    Linear(int in, int out, uint64_t seed = 1) {
        double s = std::sqrt(2.0 / in);                         // He/Kaiming
        W = Tensor::randn({in, out}, s, seed * 2654435761ull + 12345ull, true);
        b = Tensor::zeros({out}, true);
    }
    Tensor forward(const Tensor& x) override { return addBias(matmul(x, W), b); }
    std::vector<Tensor> parameters() override { return {W, b}; }
};

// activation modules (thin wrappers over the functional ops)
struct ReLU    : Module { Tensor forward(const Tensor& x) override { return relu(x); } };
struct Tanh    : Module { Tensor forward(const Tensor& x) override { return tanhT(x); } };
struct Sigmoid : Module { Tensor forward(const Tensor& x) override { return sigmoidT(x); } };

// ordered container; forward chains the children, parameters() concatenates theirs.
class Sequential : public Module {
public:
    std::vector<ModulePtr> mods;
    Sequential() = default;
    explicit Sequential(std::vector<ModulePtr> m) : mods(std::move(m)) {}
    void add(ModulePtr m) { mods.push_back(std::move(m)); }
    Tensor forward(const Tensor& x) override {
        Tensor h = x; for (auto& m : mods) h = m->forward(h); return h;
    }
    std::vector<Tensor> parameters() override {
        std::vector<Tensor> ps;
        for (auto& m : mods) { auto q = m->parameters(); ps.insert(ps.end(), q.begin(), q.end()); }
        return ps;
    }
};

// LayerNorm module (γ init 1, β init 0).
class LayerNorm : public Module {
public:
    Tensor gamma, beta; double eps;
    explicit LayerNorm(int d, double eps_ = 1e-5) : eps(eps_) {
        gamma = Tensor::zeros({d}, true); gamma.value().setOnes();
        beta  = Tensor::zeros({d}, true);
    }
    Tensor forward(const Tensor& x) override { return layerNorm(x, gamma, beta, eps); }
    std::vector<Tensor> parameters() override { return {gamma, beta}; }
};

// Embedding table (vocab × dim). Use .lookup(indices) — it maps token ids, not a Tensor.
class Embedding : public Module {
public:
    Tensor W;
    Embedding(int vocab, int dim, uint64_t seed = 1) { W = Tensor::randn({vocab, dim}, 0.02, seed, true); }
    Tensor forward(const Tensor& x) override { return x; }            // not used directly
    Tensor lookup(const std::vector<int>& idx) { return embeddingLookup(W, idx); }
    std::vector<Tensor> parameters() override { return {W}; }
};

// ===================================================================================
// [3] Transformer — Multi-Head self-Attention + a pre-norm Transformer block. The whole
// backward is AUTOMATIC (composed entirely of the gradient-checked ops above): per head
// scores = Qh·Khᵀ/√d → (+causal mask) → softmax → ·Vh, heads concatenated then projected.
// ===================================================================================
// constant additive causal mask: 0 on/below the diagonal, −1e9 above (no gradient).
inline Tensor causalMask(int S) {
    Tensor m = Tensor::zeros({S, S}, false);
    Eigen::Map<RowMat> M(m.value().data(), S, S);
    for (int i = 0; i < S; i++) for (int j = i + 1; j < S; j++) M(i, j) = -1e9;
    return m;
}

class MultiHeadAttention : public Module {
public:
    int d_model, n_heads, d_head; bool causal;
    Tensor Wq, bq, Wk, bk, Wv, bv, Wo, bo;
    MultiHeadAttention(int d_model_, int n_heads_, bool causal_ = false, uint64_t seed = 1)
        : d_model(d_model_), n_heads(n_heads_), d_head(d_model_ / n_heads_), causal(causal_) {
        double s = std::sqrt(1.0 / d_model);
        Wq = Tensor::randn({d_model, d_model}, s, seed*7+1, true); bq = Tensor::zeros({d_model}, true);
        Wk = Tensor::randn({d_model, d_model}, s, seed*7+2, true); bk = Tensor::zeros({d_model}, true);
        Wv = Tensor::randn({d_model, d_model}, s, seed*7+3, true); bv = Tensor::zeros({d_model}, true);
        Wo = Tensor::randn({d_model, d_model}, s, seed*7+4, true); bo = Tensor::zeros({d_model}, true);
    }
    Tensor forward(const Tensor& x) override {           // x: (seq, d_model)
        const int S = x.shape()[0];
        Tensor Q = addBias(matmul(x, Wq), bq), K = addBias(matmul(x, Wk), bk), V = addBias(matmul(x, Wv), bv);
        const double sc = 1.0 / std::sqrt((double)d_head);
        Tensor mask; if (causal) mask = causalMask(S);
        Tensor cat;
        for (int h = 0; h < n_heads; h++) {
            Tensor Qh = sliceCols(Q, h*d_head, (h+1)*d_head);
            Tensor Kh = sliceCols(K, h*d_head, (h+1)*d_head);
            Tensor Vh = sliceCols(V, h*d_head, (h+1)*d_head);
            Tensor scores = matmul(Qh, transpose2d(Kh)) * sc;          // (S,S)
            if (causal) scores = scores + mask;
            Tensor Oh = matmul(softmaxRow(scores), Vh);                // (S,d_head)
            cat = (h == 0) ? Oh : concatCols(cat, Oh);
        }
        return addBias(matmul(cat, Wo), bo);
    }
    std::vector<Tensor> parameters() override { return {Wq,bq,Wk,bk,Wv,bv,Wo,bo}; }
};

// Pre-norm Transformer block: x += attn(LN(x));  x += FFN(LN(x)), FFN = Linear→GELU→Linear.
class TransformerBlock : public Module {
public:
    MultiHeadAttention attn; LayerNorm ln1, ln2; Linear ff1, ff2;
    TransformerBlock(int d_model, int n_heads, int d_ff, bool causal = false, uint64_t seed = 1)
        : attn(d_model, n_heads, causal, seed), ln1(d_model), ln2(d_model),
          ff1(d_model, d_ff, seed*11+5), ff2(d_ff, d_model, seed*11+6) {}
    Tensor forward(const Tensor& x) override {
        Tensor x1 = x + attn.forward(ln1.forward(x));
        return x1 + ff2.forward(gelu(ff1.forward(ln2.forward(x1))));
    }
    std::vector<Tensor> parameters() override {
        std::vector<Tensor> ps;
        for (Module* m : {(Module*)&attn, (Module*)&ln1, (Module*)&ln2, (Module*)&ff1, (Module*)&ff2}) {
            auto q = m->parameters(); ps.insert(ps.end(), q.begin(), q.end());
        }
        return ps;
    }
};

// Self-hosted GPT-style causal language model: token embedding + learned positional
// embedding -> N pre-norm causal Transformer blocks -> final LayerNorm -> LM head (vocab
// logits).  forwardIds(ids) returns (T, vocab) next-token logits.  The ENTIRE backward is
// the same gradient-checked autograd engine — zero Python, zero external deps: the literal
// "self-hosted AI" pitch.  Pair with gptGenerate() for autoregressive sampling.
class GPT : public Module {
public:
    int vocab, d_model, n_layers, n_heads, d_ff, max_seq;
    Embedding tok;
    Tensor pos;                                              // learned positional embedding (max_seq, d_model)
    std::vector<std::shared_ptr<TransformerBlock>> blocks;
    LayerNorm lnf;
    Linear head;
    GPT(int vocab_, int d_model_, int n_layers_, int n_heads_, int d_ff_, int max_seq_, uint64_t seed = 1)
        : vocab(vocab_), d_model(d_model_), n_layers(n_layers_), n_heads(n_heads_), d_ff(d_ff_),
          max_seq(max_seq_), tok(vocab_, d_model_, seed * 13 + 1),
          lnf(d_model_), head(d_model_, vocab_, seed * 13 + 2) {
        pos = Tensor::randn({ max_seq_, d_model_ }, 0.02, seed * 13 + 3, true);
        for (int i = 0; i < n_layers_; i++)
            blocks.push_back(std::make_shared<TransformerBlock>(d_model_, n_heads_, d_ff_, true, seed * 13 + 10 + i));
    }
    Tensor forwardIds(const std::vector<int>& ids) {
        int T = (int)ids.size();
        Tensor h = tok.lookup(ids) + sliceRows(pos, 0, T);   // token + positional
        for (auto& b : blocks) h = b->forward(h);
        return head.forward(lnf.forward(h));                 // (T, vocab)
    }
    Tensor forward(const Tensor& x) override { return x; }   // use forwardIds()
    std::vector<Tensor> parameters() override {
        std::vector<Tensor> ps = tok.parameters(); ps.push_back(pos);
        for (auto& b : blocks) { auto q = b->parameters(); ps.insert(ps.end(), q.begin(), q.end()); }
        auto a = lnf.parameters();  ps.insert(ps.end(), a.begin(), a.end());
        auto c = head.parameters(); ps.insert(ps.end(), c.begin(), c.end());
        return ps;
    }
};

// Autoregressive generation: extend ctx by n_new tokens.  greedy=argmax, else temperature
// sampling.  Context is truncated to the model's max_seq window (sliding attention).
inline std::vector<int> gptGenerate(GPT& model, std::vector<int> ctx, int n_new,
                                    double temperature, std::mt19937_64& rng, bool greedy = false) {
    std::vector<int> out; out.reserve(n_new);
    for (int i = 0; i < n_new; i++) {
        std::vector<int> w = ((int)ctx.size() > model.max_seq)
                             ? std::vector<int>(ctx.end() - model.max_seq, ctx.end()) : ctx;
        Tensor logits = model.forwardIds(w);
        int T = (int)w.size();
        Eigen::Map<RowMat> L(logits.value().data(), T, model.vocab);
        Eigen::VectorXd last = L.row(T - 1).transpose();
        int next;
        if (greedy) { last.maxCoeff(&next); }
        else {
            Eigen::ArrayXd z = last.array() / temperature;
            Eigen::ArrayXd p = (z - z.maxCoeff()).exp(); p /= p.sum();
            double u = std::uniform_real_distribution<double>(0, 1)(rng), c = 0; next = model.vocab - 1;
            for (int k = 0; k < model.vocab; k++) { c += p(k); if (u <= c) { next = k; break; } }
        }
        ctx.push_back(next); out.push_back(next);
    }
    return out;
}

// 2-D convolution layer (single image (Cin,H,W)). He init scaled by fan-in Cin·k·k.
class Conv2d : public Module {
public:
    int Cin, Cout, kh, kw, stride, pad; Tensor W, b;
    Conv2d(int Cin_, int Cout_, int k, int stride_ = 1, int pad_ = 0, uint64_t seed = 1)
        : Cin(Cin_), Cout(Cout_), kh(k), kw(k), stride(stride_), pad(pad_) {
        double s = std::sqrt(2.0 / (Cin * k * k));
        W = Tensor::randn({Cout * Cin * k * k}, s, seed, true);
        b = Tensor::zeros({Cout}, true);
    }
    Tensor forward(const Tensor& x) override {           // x: (Cin,H,W)
        int H = x.shape()[1], Wd = x.shape()[2];
        return conv2d(x, W, b, Cin, H, Wd, Cout, kh, kw, stride, pad);
    }
    std::vector<Tensor> parameters() override { return {W, b}; }
};
// pooling modules (read C,H,W from the input shape)
class MaxPool2d : public Module {
public:
    int k, stride;
    explicit MaxPool2d(int k_, int stride_ = -1) : k(k_), stride(stride_ < 0 ? k_ : stride_) {}
    Tensor forward(const Tensor& x) override { return maxPool2d(x, x.shape()[0], x.shape()[1], x.shape()[2], k, stride); }
};
class AvgPool2d : public Module {
public:
    int k, stride;
    explicit AvgPool2d(int k_, int stride_ = -1) : k(k_), stride(stride_ < 0 ? k_ : stride_) {}
    Tensor forward(const Tensor& x) override { return avgPool2d(x, x.shape()[0], x.shape()[1], x.shape()[2], k, stride); }
};

// ===================================================================================
// [4] Recurrent layers. forward(x) takes a sequence x:(T, in_dim) and returns ALL hidden
// states (T, hid). Back-prop-through-time is AUTOMATIC — the cell is just verified ops
// unrolled over the time loop. Use sliceRows(out, T-1, T) for the last state (classification).
// ===================================================================================
class RNN : public Module {
public:
    int in_dim, hid; Tensor Wx, Wh, b;
    RNN(int in_, int hid_, uint64_t seed = 1) : in_dim(in_), hid(hid_) {
        double s = std::sqrt(1.0 / hid);
        Wx = Tensor::randn({in_dim, hid}, s, seed*5+1, true);
        Wh = Tensor::randn({hid, hid}, s, seed*5+2, true);
        b  = Tensor::zeros({hid}, true);
    }
    Tensor forward(const Tensor& x) override {
        int T = x.shape()[0];
        Tensor h = Tensor::zeros({1, hid}, false), outs;
        for (int t = 0; t < T; t++) {
            Tensor xt = sliceRows(x, t, t + 1);
            h = tanhT(addBias(matmul(xt, Wx) + matmul(h, Wh), b));
            outs = (t == 0) ? h : concatRows(outs, h);
        }
        return outs;
    }
    std::vector<Tensor> parameters() override { return {Wx, Wh, b}; }
};

// LSTM with the 4 gates packed (order i,f,g,o).  c = f⊙c + i⊙g ;  h = o⊙tanh(c).
class LSTM : public Module {
public:
    int in_dim, hid; Tensor Wx, Wh, b;
    LSTM(int in_, int hid_, uint64_t seed = 1) : in_dim(in_), hid(hid_) {
        double s = std::sqrt(1.0 / hid);
        Wx = Tensor::randn({in_dim, 4*hid}, s, seed*5+1, true);
        Wh = Tensor::randn({hid, 4*hid}, s, seed*5+2, true);
        b  = Tensor::zeros({4*hid}, true);
    }
    Tensor forward(const Tensor& x) override {
        int T = x.shape()[0];
        Tensor h = Tensor::zeros({1, hid}, false), c = Tensor::zeros({1, hid}, false), outs;
        for (int t = 0; t < T; t++) {
            Tensor xt = sliceRows(x, t, t + 1);
            Tensor z = addBias(matmul(xt, Wx) + matmul(h, Wh), b);     // (1,4hid)
            Tensor i  = sigmoidT(sliceCols(z, 0*hid, 1*hid));
            Tensor f  = sigmoidT(sliceCols(z, 1*hid, 2*hid));
            Tensor gg = tanhT   (sliceCols(z, 2*hid, 3*hid));
            Tensor o  = sigmoidT(sliceCols(z, 3*hid, 4*hid));
            c = f * c + i * gg;
            h = o * tanhT(c);
            outs = (t == 0) ? h : concatRows(outs, h);
        }
        return outs;
    }
    std::vector<Tensor> parameters() override { return {Wx, Wh, b}; }
};

// GRU.  z,r gates;  n = tanh(Wxn·x + r⊙(Whn·h) + bn);  h = (1−z)⊙n + z⊙h.
class GRU : public Module {
public:
    int in_dim, hid; Tensor Wxz, Whz, bz, Wxr, Whr, br, Wxn, Whn, bn;
    GRU(int in_, int hid_, uint64_t seed = 1) : in_dim(in_), hid(hid_) {
        double s = std::sqrt(1.0 / hid);
        Wxz = Tensor::randn({in_dim, hid}, s, seed*9+1, true); Whz = Tensor::randn({hid, hid}, s, seed*9+2, true); bz = Tensor::zeros({hid}, true);
        Wxr = Tensor::randn({in_dim, hid}, s, seed*9+3, true); Whr = Tensor::randn({hid, hid}, s, seed*9+4, true); br = Tensor::zeros({hid}, true);
        Wxn = Tensor::randn({in_dim, hid}, s, seed*9+5, true); Whn = Tensor::randn({hid, hid}, s, seed*9+6, true); bn = Tensor::zeros({hid}, true);
    }
    Tensor forward(const Tensor& x) override {
        int T = x.shape()[0];
        Tensor h = Tensor::zeros({1, hid}, false), outs;
        for (int t = 0; t < T; t++) {
            Tensor xt = sliceRows(x, t, t + 1);
            Tensor z = sigmoidT(addBias(matmul(xt, Wxz) + matmul(h, Whz), bz));
            Tensor r = sigmoidT(addBias(matmul(xt, Wxr) + matmul(h, Whr), br));
            Tensor n = tanhT(addBias(matmul(xt, Wxn) + r * matmul(h, Whn), bn));
            h = (n - z * n) + z * h;                                   // (1−z)⊙n + z⊙h
            outs = (t == 0) ? h : concatRows(outs, h);
        }
        return outs;
    }
    std::vector<Tensor> parameters() override { return {Wxz, Whz, bz, Wxr, Whr, br, Wxn, Whn, bn}; }
};

// ===================================================================================
// [2b] Optimizers — update parameters from their accumulated .grad().
// ===================================================================================
class SGD {
    std::vector<Tensor> params_; double lr_, momentum_;
    std::vector<Eigen::VectorXd> vel_;
public:
    SGD(std::vector<Tensor> params, double lr, double momentum = 0.0)
        : params_(std::move(params)), lr_(lr), momentum_(momentum) {
        for (auto& t : params_) vel_.push_back(Eigen::VectorXd::Zero(t.size()));
    }
    void zeroGrad() { for (auto& t : params_) t.zeroGrad(); }
    void setLR(double v) { lr_ = v; }
    void step() {
        for (size_t i = 0; i < params_.size(); i++) {
            if (momentum_ > 0.0) { vel_[i] = momentum_ * vel_[i] - lr_ * params_[i].grad();
                                   params_[i].value() += vel_[i]; }
            else params_[i].value() -= lr_ * params_[i].grad();
        }
    }
};

class Adam {
    std::vector<Tensor> params_; double lr_, b1_, b2_, eps_, wd_; long t_;
    std::vector<Eigen::VectorXd> m_, v_;
public:
    Adam(std::vector<Tensor> params, double lr = 1e-3, double b1 = 0.9, double b2 = 0.999,
         double eps = 1e-8, double weight_decay = 0.0)
        : params_(std::move(params)), lr_(lr), b1_(b1), b2_(b2), eps_(eps), wd_(weight_decay), t_(0) {
        for (auto& t : params_) { m_.push_back(Eigen::VectorXd::Zero(t.size()));
                                  v_.push_back(Eigen::VectorXd::Zero(t.size())); }
    }
    void zeroGrad() { for (auto& t : params_) t.zeroGrad(); }
    void setLR(double v) { lr_ = v; }
    void step() {
        t_++;
        const double bc1 = 1.0 - std::pow(b1_, (double)t_), bc2 = 1.0 - std::pow(b2_, (double)t_);
        for (size_t i = 0; i < params_.size(); i++) {
            Eigen::VectorXd gg = params_[i].grad();
            if (wd_ > 0.0) gg += wd_ * params_[i].value();             // AdamW-style decoupled decay
            m_[i] = b1_ * m_[i] + (1.0 - b1_) * gg;
            v_[i] = b2_ * v_[i] + (1.0 - b2_) * gg.cwiseProduct(gg);
            Eigen::ArrayXd mhat = (m_[i] / bc1).array();
            Eigen::ArrayXd vhat = (v_[i] / bc2).array();
            params_[i].value().array() -= lr_ * mhat / (vhat.sqrt() + eps_);
        }
    }
};

// ===================================================================================
// [2c] Training utilities — batching, metrics, early-stop, grad-clip, LR schedules, and
// checkpoint save/load (the self-hosted "train -> save -> load -> predict" loop).
// ===================================================================================
inline void clipGradNorm(std::vector<Tensor>& params, double max_norm) {
    double sq = 0; for (auto& p : params) sq += p.grad().squaredNorm();
    double norm = std::sqrt(sq);
    if (norm > max_norm && norm > 0) { double s = max_norm / norm; for (auto& p : params) p.grad() *= s; }
}

// shuffled mini-batch index generator (one epoch = a partition of [0,n) into batches).
class DataLoader {
    int n_, bs_; bool shuffle_; std::mt19937_64 rng_; std::vector<int> order_;
public:
    DataLoader(int n, int batch_size, bool shuffle = true, uint64_t seed = 0)
        : n_(n), bs_(batch_size), shuffle_(shuffle), rng_(seed) { order_.resize(n); for (int i = 0; i < n; i++) order_[i] = i; }
    std::vector<std::vector<int>> epoch() {
        if (shuffle_) std::shuffle(order_.begin(), order_.end(), rng_);
        std::vector<std::vector<int>> batches;
        for (int i = 0; i < n_; i += bs_) { int e = std::min(i + bs_, n_); batches.emplace_back(order_.begin() + i, order_.begin() + e); }
        return batches;
    }
};
// build a (B,D) batch tensor from selected rows of an (N,D) row-major matrix.
inline Tensor batchRows(const Eigen::MatrixXd& X, const std::vector<int>& idx) {
    int B = (int)idx.size(), D = (int)X.cols();
    Tensor t = Tensor::zeros({B, D}, false);
    Eigen::Map<RowMat> M(t.value().data(), B, D);
    for (int i = 0; i < B; i++) M.row(i) = X.row(idx[i]);
    return t;
}

// ---- metrics ----
inline double accuracy(const Tensor& logits, const std::vector<int>& targets) {
    int N = logits.shape()[0], C = logits.shape()[1], correct = 0; const Eigen::VectorXd& v = logits.p->v;
    for (int i = 0; i < N; i++) { int am = 0; double best = v(i*C);
        for (int c = 1; c < C; c++) if (v(i*C+c) > best) { best = v(i*C+c); am = c; }
        if (am == targets[i]) correct++; }
    return (double)correct / N;
}
struct PRF { double precision, recall, f1; };
inline PRF binaryPRF(const std::vector<int>& pred, const std::vector<int>& truth, int positive = 1) {
    long tp = 0, fp = 0, fn = 0;
    for (size_t i = 0; i < pred.size(); i++) { bool pp = pred[i]==positive, tt = truth[i]==positive;
        if (pp && tt) tp++; else if (pp && !tt) fp++; else if (!pp && tt) fn++; }
    double prec = (tp+fp) ? (double)tp/(tp+fp) : 0, rec = (tp+fn) ? (double)tp/(tp+fn) : 0;
    return {prec, rec, (prec+rec) ? 2*prec*rec/(prec+rec) : 0};
}
inline double r2Score(const Eigen::VectorXd& pred, const Eigen::VectorXd& y) {
    double ybar = y.mean(), ss_res = (y - pred).squaredNorm(), ss_tot = (y.array() - ybar).matrix().squaredNorm();
    return ss_tot > 0 ? 1.0 - ss_res / ss_tot : 0.0;
}

// ---- early stopping on a monitored (validation) loss ----
class EarlyStopping {
    double best_; int patience_, bad_;
public:
    explicit EarlyStopping(int patience = 10) : best_(1e300), patience_(patience), bad_(0) {}
    bool step(double val_loss) { if (val_loss < best_ - 1e-12) { best_ = val_loss; bad_ = 0; } else if (++bad_ >= patience_) return true; return false; }
    double best() const { return best_; }
};

// ---- LR schedules (return the LR to set on the optimizer for a given epoch) ----
inline double lrStep(double base, int epoch, int step_size, double gamma) { return base * std::pow(gamma, epoch / step_size); }
inline double lrExp(double base, int epoch, double gamma) { return base * std::pow(gamma, epoch); }
inline double lrCosine(double base, int epoch, int total) { return 0.5 * base * (1.0 + std::cos(3.14159265358979323846 * epoch / std::max(1, total))); }
inline double lrWarmupCosine(double base, int epoch, int warmup, int total) {
    if (epoch < warmup) return base * (epoch + 1.0) / warmup;
    return lrCosine(base, epoch - warmup, total - warmup);
}

// ---- checkpoint save / load (binary): self-hosted persistence ----
inline void saveParams(const std::vector<Tensor>& params, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    int np = (int)params.size(); f.write((char*)&np, sizeof(int));
    for (auto& p : params) {
        int rk = (int)p.p->shape.size(); f.write((char*)&rk, sizeof(int));
        for (int d : p.p->shape) f.write((char*)&d, sizeof(int));
        int n = p.p->size(); f.write((char*)&n, sizeof(int)); f.write((char*)p.p->v.data(), (long)n * sizeof(double));
    }
}
inline bool loadParams(std::vector<Tensor>& params, const std::string& path) {
    std::ifstream f(path, std::ios::binary); if (!f) return false;
    int np = 0; f.read((char*)&np, sizeof(int)); if (np != (int)params.size()) return false;
    for (auto& p : params) {
        int rk = 0; f.read((char*)&rk, sizeof(int)); for (int d = 0; d < rk; d++) { int dd; f.read((char*)&dd, sizeof(int)); }
        int n = 0; f.read((char*)&n, sizeof(int)); if (n != p.p->size()) return false;
        f.read((char*)p.p->v.data(), (long)n * sizeof(double));
    }
    return true;
}

// ===================================================================================
// [2d] PCA / SVD — classical linear dimensionality reduction (whitening / preprocessing
// AND a PCA-reconstruction-error anomaly baseline). Operates on Eigen matrices directly.
// ===================================================================================
class PCA {
public:
    Eigen::MatrixXd components;          // (D, k) top-k principal axes (orthonormal columns)
    Eigen::VectorXd mean;                // (D) feature means
    Eigen::VectorXd explained_variance;  // (k) variance along each component

    void fit(const Eigen::MatrixXd& X, int k) {
        const int N = (int)X.rows();
        mean = X.colwise().mean();
        Eigen::MatrixXd Xc = X.rowwise() - mean.transpose();
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(Xc, Eigen::ComputeThinV);
        const Eigen::MatrixXd& V = svd.matrixV();          // (D, min(N,D))
        Eigen::VectorXd S = svd.singularValues();
        int kk = std::min(k, (int)V.cols());
        components = V.leftCols(kk);
        explained_variance = (S.array().square() / std::max(1, N - 1)).head(kk);
    }
    Eigen::MatrixXd transform(const Eigen::MatrixXd& X) const {
        return (X.rowwise() - mean.transpose()) * components;             // (N, k)
    }
    Eigen::MatrixXd reconstruct(const Eigen::MatrixXd& X) const {
        return (transform(X) * components.transpose()).rowwise() + mean.transpose();   // (N, D)
    }
    // per-row reconstruction error — the classical anomaly score.
    Eigen::VectorXd reconstructionError(const Eigen::MatrixXd& X) const {
        return (reconstruct(X) - X).rowwise().squaredNorm();
    }
};

// ===================================================================================
// [2e] ANOMALY DETECTION (Co-requested) — deep Autoencoder: train on NORMAL data, score
// by reconstruction error (off-manifold points reconstruct poorly -> high score).
// ===================================================================================
class Autoencoder : public Module {
public:
    Sequential enc, dec;
    // symmetric: d_in -> hidden... -> d_latent (encoder); mirror back to d_in (decoder).
    Autoencoder(int d_in, std::vector<int> hidden, int d_latent, uint64_t seed = 1) {
        uint64_t s = seed; int prev = d_in;
        for (int h : hidden) { enc.add(std::make_shared<Linear>(prev, h, s++)); enc.add(std::make_shared<ReLU>()); prev = h; }
        enc.add(std::make_shared<Linear>(prev, d_latent, s++));
        prev = d_latent;
        for (auto it = hidden.rbegin(); it != hidden.rend(); ++it) { dec.add(std::make_shared<Linear>(prev, *it, s++)); dec.add(std::make_shared<ReLU>()); prev = *it; }
        dec.add(std::make_shared<Linear>(prev, d_in, s++));
    }
    Tensor encode(const Tensor& x) { return enc.forward(x); }
    Tensor forward(const Tensor& x) override { return dec.forward(enc.forward(x)); }
    std::vector<Tensor> parameters() override {
        auto a = enc.parameters(); auto b = dec.parameters(); a.insert(a.end(), b.begin(), b.end()); return a;
    }
};
// per-row reconstruction error between a model's output and the input (the anomaly score).
inline Eigen::VectorXd reconstructionError(const Tensor& recon, const Tensor& x) {
    int N = x.shape()[0], D = x.shape()[1]; Eigen::VectorXd e(N);
    const Eigen::VectorXd& r = recon.p->v; const Eigen::VectorXd& xv = x.p->v;
    for (int i = 0; i < N; i++) { double s = 0; for (int j = 0; j < D; j++) { double d = r(i*D+j) - xv(i*D+j); s += d*d; } e(i) = s; }
    return e;
}

// Variational Autoencoder: encoder -> (mu, logvar); reparam z = mu + e^{0.5·logvar}·eps;
// decoder -> recon.  Loss = reconstruction + beta·KL,  KL = -0.5 Σ(1 + logvar − mu² − e^{logvar}).
// run(x, eps) returns recon + the (scalar) KL; pass FIXED eps to gradient-check, fresh eps to train.
class VAE : public Module {
public:
    Sequential enc, dec; int latent;
    VAE(int d_in, int hidden, int latent_, uint64_t seed = 1) : latent(latent_) {
        enc.add(std::make_shared<Linear>(d_in, hidden, seed));     enc.add(std::make_shared<ReLU>());
        enc.add(std::make_shared<Linear>(hidden, 2*latent, seed+1));
        dec.add(std::make_shared<Linear>(latent, hidden, seed+2)); dec.add(std::make_shared<ReLU>());
        dec.add(std::make_shared<Linear>(hidden, d_in, seed+3));
    }
    struct Out { Tensor recon, kl, mu, logvar, z; };
    Out run(const Tensor& x, const Tensor& eps) {
        Tensor h = enc.forward(x);
        Tensor mu = sliceCols(h, 0, latent), logvar = sliceCols(h, latent, 2*latent);
        Tensor z = mu + expT(logvar * 0.5) * eps;                  // reparameterization
        Tensor recon = dec.forward(z);
        int n = x.shape()[0] * latent;                             // the +1 constant of the KL
        Tensor kl = (Tensor::scalar((double)n, false) + sum(logvar) - sum(mu*mu) - sum(expT(logvar))) * (-0.5);
        return {recon, kl, mu, logvar, z};
    }
    Tensor forward(const Tensor& x) override { return x; }         // use run()
    std::vector<Tensor> parameters() override {
        auto a = enc.parameters(); auto b = dec.parameters(); a.insert(a.end(), b.begin(), b.end()); return a;
    }
};

// ---- Isolation Forest (Liu et al. 2008): classical tree-based anomaly detector. ----
// Random axis-parallel splits isolate points; anomalies need FEWER splits (shorter path).
// score(x) = 2^{−E[path]/c(n)} ∈ (0,1); near 1 = anomalous, ~0.5 = normal.
struct ITreeNode { int feature = -1; double split = 0; int size = 0; bool leaf = true; std::unique_ptr<ITreeNode> left, right; };
class IsolationForest {
    int n_trees_, sample_size_, max_depth_; std::mt19937_64 rng_; double c_ = 1.0;
    std::vector<std::unique_ptr<ITreeNode>> trees_;
    static double cFactor(int n) { if (n <= 1) return 0.0; double H = std::log((double)(n-1)) + 0.5772156649015329; return 2.0*H - 2.0*(n-1)/n; }
    std::unique_ptr<ITreeNode> build(std::vector<int> idx, const Eigen::MatrixXd& X, int depth) {
        auto node = std::make_unique<ITreeNode>(); node->size = (int)idx.size();
        if (depth >= max_depth_ || idx.size() <= 1) return node;
        int D = (int)X.cols(); int f = std::uniform_int_distribution<int>(0, D-1)(rng_);
        double mn = 1e300, mx = -1e300; for (int i : idx) { double v = X(i,f); mn = std::min(mn,v); mx = std::max(mx,v); }
        if (mn >= mx) return node;
        double sp = std::uniform_real_distribution<double>(mn, mx)(rng_);
        std::vector<int> L, R; for (int i : idx) (X(i,f) < sp ? L : R).push_back(i);
        node->leaf = false; node->feature = f; node->split = sp;
        node->left = build(std::move(L), X, depth+1); node->right = build(std::move(R), X, depth+1);
        return node;
    }
    double pathLen(const Eigen::VectorXd& x, const ITreeNode* n, int depth) const {
        if (n->leaf) return depth + cFactor(n->size);
        return pathLen(x, (x(n->feature) < n->split ? n->left.get() : n->right.get()), depth+1);
    }
public:
    IsolationForest(int n_trees = 100, int sample_size = 128, uint64_t seed = 0)
        : n_trees_(n_trees), sample_size_(sample_size), rng_(seed) {
        max_depth_ = (int)std::ceil(std::log2((double)std::max(2, sample_size)));
    }
    void fit(const Eigen::MatrixXd& X) {
        int N = (int)X.rows(), ss = std::min(sample_size_, N); trees_.clear();
        for (int t = 0; t < n_trees_; t++) {
            std::vector<int> idx(N); for (int i = 0; i < N; i++) idx[i] = i;
            std::shuffle(idx.begin(), idx.end(), rng_); idx.resize(ss);
            trees_.push_back(build(std::move(idx), X, 0));
        }
        c_ = cFactor(ss);
    }
    double score(const Eigen::VectorXd& x) const {
        double s = 0; for (auto& t : trees_) s += pathLen(x, t.get(), 0); s /= trees_.size();
        return std::pow(2.0, -s / c_);
    }
    Eigen::VectorXd scores(const Eigen::MatrixXd& X) const {
        Eigen::VectorXd s(X.rows()); for (int i = 0; i < X.rows(); i++) s(i) = score(X.row(i).transpose()); return s;
    }
};

// ===================================================================================
// [4] Reinforcement learning — environments + tabular control + deep RL (DQN, policy
//     gradient, PPO).  The deep agents REUSE the autograd MLP above as their function
//     approximators, so their backward passes are the same gradient-checked engine.
//     Every agent has a learning anchor in nn_recovery.cpp (reaches a KNOWN return:
//     for GridWorld the optimum is computed exactly by value iteration on the model).
// ===================================================================================
namespace rl {

struct StepOut { Eigen::VectorXd obs; double reward; bool done; };

// ---- GridWorld: deterministic, fully-known model -> exact optimal return by DP --------
// Cells indexed s = r*cols + c.  Actions 0=up 1=down 2=left 3=right (edges = stay put).
// Reward −1 per step until the absorbing goal (so the optimum is −shortest-path-length).
class GridWorld {
public:
    int rows, cols, start, goal, maxSteps, steps, cur;
    std::vector<int> blocked;
    GridWorld(int r = 4, int c = 4, int start_ = 0, int goal_ = -1, int maxSteps_ = 100)
        : rows(r), cols(c), start(start_), goal(goal_ < 0 ? r * c - 1 : goal_),
          maxSteps(maxSteps_), steps(0), cur(start_) {}
    int nStates()  const { return rows * cols; }
    int nActions() const { return 4; }
    int obsDim()   const { return rows * cols; }
    int stateIndex() const { return cur; }
    bool isBlocked(int s) const { for (int b : blocked) if (b == s) return true; return false; }
    Eigen::VectorXd oneHot(int s) const { Eigen::VectorXd o = Eigen::VectorXd::Zero(rows * cols); o(s) = 1.0; return o; }
    Eigen::VectorXd obs() const { return oneHot(cur); }
    // pure model transition (used both by the live env AND by value iteration)
    void model(int s, int a, int& sp, double& r, bool& done) const {
        if (s == goal) { sp = s; r = 0.0; done = true; return; }       // absorbing
        int rr = s / cols, cc = s % cols, nr = rr, nc = cc;
        if      (a == 0) nr = std::max(0, rr - 1);
        else if (a == 1) nr = std::min(rows - 1, rr + 1);
        else if (a == 2) nc = std::max(0, cc - 1);
        else             nc = std::min(cols - 1, cc + 1);
        int ns = nr * cols + nc;
        if (isBlocked(ns)) ns = s;                                     // walls block movement
        sp = ns; r = -1.0; done = (ns == goal);
    }
    Eigen::VectorXd reset() { cur = start; steps = 0; return obs(); }
    StepOut step(int a) {
        int sp; double r; bool done; model(cur, a, sp, r, done);
        cur = sp; steps++; if (steps >= maxSteps) done = true;
        return { obs(), r, done };
    }
};

// exact optimal start-state return via value iteration on the known model (ground truth).
inline double gridworldOptimalReturn(const GridWorld& env, double gamma = 1.0) {
    int S = env.nStates(), A = env.nActions();
    Eigen::VectorXd V = Eigen::VectorXd::Zero(S);
    for (int it = 0; it < 100000; it++) {
        double delta = 0;
        for (int s = 0; s < S; s++) {
            if (s == env.goal) continue;
            double best = -1e300;
            for (int a = 0; a < A; a++) { int sp; double r; bool d; env.model(s, a, sp, r, d);
                best = std::max(best, r + (d ? 0.0 : gamma * V(sp))); }
            delta = std::max(delta, std::abs(best - V(s))); V(s) = best;
        }
        if (delta < 1e-12) break;
    }
    return V(env.start);
}

// ---- CartPole: classic continuous-control benchmark (Barto/Sutton dynamics) -----------
class CartPole {
    double g_ = 9.8, mc_ = 1.0, mp_ = 0.1, l_ = 0.5, fmag_ = 10.0, tau_ = 0.02;
    double x_, xd_, th_, thd_; int steps_, maxSteps_;
    std::mt19937_64 rng_;
public:
    explicit CartPole(int maxSteps = 200, uint64_t seed = 0) : steps_(0), maxSteps_(maxSteps), rng_(seed) {}
    int obsDim()   const { return 4; }
    int nActions() const { return 2; }
    Eigen::VectorXd state() const { Eigen::VectorXd s(4); s << x_, xd_, th_, thd_; return s; }
    Eigen::VectorXd reset() {
        std::uniform_real_distribution<double> u(-0.05, 0.05);
        x_ = u(rng_); xd_ = u(rng_); th_ = u(rng_); thd_ = u(rng_); steps_ = 0;
        return state();
    }
    StepOut step(int a) {
        double force = (a == 1 ? fmag_ : -fmag_);
        double ct = std::cos(th_), st = std::sin(th_), tm = mc_ + mp_, pml = mp_ * l_;
        double temp = (force + pml * thd_ * thd_ * st) / tm;
        double thacc = (g_ * st - ct * temp) / (l_ * (4.0 / 3.0 - mp_ * ct * ct / tm));
        double xacc  = temp - pml * thacc * ct / tm;
        x_ += tau_ * xd_;  xd_ += tau_ * xacc;
        th_ += tau_ * thd_; thd_ += tau_ * thacc;
        steps_++;
        bool done = (std::fabs(x_) > 2.4 || std::fabs(th_) > 0.2094 || steps_ >= maxSteps_);
        return { state(), 1.0, done };       // +1 per surviving step
    }
};

// ---- tabular control: Q-learning, SARSA, Double-Q ------------------------------------
class QLearning {
public:
    Eigen::MatrixXd Q; double alpha, gamma; std::mt19937_64 rng;
    QLearning(int S, int A, double alpha_ = 0.5, double gamma_ = 1.0, uint64_t seed = 0)
        : Q(Eigen::MatrixXd::Zero(S, A)), alpha(alpha_), gamma(gamma_), rng(seed) {}
    int greedy(int s) const { int a; Q.row(s).maxCoeff(&a); return a; }
    int epsGreedy(int s, double eps) {
        if (std::uniform_real_distribution<double>(0, 1)(rng) < eps)
            return std::uniform_int_distribution<int>(0, (int)Q.cols() - 1)(rng);
        return greedy(s);
    }
    void update(int s, int a, double r, int sp, bool done) {
        double target = r + (done ? 0.0 : gamma * Q.row(sp).maxCoeff());
        Q(s, a) += alpha * (target - Q(s, a));
    }
};

class SARSA {
public:
    Eigen::MatrixXd Q; double alpha, gamma; std::mt19937_64 rng;
    SARSA(int S, int A, double alpha_ = 0.5, double gamma_ = 1.0, uint64_t seed = 0)
        : Q(Eigen::MatrixXd::Zero(S, A)), alpha(alpha_), gamma(gamma_), rng(seed) {}
    int greedy(int s) const { int a; Q.row(s).maxCoeff(&a); return a; }
    int epsGreedy(int s, double eps) {
        if (std::uniform_real_distribution<double>(0, 1)(rng) < eps)
            return std::uniform_int_distribution<int>(0, (int)Q.cols() - 1)(rng);
        return greedy(s);
    }
    void update(int s, int a, double r, int sp, int ap, bool done) {        // on-policy target
        double target = r + (done ? 0.0 : gamma * Q(sp, ap));
        Q(s, a) += alpha * (target - Q(s, a));
    }
};

class DoubleQLearning {                                                     // de-biased Q-learning
public:
    Eigen::MatrixXd Qa, Qb; double alpha, gamma; std::mt19937_64 rng;
    DoubleQLearning(int S, int A, double alpha_ = 0.5, double gamma_ = 1.0, uint64_t seed = 0)
        : Qa(Eigen::MatrixXd::Zero(S, A)), Qb(Eigen::MatrixXd::Zero(S, A)),
          alpha(alpha_), gamma(gamma_), rng(seed) {}
    int greedy(int s) const { int a; (Qa.row(s) + Qb.row(s)).maxCoeff(&a); return a; }
    int epsGreedy(int s, double eps) {
        if (std::uniform_real_distribution<double>(0, 1)(rng) < eps)
            return std::uniform_int_distribution<int>(0, (int)Qa.cols() - 1)(rng);
        return greedy(s);
    }
    void update(int s, int a, double r, int sp, bool done) {
        if (std::uniform_int_distribution<int>(0, 1)(rng) == 0) {
            int astar; Qa.row(sp).maxCoeff(&astar);
            double target = r + (done ? 0.0 : gamma * Qb(sp, astar));
            Qa(s, a) += alpha * (target - Qa(s, a));
        } else {
            int bstar; Qb.row(sp).maxCoeff(&bstar);
            double target = r + (done ? 0.0 : gamma * Qa(sp, bstar));
            Qb(s, a) += alpha * (target - Qb(s, a));
        }
    }
};

// ---- helpers shared by the deep agents ------------------------------------------------
inline std::shared_ptr<Sequential> mlp(const std::vector<int>& dims, uint64_t seed = 1) {
    auto net = std::make_shared<Sequential>();
    for (size_t i = 0; i + 1 < dims.size(); i++) {
        net->add(std::make_shared<Linear>(dims[i], dims[i + 1], seed + i * 7919));
        if (i + 2 < dims.size()) net->add(std::make_shared<ReLU>());        // hidden -> ReLU; output raw
    }
    return net;
}
inline void copyParams(std::vector<Tensor> dst, const std::vector<Tensor>& src) {
    for (size_t i = 0; i < dst.size(); i++) dst[i].p->v = src[i].p->v;
}

struct Transition { Eigen::VectorXd s; int a; double r; Eigen::VectorXd sp; bool done; };

class ReplayBuffer {
    std::vector<Transition> buf_; size_t cap_, pos_ = 0; std::mt19937_64 rng_;
public:
    ReplayBuffer(size_t capacity, uint64_t seed = 0) : cap_(capacity), rng_(seed) { buf_.reserve(capacity); }
    void push(const Transition& t) { if (buf_.size() < cap_) buf_.push_back(t); else buf_[pos_] = t; pos_ = (pos_ + 1) % cap_; }
    size_t size() const { return buf_.size(); }
    std::vector<Transition> sample(int n) {
        std::vector<Transition> out; out.reserve(n);
        std::uniform_int_distribution<size_t> d(0, buf_.size() - 1);
        for (int i = 0; i < n; i++) out.push_back(buf_[d(rng_)]);
        return out;
    }
};

// ---- DQN: MLP Q-network + replay buffer + target net (+ optional Double-DQN target) ----
class DQNAgent {
public:
    std::shared_ptr<Sequential> online, target;
    std::unique_ptr<Adam> opt;
    ReplayBuffer buf;
    int obsDim, nActions; double gamma; bool doubleDQN; std::mt19937_64 rng;

    DQNAgent(int obsDim_, int nActions_, std::vector<int> hidden, double lr = 1e-3,
             double gamma_ = 0.99, int bufCap = 20000, bool doubleDQN_ = true, uint64_t seed = 0)
        : buf(bufCap, seed + 1), obsDim(obsDim_), nActions(nActions_),
          gamma(gamma_), doubleDQN(doubleDQN_), rng(seed) {
        std::vector<int> dims{ obsDim_ }; for (int h : hidden) dims.push_back(h); dims.push_back(nActions_);
        online = mlp(dims, seed * 2 + 1);
        target = mlp(dims, seed * 2 + 1);
        copyParams(target->parameters(), online->parameters());
        opt.reset(new Adam(online->parameters(), lr));
    }
    int act(const Eigen::VectorXd& s, double eps) {
        if (std::uniform_real_distribution<double>(0, 1)(rng) < eps)
            return std::uniform_int_distribution<int>(0, nActions - 1)(rng);
        Tensor x = Tensor::from({ 1, obsDim }, s, false);
        Eigen::VectorXd q = online->forward(x).value(); int a; q.maxCoeff(&a); return a;
    }
    void remember(const Transition& t) { buf.push(t); }
    void syncTarget() { copyParams(target->parameters(), online->parameters()); }

    double trainStep(int batch) {
        if ((int)buf.size() < batch) return 0.0;
        auto B = buf.sample(batch); int N = (int)B.size();
        Eigen::VectorXd Sflat(N * obsDim), SPflat(N * obsDim); std::vector<int> acts(N);
        for (int i = 0; i < N; i++) { acts[i] = B[i].a;
            Sflat.segment(i * obsDim, obsDim) = B[i].s; SPflat.segment(i * obsDim, obsDim) = B[i].sp; }
        // --- TD targets (detached: read raw values, no autograd through the target net) ---
        Tensor XP = Tensor::from({ N, obsDim }, SPflat, false);
        Tensor QtT = target->forward(XP); Eigen::Map<RowMat> Qt(QtT.value().data(), N, nActions);
        Eigen::VectorXd tgt(N);
        if (doubleDQN) {
            Tensor QoT = online->forward(XP); Eigen::Map<RowMat> Qo(QoT.value().data(), N, nActions);
            for (int i = 0; i < N; i++) { if (B[i].done) { tgt(i) = B[i].r; continue; }
                int am; Qo.row(i).maxCoeff(&am); tgt(i) = B[i].r + gamma * Qt(i, am); }
        } else {
            for (int i = 0; i < N; i++) tgt(i) = B[i].done ? B[i].r : B[i].r + gamma * Qt.row(i).maxCoeff();
        }
        // --- online forward (autograd) + gather the acted-on Q + Huber-free MSE regression ---
        Tensor X = Tensor::from({ N, obsDim }, Sflat, false);
        Tensor Q = online->forward(X);
        Tensor qa = gatherColumns(Q, acts);
        Tensor loss = mseLoss(qa, Tensor::from({ N, 1 }, tgt, false));
        opt->zeroGrad(); loss.backward(); opt->step();
        return loss.item();
    }
};

// ---- REINFORCE: Monte-Carlo policy gradient with a standardized-return baseline -------
class ReinforceAgent {
public:
    std::shared_ptr<Sequential> policy;
    std::unique_ptr<Adam> opt;
    int obsDim, nActions; double gamma, entCoef; bool baseline; std::mt19937_64 rng;
    ReinforceAgent(int obsDim_, int nActions_, std::vector<int> hidden, double lr = 1e-2,
                   double gamma_ = 0.99, bool baseline_ = true, uint64_t seed = 0, double entCoef_ = 0.01)
        : obsDim(obsDim_), nActions(nActions_), gamma(gamma_), entCoef(entCoef_), baseline(baseline_), rng(seed) {
        std::vector<int> dims{ obsDim_ }; for (int h : hidden) dims.push_back(h); dims.push_back(nActions_);
        policy = mlp(dims, seed * 3 + 7);
        opt.reset(new Adam(policy->parameters(), lr));
    }
    Eigen::VectorXd probs(const Eigen::VectorXd& s) {
        Tensor x = Tensor::from({ 1, obsDim }, s, false);
        return softmaxRow(policy->forward(x)).value();
    }
    int sample(const Eigen::VectorXd& s) {
        Eigen::VectorXd p = probs(s);
        double u = std::uniform_real_distribution<double>(0, 1)(rng), c = 0;
        for (int k = 0; k < nActions; k++) { c += p(k); if (u <= c) return k; }
        return nActions - 1;
    }
    double update(const std::vector<Eigen::VectorXd>& S, const std::vector<int>& A, const std::vector<double>& R) {
        int T = (int)S.size();
        Eigen::VectorXd G(T); double g = 0;
        for (int t = T - 1; t >= 0; t--) { g = R[t] + gamma * g; G(t) = g; }
        Eigen::VectorXd adv = G;
        if (baseline) { double m = G.mean(), sd = std::sqrt((G.array() - m).square().mean() + 1e-8);
                        adv = (G.array() - m) / sd; }
        Eigen::VectorXd Sflat(T * obsDim); for (int t = 0; t < T; t++) Sflat.segment(t * obsDim, obsDim) = S[t];
        Tensor X = Tensor::from({ T, obsDim }, Sflat, false);
        Tensor logits = policy->forward(X);
        Tensor logp = gatherColumns(logSoftmaxRow(logits), A);                       // (T,1)
        Tensor pg = sum(logp * Tensor::from({ T, 1 }, adv, false)) * (-1.0);         // −Σ logπ·A
        // entropy bonus: +entCoef·Σ(π·logπ) = −entCoef·H → keeps the policy exploring (prevents a
        // premature collapse to a degenerate argmax that can loop and never reach the goal).
        Tensor ent = sum(softmaxRow(logits) * logSoftmaxRow(logits));
        Tensor loss = pg + ent * entCoef;
        opt->zeroGrad(); loss.backward(); opt->step();
        return loss.item();
    }
};

// ---- PPO: clipped-surrogate actor-critic with GAE(λ) ----------------------------------
class PPOAgent {
public:
    std::shared_ptr<Sequential> actor, critic;
    std::unique_ptr<Adam> optA, optC;
    int obsDim, nActions; double gamma, lam, clip, entCoef; int epochs; std::mt19937_64 rng;
    PPOAgent(int obsDim_, int nActions_, std::vector<int> hidden, double lr = 3e-3,
             double gamma_ = 0.99, double lam_ = 0.95, double clip_ = 0.2,
             double entCoef_ = 0.01, int epochs_ = 4, uint64_t seed = 0)
        : obsDim(obsDim_), nActions(nActions_), gamma(gamma_), lam(lam_), clip(clip_),
          entCoef(entCoef_), epochs(epochs_), rng(seed) {
        std::vector<int> ad{ obsDim_ }; for (int h : hidden) ad.push_back(h); ad.push_back(nActions_);
        std::vector<int> cd{ obsDim_ }; for (int h : hidden) cd.push_back(h); cd.push_back(1);
        actor  = mlp(ad, seed * 5 + 3);
        critic = mlp(cd, seed * 7 + 9);
        optA.reset(new Adam(actor->parameters(), lr));
        optC.reset(new Adam(critic->parameters(), lr));
    }
    Eigen::VectorXd probs(const Eigen::VectorXd& s) {
        Tensor x = Tensor::from({ 1, obsDim }, s, false);
        return softmaxRow(actor->forward(x)).value();
    }
    int sample(const Eigen::VectorXd& s) {
        Eigen::VectorXd p = probs(s);
        double u = std::uniform_real_distribution<double>(0, 1)(rng), c = 0;
        for (int k = 0; k < nActions; k++) { c += p(k); if (u <= c) return k; }
        return nActions - 1;
    }
    double value(const Eigen::VectorXd& s) {
        Tensor x = Tensor::from({ 1, obsDim }, s, false);
        return critic->forward(x).item();
    }
    // One PPO update over a collected rollout (states, actions, rewards, done-flags).
    void update(const std::vector<Eigen::VectorXd>& S, const std::vector<int>& A,
                const std::vector<double>& R, const std::vector<char>& done) {
        int T = (int)S.size();
        Eigen::VectorXd Sflat(T * obsDim); for (int t = 0; t < T; t++) Sflat.segment(t * obsDim, obsDim) = S[t];
        Tensor X = Tensor::from({ T, obsDim }, Sflat, false);
        // value estimates V(s_t) and bootstrap V(s_{t+1}) (detached)
        Eigen::VectorXd V(T); { Eigen::VectorXd vv = critic->forward(X).value(); V = vv; }
        // GAE(λ) advantages + returns
        Eigen::VectorXd adv(T), ret(T); double gae = 0;
        for (int t = T - 1; t >= 0; t--) {
            double vnext = (done[t] || t == T - 1) ? 0.0 : V(t + 1);
            double nonterm = done[t] ? 0.0 : 1.0;
            double delta = R[t] + gamma * vnext * nonterm - V(t);
            gae = delta + gamma * lam * nonterm * gae;
            adv(t) = gae; ret(t) = gae + V(t);
        }
        double am = adv.mean(), asd = std::sqrt((adv.array() - am).square().mean() + 1e-8);
        Eigen::VectorXd advN = (adv.array() - am) / asd;
        // old log-probs (detached)
        Eigen::VectorXd logpOld(T);
        { Tensor lp = gatherColumns(logSoftmaxRow(actor->forward(X)), A); logpOld = lp.value(); }
        Tensor advT = Tensor::from({ T, 1 }, advN, false);
        Tensor oldT = Tensor::from({ T, 1 }, logpOld, false);
        Tensor retT = Tensor::from({ T, 1 }, ret, false);
        for (int e = 0; e < epochs; e++) {
            // actor: clipped surrogate + entropy bonus
            Tensor logits = actor->forward(X);
            Tensor logp = gatherColumns(logSoftmaxRow(logits), A);          // (T,1)
            Tensor ratio = expT(logp - oldT);                               // (T,1)
            Tensor surr1 = ratio * advT;
            Tensor surr2 = clampT(ratio, 1.0 - clip, 1.0 + clip) * advT;
            Tensor pg = mean(minT(surr1, surr2)) * (-1.0);                  // maximize -> minimize neg
            Tensor probsT = softmaxRow(logits);
            Tensor ent = sum(probsT * logSoftmaxRow(logits)) * (1.0 / T);   // −entropy (mean)
            Tensor aloss = pg + ent * entCoef;                             // +entCoef·(−entropy)
            optA->zeroGrad(); aloss.backward(); optA->step();
            // critic: value regression to returns
            Tensor v = critic->forward(X);
            Tensor closs = mseLoss(v, retT);
            optC->zeroGrad(); closs.backward(); optC->step();
        }
    }
};

} // namespace rl

}} // namespace djehuti::nn
