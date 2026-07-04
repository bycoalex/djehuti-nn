// =====================================================================
// nn_demo.cpp — the 30-second "it learns" demo (AI suite, impulse-buy entry).
//
// Trains a small neural net on a 3-class problem to high accuracy in a blink,
// then flags anomalies with an Isolation Forest — all in pure C++17 + Eigen,
// zero external dependencies.  For the rigor proof (per-op gradient checks +
// every-architecture learning anchor) run examples/nn_recovery.cpp; for the
// self-hosted Transformer language model run examples/nn_llm_demo.cpp.
//
// build: g++-15 -std=c++17 -O3 -march=native -fopenmp -I. -I<eigen> \
//        examples/nn_demo.cpp -o build/nn_demo -lpthread -lm
//
// Copyright © Co Bros & Alex Ramanaidou (Webuildcodes). All rights reserved.
// Licensed under the Non‑Exclusive Source License – see LICENSE file.
//
// =====================================================================
#include "djehuti_nn.hpp"
#include <cstdio>
#include <vector>
#include <random>

using namespace djehuti::nn;

int main() {
    std::printf("Djehuti SDK — AI suite quick demo (train a net + detect anomalies, zero deps)\n\n");
    int fails = 0;

    // ---- 1) a classifier learns 3 Gaussian blobs (softmax + cross-entropy) ----
    std::printf("== a 2->16->3 MLP classifies three clusters ==\n");
    std::mt19937_64 g(1); std::normal_distribution<double> nd(0.0, 0.35);
    const int N = 240; Eigen::MatrixXd X(N, 2); std::vector<int> y(N);
    const double cx[3] = {0, 2, -2}, cy[3] = {0, 2, 2};
    for (int i = 0; i < N; i++) { int c = i % 3; y[i] = c; X(i,0) = cx[c] + nd(g); X(i,1) = cy[c] + nd(g); }

    djehuti::nn::Sequential net;                                // (qualified: Eigen squats "Sequential")
    net.add(std::make_shared<Linear>(2, 16, 1));
    net.add(std::make_shared<ReLU>());
    net.add(std::make_shared<Linear>(16, 3, 2));
    auto params = net.parameters();
    Adam opt(params, 5e-2);

    Eigen::VectorXd xflat(N * 2); for (int i = 0; i < N; i++) { xflat(2*i) = X(i,0); xflat(2*i+1) = X(i,1); }
    Tensor Xt = Tensor::from({N, 2}, xflat, false);            // row-major flatten
    double loss0 = 0, loss = 0;
    for (int e = 0; e < 200; e++) {
        opt.zeroGrad();
        Tensor logits = net.forward(Xt);
        Tensor L = crossEntropyLoss(logits, y);
        L.backward(); opt.step();
        loss = L.item(); if (e == 0) loss0 = loss;
    }
    double acc = accuracy(net.forward(Xt), y);
    std::printf("  loss %.3f -> %.3f   train accuracy = %.1f%%\n", loss0, loss, 100.0 * acc);
    bool okc = acc > 0.95;
    std::printf("  [%s] the net learns the classes (>95%%)\n", okc ? "PASS" : "FAIL"); if (!okc) fails++;

    // ---- 2) Isolation Forest flags outliers (business-grade anomaly detection) ----
    std::printf("\n== an Isolation Forest flags off-distribution points ==\n");
    const int Nn = 200, Na = 20; Eigen::MatrixXd Xn(Nn, 2), Xa(Na, 2);
    for (int i = 0; i < Nn; i++) { Xn(i,0) = nd(g); Xn(i,1) = nd(g); }            // normal
    for (int i = 0; i < Na; i++) { Xa(i,0) = 5.0 + nd(g); Xa(i,1) = 5.0 + nd(g); } // outliers
    IsolationForest iso(150, 128, 7); iso.fit(Xn);
    double sn = iso.scores(Xn).mean(), sa = iso.scores(Xa).mean();
    std::printf("  mean anomaly score: normal=%.3f  outliers=%.3f\n", sn, sa);
    bool oka = sa > sn + 0.1;
    std::printf("  [%s] outliers score higher than normal data\n", oka ? "PASS" : "FAIL"); if (!oka) fails++;

    std::printf("\n%s — that's the entry suite. Pro adds Transformers, deep RL, and a self-hosted LLM.\n",
                fails ? "SOME FAILURES" : "ALL PASS");
    return fails ? 1 : 0;
}
