// =====================================================================
// nn_recovery.cpp — NN suite (#34) validation.
// THE headline rigor proof ("we don't sell a toy"): for every autograd op,
// the analytic (reverse-mode) gradient must match a central FINITE-DIFFERENCE
// gradient to ~1e-6. A single mismatched backward =>
//
//   [1] autograd core: per-op gradient checks + a full MLP-shaped graph.   <-- HERE
//   (later steps add: learning anchors, layers, optimizers, RL, LLM.)
//
// build (NaN-honest): g++-15 -std=c++17 -O3 -march=native -fopenmp -I. -I<eigen> \
//                     examples/nn_recovery.cpp -o build/nn_recovery -lpthread -lm
//
// Copyright © Co Bros & Alex Ramanaidou (Webuildcodes). All rights reserved.
// Licensed under the Non‑Exclusive Source License – see LICENSE file.
//
// =====================================================================
#include "djehuti_nn.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <functional>
#include <random>

using namespace djehuti::nn;

static int fails = 0;

// Central finite-difference gradient check: rebuilds the graph via f() reading the
// CURRENT leaf values, so perturbing a leaf and re-running f() gives the numeric grad.
static void gradCheck(const char* name, std::vector<Tensor> leaves, std::function<Tensor()> f,
                      double h = 1e-5, double tol = 1e-5) {
    for (auto& L : leaves) L.zeroGrad();
    Tensor y = f();
    y.backward();
    std::vector<Eigen::VectorXd> analytic;
    for (auto& L : leaves) analytic.push_back(L.grad());

    double maxerr = 0; int worst_leaf = -1, worst_i = -1;
    for (size_t li = 0; li < leaves.size(); li++) {
        for (int i = 0; i < leaves[li].size(); i++) {
            double orig = leaves[li].value()(i);
            leaves[li].value()(i) = orig + h; double fp = f().item();
            leaves[li].value()(i) = orig - h; double fm = f().item();
            leaves[li].value()(i) = orig;
            double fd = (fp - fm) / (2.0 * h);
            double err = std::fabs(fd - analytic[li](i));
            if (err > maxerr) { maxerr = err; worst_leaf = (int)li; worst_i = i; }
        }
    }
    bool ok = maxerr <= tol;
    std::printf("  [%s] grad-check %-22s  max|Δ|=%.2e  (leaf %d[%d])\n",
                ok ? "PASS" : "FAIL", name, maxerr, worst_leaf, worst_i);
    if (!ok) fails++;
}

int main() {
    std::printf("Djehuti SDK — NN suite recovery [step 1: autograd gradient checks]\n\n");
    std::printf("== per-op gradient checks (analytic backward vs central finite-diff) ==\n");

    // elementwise add / sub / mul, scalar mul
    {
        Tensor a = Tensor::randn({3,2}, 1.0, 1), b = Tensor::randn({3,2}, 1.0, 2);
        gradCheck("add",    {a,b}, [&]{ return sum(a + b); });
        gradCheck("sub",    {a,b}, [&]{ return sum(a - b); });
        gradCheck("mul",    {a,b}, [&]{ return sum(a * b); });
        gradCheck("scale",  {a},   [&]{ return sum(a * 2.5); });
    }
    // matmul
    {
        Tensor A = Tensor::randn({3,4}, 1.0, 3), B = Tensor::randn({4,2}, 1.0, 4);
        gradCheck("matmul", {A,B}, [&]{ return sum(matmul(A, B)); });
    }
    // bias broadcast
    {
        Tensor X = Tensor::randn({4,3}, 1.0, 5), b = Tensor::randn({3}, 1.0, 6);
        gradCheck("addBias", {X,b}, [&]{ return sum(addBias(X, b)); });
    }
    // activations
    {
        Tensor a = Tensor::randn({4,3}, 1.0, 7);
        gradCheck("relu",      {a}, [&]{ return sum(relu(a)); });
        gradCheck("sigmoid",   {a}, [&]{ return sum(sigmoidT(a)); });
        gradCheck("tanh",      {a}, [&]{ return sum(tanhT(a)); });
        gradCheck("leakyRelu", {a}, [&]{ return sum(leakyRelu(a, 0.1)); });
        gradCheck("gelu",      {a}, [&]{ return sum(gelu(a)); });
        gradCheck("silu",      {a}, [&]{ return sum(silu(a)); });
        gradCheck("expT",      {a}, [&]{ return sum(expT(a)); });
    }
    // softmax (non-trivial scalar: sum(softmax∘const) — sum(softmax) alone is constant) + classification losses
    {
        Tensor a = Tensor::randn({4,3}, 1.0, 15);
        Tensor C = Tensor::randn({4,3}, 1.0, 16);                 // constant weights
        gradCheck("softmaxRow",   {a}, [&]{ return sum(softmaxRow(a) * C); });
        std::vector<int> tgt = {0, 2, 1, 0};
        gradCheck("crossEntropy", {a}, [&]{ return crossEntropyLoss(a, tgt); });
        Tensor z = Tensor::randn({5,1}, 1.0, 17);
        Eigen::VectorXd yv(5); yv << 1,0,1,1,0;
        Tensor y = Tensor::from({5,1}, yv, false);
        gradCheck("bceWithLogits", {z}, [&]{ return bceWithLogits(z, y); });
    }
    // RL-policy ops: logT, log-softmax, column-gather, clamp, elementwise-min (PPO/REINFORCE/DQN)
    {
        Tensor a = Tensor::randn({4,3}, 1.0, 30);
        Tensor pos = Tensor::randn({3,2}, 0.5, 31); for (int i = 0; i < pos.size(); i++) pos.value()(i) = std::fabs(pos.value()(i)) + 0.3;
        Tensor Cls = Tensor::randn({4,3}, 1.0, 33);              // constant weights for log-softmax
        Tensor Cg = Tensor::randn({4,1}, 1.0, 32);
        std::vector<int> idx = {0, 2, 1, 0};
        gradCheck("logT",          {pos}, [&]{ return sum(logT(pos)); });
        gradCheck("logSoftmaxRow", {a},   [&]{ return sum(logSoftmaxRow(a) * Cls); });
        gradCheck("gatherColumns", {a},   [&]{ return sum(gatherColumns(a, idx) * Cg); });
        Tensor r = Tensor::randn({5,1}, 0.4, 34); for (int i = 0; i < r.size(); i++) r.value()(i) += 1.0;  // ratios near 1
        Tensor adv = Tensor::randn({5,1}, 1.0, 35);
        gradCheck("clampT",        {r},   [&]{ return sum(clampT(r, 0.8, 1.2) * adv); });
        Tensor b = Tensor::randn({5,1}, 1.0, 36);
        gradCheck("minT",          {r,b}, [&]{ return sum(minT(r, b)); });
    }
    // shape ops + LayerNorm + Embedding (transformer prerequisites)
    {
        Tensor a = Tensor::randn({4,3}, 1.0, 20), Ct = Tensor::randn({3,4}, 1.0, 21);
        gradCheck("transpose2d", {a}, [&]{ return sum(transpose2d(a) * Ct); });

        Tensor a6 = Tensor::randn({4,6}, 1.0, 22), C34 = Tensor::randn({4,3}, 1.0, 23);
        gradCheck("sliceCols", {a6}, [&]{ return sum(sliceCols(a6, 1, 4) * C34); });

        Tensor u = Tensor::randn({4,2}, 1.0, 24), v = Tensor::randn({4,3}, 1.0, 25), C45 = Tensor::randn({4,5}, 1.0, 26);
        gradCheck("concatCols", {u,v}, [&]{ return sum(concatCols(u, v) * C45); });

        Tensor x = Tensor::randn({4,5}, 1.0, 27), gm = Tensor::randn({5}, 0.5, 28),
               bt = Tensor::randn({5}, 0.5, 29), C45b = Tensor::randn({4,5}, 1.0, 30);
        gradCheck("layerNorm dx/dγ/dβ", {x, gm, bt}, [&]{ return sum(layerNorm(x, gm, bt) * C45b); });

        Tensor W = Tensor::randn({6,4}, 1.0, 31), C44 = Tensor::randn({4,4}, 1.0, 32);
        std::vector<int> idx = {2, 0, 5, 1};
        gradCheck("embeddingLookup", {W}, [&]{ return sum(embeddingLookup(W, idx) * C44); });
    }
    // a FULL Multi-Head Attention block: gradient-check its output wrt input + ALL 8 params.
    // (the whole attention backward is composed of the verified ops -> must be exact too)
    {
        MultiHeadAttention mha(4, 2, /*causal=*/false, 3);
        Tensor x = Tensor::randn({3,4}, 1.0, 40, true), C = Tensor::randn({3,4}, 1.0, 41);
        gradCheck("MultiHeadAttention", {x, mha.Wq, mha.bq, mha.Wk, mha.bk, mha.Wv, mha.bv, mha.Wo, mha.bo},
                  [&]{ return sum(mha.forward(x) * C); }, 1e-5, 1e-4);
    }
    // Conv2d (explicit-loop) + pooling: gradient-check dx/dW/db and the pool routings
    {
        Tensor x  = Tensor::randn({2,4,4}, 1.0, 50, true);
        Tensor Wc = Tensor::randn({3*2*3*3}, 0.5, 51, true), bc = Tensor::randn({3}, 0.3, 52, true);
        Tensor Cc = Tensor::randn({3,4,4}, 1.0, 53);
        gradCheck("conv2d dx/dW/db", {x, Wc, bc},
                  [&]{ return sum(conv2d(x, Wc, bc, 2, 4, 4, 3, 3, 3, 1, 1) * Cc); }, 1e-5, 1e-4);

        Tensor xp = Tensor::randn({1,4,4}, 1.0, 54, true), Cp = Tensor::randn({1,2,2}, 1.0, 55);
        gradCheck("maxPool2d", {xp}, [&]{ return sum(maxPool2d(xp, 1, 4, 4, 2, 2) * Cp); });
        gradCheck("avgPool2d", {xp}, [&]{ return sum(avgPool2d(xp, 1, 4, 4, 2, 2) * Cp); });
    }
    // recurrent cells unrolled over T=3 (back-prop-through-time): gradient-check input + ALL params
    {
        Tensor x = Tensor::randn({3,2}, 1.0, 60, true), C = Tensor::randn({3,4}, 1.0, 61);   // T=3, in=2, hid=4
        RNN rnn(2, 4, 1);
        gradCheck("RNN (BPTT)",  {x, rnn.Wx, rnn.Wh, rnn.b}, [&]{ return sum(rnn.forward(x) * C); }, 1e-5, 1e-4);
        LSTM lstm(2, 4, 1);
        gradCheck("LSTM (BPTT)", {x, lstm.Wx, lstm.Wh, lstm.b}, [&]{ return sum(lstm.forward(x) * C); }, 1e-5, 1e-4);
        GRU gru(2, 4, 1);
        gradCheck("GRU (BPTT)",  {x, gru.Wxz, gru.Whz, gru.bz, gru.Wxr, gru.Whr, gru.br, gru.Wxn, gru.Whn, gru.bn},
                  [&]{ return sum(gru.forward(x) * C); }, 1e-5, 1e-4);
    }
    // reductions
    {
        Tensor a = Tensor::randn({5,2}, 1.0, 8);
        gradCheck("sum",  {a}, [&]{ return sum(a); });
        gradCheck("mean", {a}, [&]{ return mean(a); });
    }
    // mse loss vs a constant target
    {
        Tensor pred = Tensor::randn({4,2}, 1.0, 9);
        Tensor tgt  = Tensor::randn({4,2}, 1.0, 10);   // treated as constant
        gradCheck("mseLoss", {pred}, [&]{ return mseLoss(pred, tgt); });
    }

    std::printf("\n== full MLP-shaped graph (the real chain: matmul->bias->relu->matmul->bias->mse) ==\n");
    {
        // x (4x3) input; 3->5 relu -> 5->2 linear; MSE to target. Check ALL parameter grads.
        Tensor x  = Tensor::randn({4,3}, 1.0, 11, false);
        Tensor W1 = Tensor::randn({3,5}, 0.5, 12), b1 = Tensor::zeros({5}, true);
        Tensor W2 = Tensor::randn({5,2}, 0.5, 13), b2 = Tensor::zeros({2}, true);
        Tensor y  = Tensor::randn({4,2}, 1.0, 14, false);
        auto fwd = [&]{
            Tensor h = relu(addBias(matmul(x, W1), b1));
            Tensor o = addBias(matmul(h, W2), b2);
            return mseLoss(o, y);
        };
        gradCheck("MLP dW1/db1/dW2/db2", {W1, b1, W2, b2}, fwd);
    }

    std::printf("\n== it LEARNS: a 2-8-1 MLP solves XOR (the classic non-linear test) ==\n");
    {
        Eigen::VectorXd xs(8); xs << 0,0,  0,1,  1,0,  1,1;       // 4x2 row-major
        Eigen::VectorXd ys(4); ys << 0,    1,    1,    0;          // 4x1  (XOR)
        Tensor X = Tensor::from({4,2}, xs, false);
        Tensor Y = Tensor::from({4,1}, ys, false);

        djehuti::nn::Sequential net;     // qualified: Eigen also defines a 'Sequential' token
        net.add(std::make_shared<Linear>(2, 8, 1));
        net.add(std::make_shared<Tanh>());
        net.add(std::make_shared<Linear>(8, 1, 2));
        net.add(std::make_shared<Sigmoid>());

        Adam opt(net.parameters(), 0.05);
        double loss0 = 0, loss = 0;
        for (int epoch = 0; epoch < 3000; epoch++) {
            opt.zeroGrad();
            Tensor L = mseLoss(net(X), Y);
            L.backward();
            opt.step();
            if (epoch == 0) loss0 = L.item();
            loss = L.item();
        }
        Tensor pred = net(X);
        std::printf("  loss: %.4f -> %.6f   preds: [%.3f %.3f %.3f %.3f]  (want 0 1 1 0)\n",
                    loss0, loss, pred.value()(0), pred.value()(1), pred.value()(2), pred.value()(3));
        bool classified = (pred.value()(0) < 0.5) && (pred.value()(1) > 0.5) &&
                          (pred.value()(2) > 0.5) && (pred.value()(3) < 0.5);
        std::printf("  [%s] training loss collapses (<0.01)\n", loss < 0.01 ? "PASS" : "FAIL");  if (loss >= 0.01) fails++;
        std::printf("  [%s] all four XOR points classified correctly\n", classified ? "PASS" : "FAIL"); if (!classified) fails++;
    }

    std::printf("\n== it CLASSIFIES: a 2-16-3 MLP separates 3 Gaussian blobs (softmax + cross-entropy) ==\n");
    {
        std::mt19937_64 g(2024); std::normal_distribution<double> noise(0.0, 0.5);
        const int per = 40, K = 3, N = per * K;
        const double cx[3] = {-2, 2, 0}, cy[3] = {-2, -2, 2};
        Eigen::VectorXd xs(N * 2); std::vector<int> tgt(N);
        for (int k = 0; k < K; k++)
            for (int j = 0; j < per; j++) { int i = k * per + j;
                xs(i*2) = cx[k] + noise(g); xs(i*2+1) = cy[k] + noise(g); tgt[i] = k; }
        Tensor X = Tensor::from({N, 2}, xs, false);

        djehuti::nn::Sequential net;
        net.add(std::make_shared<Linear>(2, 16, 3));
        net.add(std::make_shared<ReLU>());
        net.add(std::make_shared<Linear>(16, 3, 4));
        Adam opt(net.parameters(), 0.05);
        double loss0 = 0, loss = 0;
        for (int e = 0; e < 400; e++) {
            opt.zeroGrad(); Tensor L = crossEntropyLoss(net(X), tgt); L.backward(); opt.step();
            if (e == 0) loss0 = L.item(); loss = L.item();
        }
        Tensor logits = net(X);
        Eigen::Map<RowMat> Z(logits.value().data(), N, 3);
        int correct = 0;
        for (int i = 0; i < N; i++) { int am = 0; double best = Z(i, 0);
            for (int c = 1; c < 3; c++) if (Z(i, c) > best) { best = Z(i, c); am = c; }
            if (am == tgt[i]) correct++; }
        double acc = (double)correct / N;
        std::printf("  loss %.3f -> %.4f   accuracy = %.1f%% (%d/%d)\n", loss0, loss, 100*acc, correct, N);
        std::printf("  [%s] cross-entropy loss more than halves\n", loss < loss0 * 0.5 ? "PASS" : "FAIL"); if (!(loss < loss0 * 0.5)) fails++;
        std::printf("  [%s] 3-class accuracy > 95%%\n", acc > 0.95 ? "PASS" : "FAIL"); if (!(acc > 0.95)) fails++;
    }

    std::printf("\n== it SEES: a tiny CNN classifies 5x5 images (vertical vs horizontal bar) ==\n");
    {
        std::mt19937_64 g(123); std::uniform_int_distribution<int> pos(0, 4);
        const int NS = 16;
        std::vector<Eigen::VectorXd> imgs; std::vector<int> labels;
        for (int s = 0; s < NS; s++) {
            Eigen::VectorXd img = Eigen::VectorXd::Zero(25); int lbl = s % 2;
            if (lbl == 0) { int col = pos(g); for (int r = 0; r < 5; r++) img(r*5 + col) = 1.0; }  // vertical bar
            else          { int row = pos(g); for (int c = 0; c < 5; c++) img(row*5 + c) = 1.0; }  // horizontal bar
            imgs.push_back(img); labels.push_back(lbl);
        }
        Conv2d conv(1, 4, 3, 1, 1, 7);                       // 1->4 ch, 3x3, pad1 -> (4,5,5)
        Linear head(4*2*2, 2, 8);                            // after maxpool(2,2): (4,2,2)=16 -> 2
        std::vector<Tensor> params;
        for (Module* m : {(Module*)&conv, (Module*)&head}) { auto q = m->parameters(); params.insert(params.end(), q.begin(), q.end()); }
        Adam opt(params, 0.02);
        auto fwd = [&](const Eigen::VectorXd& im) {
            Tensor x = Tensor::from({1,5,5}, im, false);
            Tensor h = maxPool2d(relu(conv.forward(x)), 4, 5, 5, 2, 2);   // (4,2,2)
            return head.forward(reshape(h, {1, 16}));                     // (1,2) logits
        };
        double loss0 = 0, loss = 0;
        for (int e = 0; e < 200; e++) {
            double el = 0;
            for (int s = 0; s < NS; s++) { opt.zeroGrad(); Tensor L = crossEntropyLoss(fwd(imgs[s]), {labels[s]}); L.backward(); opt.step(); el += L.item(); }
            loss = el / NS; if (e == 0) loss0 = loss;
        }
        int correct = 0;
        for (int s = 0; s < NS; s++) { Tensor lg = fwd(imgs[s]); int am = lg.value()(0) > lg.value()(1) ? 0 : 1; if (am == labels[s]) correct++; }
        std::printf("  loss %.3f -> %.4f   accuracy = %d/%d\n", loss0, loss, correct, NS);
        std::printf("  [%s] conv classification loss more than halves\n", loss < loss0 * 0.5 ? "PASS" : "FAIL"); if (!(loss < loss0 * 0.5)) fails++;
        std::printf("  [%s] CNN reaches 100%% accuracy\n", correct == NS ? "PASS" : "FAIL"); if (correct != NS) fails++;
    }

    std::printf("\n== it REMEMBERS: an LSTM recalls the first input across a sequence (BPTT) ==\n");
    {
        std::mt19937_64 g(99); std::normal_distribution<double> nd(0.0, 1.0);
        const int T = 6, NS = 24;
        std::vector<Eigen::VectorXd> seqs; std::vector<int> labels;
        for (int s = 0; s < NS; s++) { Eigen::VectorXd seq(T); for (int t = 0; t < T; t++) seq(t) = nd(g);
            seqs.push_back(seq); labels.push_back(seq(0) > 0 ? 1 : 0); }                  // label = sign of x[0]
        LSTM lstm(1, 12, 5); Linear head(12, 2, 6);
        std::vector<Tensor> params;
        for (Module* m : {(Module*)&lstm, (Module*)&head}) { auto q = m->parameters(); params.insert(params.end(), q.begin(), q.end()); }
        Adam opt(params, 0.03);
        auto fwd = [&](const Eigen::VectorXd& seq) {
            Tensor x = Tensor::from({T, 1}, seq, false);
            Tensor last = sliceRows(lstm.forward(x), T - 1, T);       // (1,12) final hidden state
            return head.forward(last);                               // (1,2)
        };
        double loss0 = 0, loss = 0;
        for (int e = 0; e < 150; e++) {
            double el = 0;
            for (int s = 0; s < NS; s++) { opt.zeroGrad(); Tensor L = crossEntropyLoss(fwd(seqs[s]), {labels[s]}); L.backward(); opt.step(); el += L.item(); }
            loss = el / NS; if (e == 0) loss0 = loss;
        }
        int correct = 0;
        for (int s = 0; s < NS; s++) { Tensor lg = fwd(seqs[s]); int am = lg.value()(0) > lg.value()(1) ? 0 : 1; if (am == labels[s]) correct++; }
        std::printf("  loss %.3f -> %.4f   accuracy = %d/%d (must carry x[0] across %d steps)\n", loss0, loss, correct, NS, T);
        std::printf("  [%s] LSTM recall loss decreases\n", loss < loss0 * 0.6 ? "PASS" : "FAIL"); if (!(loss < loss0 * 0.6)) fails++;
        std::printf("  [%s] LSTM recalls first input (>90%%)\n", correct >= 0.9 * NS ? "PASS" : "FAIL"); if (!(correct >= 0.9 * NS)) fails++;
    }

    std::printf("\n== it ATTENDS: a causal Transformer memorizes a sequence (next-token prediction) ==\n");
    {
        const int V = 6, dm = 16, nh = 2, dff = 32, L = 10;
        std::mt19937_64 g(7); std::uniform_int_distribution<int> tok(0, V - 1);
        std::vector<int> T(L); for (int i = 0; i < L; i++) T[i] = tok(g);
        std::vector<int> in_idx(T.begin(), T.end() - 1);    // first L-1 tokens
        std::vector<int> tgt(T.begin() + 1, T.end());       // the next token at each position
        std::vector<int> pos(L - 1); for (int i = 0; i < L - 1; i++) pos[i] = i;

        Embedding tokEmb(V, dm, 1), posEmb(L, dm, 2);
        TransformerBlock block(dm, nh, dff, /*causal=*/true, 3);
        Linear head(dm, V, 4);

        std::vector<Tensor> params;
        for (Module* m : {(Module*)&tokEmb, (Module*)&posEmb, (Module*)&block, (Module*)&head}) {
            auto q = m->parameters(); params.insert(params.end(), q.begin(), q.end());
        }
        Adam opt(params, 0.01);
        auto fwd = [&] {
            Tensor e = tokEmb.lookup(in_idx) + posEmb.lookup(pos);
            return head.forward(block.forward(e));          // (L-1, V) logits
        };
        double loss0 = 0, loss = 0;
        for (int e = 0; e < 500; e++) {
            opt.zeroGrad(); Tensor Lx = crossEntropyLoss(fwd(), tgt); Lx.backward(); opt.step();
            if (e == 0) loss0 = Lx.item(); loss = Lx.item();
        }
        Tensor logits = fwd(); Eigen::Map<RowMat> Z(logits.value().data(), L - 1, V);
        int correct = 0;
        for (int i = 0; i < L - 1; i++) { int am = 0; double best = Z(i, 0);
            for (int c = 1; c < V; c++) if (Z(i, c) > best) { best = Z(i, c); am = c; }
            if (am == tgt[i]) correct++; }
        std::printf("  loss %.3f -> %.4f   next-token accuracy = %d/%d\n", loss0, loss, correct, L - 1);
        std::printf("  [%s] LM loss collapses (<0.05): attention + causal mask + embeddings learn\n", loss < 0.05 ? "PASS" : "FAIL"); if (!(loss < 0.05)) fails++;
        std::printf("  [%s] greedily regenerates the whole sequence\n", correct == L - 1 ? "PASS" : "FAIL"); if (correct != L - 1) fails++;
    }

    std::printf("\n== training utilities (batching / metrics / early-stop / grad-clip / checkpoint) ==\n");
    {
        // accuracy on known logits (argmax rows: 0,1,0)
        Eigen::VectorXd lv(6); lv << 2,1,  0,3,  5,1;
        Tensor lg = Tensor::from({3,2}, lv, false);
        double acc = accuracy(lg, {0,1,0}), acc2 = accuracy(lg, {0,0,0});
        std::printf("  [%s] accuracy() = %.3f / %.3f (expect 1.000 / 0.667)\n",
                    (acc==1.0 && std::fabs(acc2-2.0/3)<1e-9) ? "PASS" : "FAIL", acc, acc2);
        if (!(acc==1.0 && std::fabs(acc2-2.0/3)<1e-9)) fails++;

        // binary precision/recall/F1: pred pos {0,1,4}, truth pos {0,4} -> tp=2,fp=1,fn=0
        PRF prf = binaryPRF({1,1,0,0,1}, {1,0,0,0,1}, 1);
        bool prfok = std::fabs(prf.precision-2.0/3)<1e-9 && std::fabs(prf.recall-1.0)<1e-9;
        std::printf("  [%s] binaryPRF precision=%.3f recall=%.3f f1=%.3f\n", prfok?"PASS":"FAIL", prf.precision, prf.recall, prf.f1);
        if (!prfok) fails++;

        // grad clipping to a target global norm (grad [3,0,4] has norm 5 -> clip to 1)
        std::vector<Tensor> ps = { Tensor::zeros({3}, true) };
        ps[0].grad() << 3,0,4; clipGradNorm(ps, 1.0);
        double gn = ps[0].grad().norm();
        std::printf("  [%s] clipGradNorm -> norm %.4f (expect 1.0000)\n", std::fabs(gn-1.0)<1e-9?"PASS":"FAIL", gn); if (std::fabs(gn-1.0)>=1e-9) fails++;

        // early stopping triggers after patience of no improvement
        EarlyStopping es(3); bool stopped = false;
        for (double l : {1.0, 0.9, 0.95, 0.96, 0.97, 0.98}) if (es.step(l)) { stopped = true; break; }
        std::printf("  [%s] EarlyStopping fires after patience\n", stopped?"PASS":"FAIL"); if (!stopped) fails++;

        // LR schedules: cosine endpoints + step decay
        bool lrok = std::fabs(lrCosine(0.1,0,10)-0.1)<1e-12 && lrCosine(0.1,10,10)<1e-9 && std::fabs(lrStep(0.1,7,3,0.1)-0.001)<1e-12;
        std::printf("  [%s] LR schedules (cosine base->0, step decay)\n", lrok?"PASS":"FAIL"); if (!lrok) fails++;

        // DataLoader covers every sample exactly once per epoch
        DataLoader dl(10, 4, true, 1); auto bs = dl.epoch();
        int total = 0; std::vector<int> seen(10, 0); for (auto& b : bs) { total += (int)b.size(); for (int i : b) seen[i]++; }
        bool dlok = (total == 10); for (int s : seen) if (s != 1) dlok = false;
        std::printf("  [%s] DataLoader covers all 10 samples once (%zu batches)\n", dlok?"PASS":"FAIL", bs.size()); if (!dlok) fails++;

        // checkpoint: save -> scramble -> load must restore weights EXACTLY
        djehuti::nn::Sequential net;
        net.add(std::make_shared<Linear>(3,4,1)); net.add(std::make_shared<ReLU>()); net.add(std::make_shared<Linear>(4,2,2));
        auto params = net.parameters();
        std::vector<Eigen::VectorXd> snap; for (auto& p : params) snap.push_back(p.value());
        saveParams(params, "/tmp/dj_nn_ckpt.bin");
        for (auto& p : params) p.value().setZero();                 // scramble
        bool loaded = loadParams(params, "/tmp/dj_nn_ckpt.bin");
        double maxdiff = 0; for (size_t i = 0; i < params.size(); i++) maxdiff = std::max(maxdiff, (params[i].value()-snap[i]).cwiseAbs().maxCoeff());
        std::printf("  [%s] checkpoint save->scramble->load restores weights exactly (max|Δ|=%.1e)\n",
                    (loaded && maxdiff==0.0) ? "PASS" : "FAIL", maxdiff);
        if (!(loaded && maxdiff==0.0)) fails++;
    }

    std::printf("\n== PCA / SVD (dimensionality reduction; anomaly-reconstruction baseline) ==\n");
    {
        std::mt19937_64 g(11); std::normal_distribution<double> nd(0.0, 1.0), noise(0.0, 0.05);
        const int N = 200; Eigen::MatrixXd X(N, 2);
        Eigen::Vector2d dir(1.0, 2.0); dir.normalize();
        for (int i = 0; i < N; i++) { double t = nd(g) * 3.0; X(i,0) = t*dir(0) + noise(g); X(i,1) = t*dir(1) + noise(g); }
        PCA pca; pca.fit(X, 2);
        double align = std::fabs(pca.components.col(0).dot(dir));
        double evr = pca.explained_variance(0) / (pca.explained_variance(0) + pca.explained_variance(1));
        double recerr = (pca.reconstruct(X) - X).cwiseAbs().maxCoeff();
        double orth = (pca.components.transpose()*pca.components - Eigen::MatrixXd::Identity(2,2)).cwiseAbs().maxCoeff();
        std::printf("  PC1·dir=%.4f  expl-var-ratio(PC1)=%.4f  full-recon max|Δ|=%.1e  orthonormality=%.1e\n", align, evr, recerr, orth);
        std::printf("  [%s] PC1 aligns with the true data axis (|cos|>0.99)\n", align>0.99?"PASS":"FAIL"); if (align<=0.99) fails++;
        std::printf("  [%s] PC1 explains >95%% of variance (recovers the 1-D structure)\n", evr>0.95?"PASS":"FAIL"); if (evr<=0.95) fails++;
        std::printf("  [%s] full (k=D) reconstruction is exact\n", recerr<1e-9?"PASS":"FAIL"); if (recerr>=1e-9) fails++;
        std::printf("  [%s] principal components orthonormal\n", orth<1e-9?"PASS":"FAIL"); if (orth>=1e-9) fails++;
    }

    std::printf("\n== it FLAGS ANOMALIES: an autoencoder scores off-manifold points high ==\n");
    {
        std::mt19937_64 g(33); std::normal_distribution<double> nd(0.0, 1.0);
        const int Nn = 120, Na = 30;
        Eigen::MatrixXd Xn(Nn, 2), Xa(Na, 2);
        for (int i = 0; i < Nn; i++) { double t = nd(g); Xn(i,0) = t; Xn(i,1) = 0.5*t + 0.05*nd(g); }  // ~1-D manifold (normal)
        for (int i = 0; i < Na; i++) { Xa(i,0) = nd(g); Xa(i,1) = nd(g) + 3.0; }                         // off-manifold (anomalies)
        std::vector<int> idn(Nn), ida(Na); for (int i = 0; i < Nn; i++) idn[i] = i; for (int i = 0; i < Na; i++) ida[i] = i;
        Tensor Xnt = batchRows(Xn, idn), Xat = batchRows(Xa, ida);

        Autoencoder ae(2, {8}, 1, 7);
        Adam opt(ae.parameters(), 0.01);
        double loss0 = 0, loss = 0;
        for (int e = 0; e < 400; e++) { opt.zeroGrad(); Tensor L = mseLoss(ae.forward(Xnt), Xnt); L.backward(); opt.step(); if (e == 0) loss0 = L.item(); loss = L.item(); }

        Eigen::VectorXd en = reconstructionError(ae.forward(Xnt), Xnt), ea = reconstructionError(ae.forward(Xat), Xat);
        std::vector<double> ens(en.data(), en.data() + Nn); std::sort(ens.begin(), ens.end());
        double thr = ens[(int)(0.95 * Nn)];
        int flagged = 0; for (int i = 0; i < Na; i++) if (ea(i) > thr) flagged++;
        double detect = (double)flagged / Na;
        std::printf("  recon loss %.4f -> %.4f   mean score normal=%.4f anomaly=%.4f   detected %d/%d at 95%%-normal thresh\n",
                    loss0, loss, en.mean(), ea.mean(), flagged, Na);
        std::printf("  [%s] anomalies score >>5x normal (off-manifold reconstructs poorly)\n", ea.mean() > 5*en.mean() ? "PASS" : "FAIL"); if (!(ea.mean() > 5*en.mean())) fails++;
        std::printf("  [%s] >80%% of anomalies flagged at the 95th-pct-normal threshold\n", detect > 0.8 ? "PASS" : "FAIL"); if (detect <= 0.8) fails++;
    }

    std::printf("\n== VAE gradient-check (reparameterization + KL, fixed noise -> deterministic) ==\n");
    {
        VAE vae(3, 5, 2, 9);
        Tensor x   = Tensor::randn({4,3}, 1.0, 70, false);
        Tensor eps = Tensor::randn({4,2}, 1.0, 71, false);            // FIXED noise
        gradCheck("VAE (recon+KL)", vae.parameters(), [&]{ auto o = vae.run(x, eps); return mseLoss(o.recon, x) + o.kl; }, 1e-5, 1e-4);
    }

    std::printf("\n== it GENERATES: a VAE trains (reconstruction + KL) on normal data ==\n");
    {
        std::mt19937_64 g(44); std::normal_distribution<double> nd(0.0, 1.0);
        const int Nn = 120; Eigen::MatrixXd Xn(Nn, 2);
        for (int i = 0; i < Nn; i++) { double t = nd(g); Xn(i,0) = t; Xn(i,1) = 0.5*t + 0.05*nd(g); }
        std::vector<int> idn(Nn); for (int i = 0; i < Nn; i++) idn[i] = i;
        Tensor Xnt = batchRows(Xn, idn);
        VAE vae(2, 8, 2, 11); Adam opt(vae.parameters(), 0.005);
        std::mt19937_64 ge(5); std::normal_distribution<double> en(0.0, 1.0);
        double l0 = 0, l = 0, klv = 0;
        for (int e = 0; e < 400; e++) {
            Eigen::VectorXd ev(Nn*2); for (int i = 0; i < Nn*2; i++) ev(i) = en(ge);   // fresh noise each step
            Tensor eps = Tensor::from({Nn, 2}, ev, false);
            opt.zeroGrad();
            auto o = vae.run(Xnt, eps);
            Tensor loss = mseLoss(o.recon, Xnt) + o.kl * (0.01 / Nn);   // small-beta per-sample KL
            loss.backward(); opt.step();
            if (e == 0) l0 = loss.item(); l = loss.item(); klv = o.kl.item();
        }
        std::printf("  total loss %.4f -> %.4f   final KL=%.3f\n", l0, l, klv);
        std::printf("  [%s] VAE training loss decreases (recon improves)\n", l < l0*0.7 ? "PASS" : "FAIL"); if (!(l < l0*0.7)) fails++;
        std::printf("  [%s] KL stays finite, >=0 and bounded (regularizer behaves)\n", (std::isfinite(klv) && klv > -1e-6 && klv < 1e4) ? "PASS" : "FAIL");
        if (!(std::isfinite(klv) && klv > -1e-6 && klv < 1e4)) fails++;
    }

    std::printf("\n== it ISOLATES: an Isolation Forest scores anomalies higher (classical, no training) ==\n");
    {
        std::mt19937_64 g(55); std::normal_distribution<double> nd(0.0, 1.0);
        const int Nn = 160, Na = 40; Eigen::MatrixXd Xn(Nn, 2), Xa(Na, 2);
        for (int i = 0; i < Nn; i++) { Xn(i,0) = nd(g)*0.5;       Xn(i,1) = nd(g)*0.5; }        // tight normal cluster
        for (int i = 0; i < Na; i++) { Xa(i,0) = nd(g)*0.5 + 4.0; Xa(i,1) = nd(g)*0.5 + 4.0; }  // far-away anomalies
        IsolationForest iforest(150, 128, 1); iforest.fit(Xn);
        Eigen::VectorXd sn = iforest.scores(Xn), sa = iforest.scores(Xa);
        std::vector<double> sns(sn.data(), sn.data() + Nn); std::sort(sns.begin(), sns.end());
        double thr = sns[(int)(0.95 * Nn)];
        int flagged = 0; for (int i = 0; i < Na; i++) if (sa(i) > thr) flagged++;
        std::printf("  mean score normal=%.4f anomaly=%.4f   detected %d/%d at 95%%-normal thresh\n", sn.mean(), sa.mean(), flagged, Na);
        std::printf("  [%s] anomalies score higher than normal (margin > 0.1)\n", sa.mean() > sn.mean()+0.1 ? "PASS" : "FAIL"); if (!(sa.mean() > sn.mean()+0.1)) fails++;
        std::printf("  [%s] >80%% of anomalies flagged\n", flagged > 0.8*Na ? "PASS" : "FAIL"); if (!(flagged > 0.8*Na)) fails++;
    }

    // =====================================================================
    // Reinforcement learning.  GridWorld is a deterministic, fully-known MDP, so the
    // OPTIMAL return is computed exactly by value iteration on the model — every agent's
    // anchor is "reaches that known optimum" (no magic constant).  CartPole is the classic
    // continuous-control benchmark; the anchor is "learns to vastly beat a random policy."
    // =====================================================================
    std::printf("\n== it PLANS (tabular RL): Q-learning / SARSA / Double-Q reach the EXACT optimal on GridWorld ==\n");
    {
        rl::GridWorld env(4, 4);                                   // start (0,0) -> goal (3,3)
        double opt = rl::gridworldOptimalReturn(env, 1.0);        // exact: −shortest-path = −6
        std::printf("  value-iteration optimal start-return = %.1f\n", opt);

        auto greedyReturnTab = [](rl::GridWorld env, auto greedyFn) {
            env.reset(); double ret = 0; for (int t = 0; t < 100; t++) {
                int s = env.stateIndex(); auto so = env.step(greedyFn(s)); ret += so.reward; if (so.done) break; }
            return ret;
        };
        // Q-learning
        rl::QLearning ql(env.nStates(), env.nActions(), 0.5, 1.0, 7);
        for (int ep = 0; ep < 3000; ep++) { env.reset(); double eps = std::max(0.05, 1.0 - ep / 1500.0);
            for (int t = 0; t < 100; t++) { int s = env.stateIndex(); int a = ql.epsGreedy(s, eps);
                auto so = env.step(a); ql.update(s, a, so.reward, env.stateIndex(), so.done); if (so.done) break; } }
        double rQ = greedyReturnTab(env, [&](int s){ return ql.greedy(s); });
        std::printf("  [%s] Q-learning greedy return = %.1f (== optimal)\n", std::fabs(rQ - opt) < 1e-9 ? "PASS" : "FAIL", rQ);
        if (!(std::fabs(rQ - opt) < 1e-9)) fails++;
        // SARSA
        rl::SARSA sa(env.nStates(), env.nActions(), 0.5, 1.0, 8);
        for (int ep = 0; ep < 3000; ep++) { env.reset(); double eps = std::max(0.05, 1.0 - ep / 1500.0);
            int s = env.stateIndex(), a = sa.epsGreedy(s, eps);
            for (int t = 0; t < 100; t++) { auto so = env.step(a); int sp = env.stateIndex();
                int ap = sa.epsGreedy(sp, eps); sa.update(s, a, so.reward, sp, ap, so.done); s = sp; a = ap; if (so.done) break; } }
        double rS = greedyReturnTab(env, [&](int s){ return sa.greedy(s); });
        std::printf("  [%s] SARSA greedy return = %.1f (>= optimal−2)\n", rS >= opt - 2.0 + 1e-9 ? "PASS" : "FAIL", rS);
        if (!(rS >= opt - 2.0 + 1e-9)) fails++;
        // Double-Q
        rl::DoubleQLearning dq(env.nStates(), env.nActions(), 0.5, 1.0, 9);
        for (int ep = 0; ep < 3000; ep++) { env.reset(); double eps = std::max(0.05, 1.0 - ep / 1500.0);
            for (int t = 0; t < 100; t++) { int s = env.stateIndex(); int a = dq.epsGreedy(s, eps);
                auto so = env.step(a); dq.update(s, a, so.reward, env.stateIndex(), so.done); if (so.done) break; } }
        double rD = greedyReturnTab(env, [&](int s){ return dq.greedy(s); });
        std::printf("  [%s] Double-Q greedy return = %.1f (== optimal)\n", std::fabs(rD - opt) < 1e-9 ? "PASS" : "FAIL", rD);
        if (!(std::fabs(rD - opt) < 1e-9)) fails++;
    }

    std::printf("\n== it APPROXIMATES (DQN): an MLP Q-network + replay + target net solves GridWorld ==\n");
    {
        rl::GridWorld env(4, 4);
        double opt = rl::gridworldOptimalReturn(env, 1.0);                  // −6
        rl::DQNAgent agent(env.obsDim(), env.nActions(), {64}, 5e-3, 0.95, 20000, true, 3);
        int gstep = 0;
        for (int ep = 0; ep < 600; ep++) {
            Eigen::VectorXd s = env.reset(); double eps = std::max(0.05, 1.0 - ep / 250.0);
            for (int t = 0; t < 100; t++) {
                int a = agent.act(s, eps); auto so = env.step(a);
                agent.remember({ s, a, so.reward, so.obs, so.done });
                agent.trainStep(32); if (++gstep % 100 == 0) agent.syncTarget();
                s = so.obs; if (so.done) break;
            }
        }
        Eigen::VectorXd s = env.reset(); double ret = 0; bool reached = false;
        for (int t = 0; t < 100; t++) { int a = agent.act(s, 0.0); auto so = env.step(a); ret += so.reward;
            s = so.obs; if (so.done) { reached = (env.stateIndex() == env.goal); break; } }
        std::printf("  DQN greedy return = %.1f (optimal %.1f), reached goal=%d\n", ret, opt, (int)reached);
        bool okdqn = reached && ret >= opt - 2.0 + 1e-9;
        std::printf("  [%s] DQN reaches the goal near-optimally\n", okdqn ? "PASS" : "FAIL"); if (!okdqn) fails++;
    }

    std::printf("\n== it IMPROVES ITS POLICY (REINFORCE): a softmax policy network learns GridWorld ==\n");
    {
        rl::GridWorld env(4, 4);
        double opt = rl::gridworldOptimalReturn(env, 1.0);
        rl::ReinforceAgent agent(env.obsDim(), env.nActions(), {64}, 1e-2, 0.99, true, 4);
        for (int ep = 0; ep < 1500; ep++) {
            Eigen::VectorXd s = env.reset();
            std::vector<Eigen::VectorXd> S; std::vector<int> A; std::vector<double> R;
            for (int t = 0; t < 100; t++) { int a = agent.sample(s); auto so = env.step(a);
                S.push_back(s); A.push_back(a); R.push_back(so.reward); s = so.obs; if (so.done) break; }
            agent.update(S, A, R);
        }
        Eigen::VectorXd s = env.reset(); double ret = 0; bool reached = false;
        for (int t = 0; t < 100; t++) { Eigen::VectorXd p = agent.probs(s); int a; p.maxCoeff(&a);
            auto so = env.step(a); ret += so.reward; s = so.obs; if (so.done) { reached = (env.stateIndex() == env.goal); break; } }
        std::printf("  REINFORCE greedy return = %.1f (optimal %.1f), reached goal=%d\n", ret, opt, (int)reached);
        bool okpg = reached && ret >= opt - 3.0 + 1e-9;
        std::printf("  [%s] REINFORCE reaches the goal near-optimally\n", okpg ? "PASS" : "FAIL"); if (!okpg) fails++;
    }

    std::printf("\n== it CONTROLS (deep RL on CartPole): DQN and PPO vastly beat a random policy ==\n");
    {
        // random-policy baseline (fixed seed) for an honest reference point
        auto randReturn = [](uint64_t seed) {
            rl::CartPole env(200, seed); std::mt19937_64 g(seed + 99); double tot = 0; int EP = 20;
            for (int e = 0; e < EP; e++) { env.reset(); for (int t = 0; t < 200; t++) {
                int a = std::uniform_int_distribution<int>(0, 1)(g); auto so = env.step(a); tot += so.reward; if (so.done) break; } }
            return tot / EP;
        };
        double base = randReturn(123);

        // ---- DQN ----
        rl::DQNAgent dqn(4, 2, {128}, 1e-3, 0.99, 50000, true, 11);
        rl::CartPole env(200, 11);
        int gstep = 0;
        for (int ep = 0; ep < 350; ep++) {
            Eigen::VectorXd s = env.reset(); double eps = std::max(0.05, 1.0 - ep / 150.0);
            for (int t = 0; t < 200; t++) { int a = dqn.act(s, eps); auto so = env.step(a);
                dqn.remember({ s, a, so.reward, so.obs, so.done }); dqn.trainStep(64);
                if (++gstep % 200 == 0) dqn.syncTarget(); s = so.obs; if (so.done) break; }
        }
        auto evalDQN = [&](uint64_t seed) { rl::CartPole e(200, seed); double tot = 0; int EP = 10;
            for (int k = 0; k < EP; k++) { Eigen::VectorXd s = e.reset(); for (int t = 0; t < 200; t++) {
                int a = dqn.act(s, 0.0); auto so = e.step(a); tot += so.reward; s = so.obs; if (so.done) break; } }
            return tot / EP; };
        double rDQN = evalDQN(777);
        std::printf("  random baseline = %.1f   DQN eval return = %.1f\n", base, rDQN);
        bool okDQN = rDQN >= 120.0 && rDQN >= 3.0 * base;
        std::printf("  [%s] DQN solves CartPole (>=120 and >=3x random)\n", okDQN ? "PASS" : "FAIL"); if (!okDQN) fails++;

        // ---- PPO (actor-critic + GAE + clipped surrogate) ----
        rl::PPOAgent ppo(4, 2, {64}, 3e-3, 0.99, 0.95, 0.2, 0.01, 6, 21);
        for (int it = 0; it < 60; it++) {
            std::vector<Eigen::VectorXd> S; std::vector<int> A; std::vector<double> R; std::vector<char> D;
            rl::CartPole cenv(200, 21 + it);
            int collected = 0;
            while (collected < 1024) { Eigen::VectorXd s = cenv.reset();
                for (int t = 0; t < 200; t++) { int a = ppo.sample(s); auto so = cenv.step(a);
                    S.push_back(s); A.push_back(a); R.push_back(so.reward); D.push_back(so.done ? 1 : 0);
                    s = so.obs; collected++; if (so.done) break; } }
            ppo.update(S, A, R, D);
        }
        auto evalPPO = [&](uint64_t seed) { rl::CartPole e(200, seed); double tot = 0; int EP = 10;
            for (int k = 0; k < EP; k++) { Eigen::VectorXd s = e.reset(); for (int t = 0; t < 200; t++) {
                Eigen::VectorXd p = ppo.probs(s); int a; p.maxCoeff(&a); auto so = e.step(a); tot += so.reward; s = so.obs; if (so.done) break; } }
            return tot / EP; };
        double rPPO = evalPPO(888);
        std::printf("  PPO eval return = %.1f\n", rPPO);
        bool okPPO = rPPO >= 80.0 && rPPO >= 3.0 * base;
        std::printf("  [%s] PPO solves CartPole (>=80 and >=3x random)\n", okPPO ? "PASS" : "FAIL"); if (!okPPO) fails++;
    }

    std::printf("\n%s  (%d failure%s)\n", fails ? "SOME FAILURES" : "ALL PASS", fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
