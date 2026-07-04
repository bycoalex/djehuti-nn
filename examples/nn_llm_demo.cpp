// =====================================================================
// nn_llm_demo.cpp — the SELF-HOSTED FLAGSHIP (#34, build-order step 11).
//
// A char-level GPT-style Transformer language model that TRAINS and GENERATES
// text with ZERO external dependencies — pure C++17 + Eigen, the same
// gradient-checked autograd engine used by every other piece of the suite.
// This is the literal "self-hosted AI" pitch: no Python, no cloud, no GPU
// required; clone, compile, run.  The whole thing is one header.
//
// It also self-checks (exit 0 == all green), so verify.sh can gate it like the
// recovery suite:
//   1. training next-token loss collapses (the model learns the text),
//   2. train -> save -> scramble -> load restores the weights EXACTLY,
//   3. greedy continuation of a seen prefix reproduces the corpus (it learned),
//   4. temperature sampling produces only valid in-vocab characters.
//
// build (NaN-honest): g++-15 -std=c++17 -O3 -march=native -fopenmp -I. -I<eigen> \
//                     examples/nn_llm_demo.cpp -o build/nn_llm_demo -lpthread -lm
//
// Copyright © Co Bros & Alex Ramanaidou (Webuildcodes). All rights reserved.
// Licensed under the Non‑Exclusive Source License – see LICENSE file.
//
// =====================================================================
#include "djehuti_nn.hpp"
#include <cstdio>
#include <string>
#include <vector>
#include <unordered_map>
#include <random>

using namespace djehuti::nn;

int main() {
    std::printf("Djehuti SDK — self-hosted char-level Transformer LM (zero deps)\n\n");
    int fails = 0;

    // ---- corpus: small, structured, recognizable (doubles as the demo's voice) ----
    const std::string text =
        "djehuti is a self-hosted engine. it learns from data and writes plain reports. "
        "no cloud, no python, just c++ and math. the lab runs offline and the numbers do not lie.\n";

    // ---- char-level tokenizer (build vocab from the unique characters) ----
    std::vector<char> id2ch; std::unordered_map<char,int> ch2id;
    for (char c : text) if (!ch2id.count(c)) { ch2id[c] = (int)id2ch.size(); id2ch.push_back(c); }
    const int vocab = (int)id2ch.size();
    std::vector<int> data; data.reserve(text.size());
    for (char c : text) data.push_back(ch2id[c]);
    const int T = (int)data.size();
    std::printf("  corpus: %d chars, vocab = %d unique chars\n", T, vocab);

    // ---- model: a small but real causal Transformer (2 blocks, 4 heads, d=64) ----
    const int d_model = 64, n_layers = 2, n_heads = 4, d_ff = 128, max_seq = T;
    GPT model(vocab, d_model, n_layers, n_heads, d_ff, max_seq, 7);
    auto params = model.parameters();
    long nparam = 0; for (auto& p : params) nparam += p.size();
    std::printf("  model: %d layers, %d heads, d_model=%d, d_ff=%d  (%ld parameters)\n\n",
                n_layers, n_heads, d_model, d_ff, nparam);

    // teacher-forcing next-token targets: predict data[t+1] from data[0..t]
    std::vector<int> inp(data.begin(), data.end() - 1);     // length T-1
    std::vector<int> tgt(data.begin() + 1, data.end());     // length T-1

    // ---- train ----
    Adam opt(params, 3e-3);
    double loss0 = 0, loss = 0;
    const int STEPS = 150;
    std::printf("== training (next-token cross-entropy) ==\n");
    for (int step = 0; step < STEPS; step++) {
        opt.zeroGrad();
        Tensor logits = model.forwardIds(inp);              // (T-1, vocab)
        Tensor L = crossEntropyLoss(logits, tgt);
        L.backward();
        clipGradNorm(params, 1.0);
        opt.step();
        loss = L.item(); if (step == 0) loss0 = loss;
        if (step % 40 == 0 || step == STEPS - 1)
            std::printf("  step %3d   loss = %.4f\n", step, loss);
    }
    std::printf("  loss %.4f -> %.4f\n", loss0, loss);
    bool okLoss = loss < 0.15 && loss < loss0 * 0.2;
    std::printf("  [%s] training loss collapses (model learns the text)\n", okLoss ? "PASS" : "FAIL");
    if (!okLoss) fails++;

    // ---- train -> save -> scramble -> load restores weights EXACTLY ----
    std::printf("\n== self-hosted persistence: save -> load round-trip ==\n");
    saveParams(params, "/tmp/dj_llm.bin");
    std::vector<Eigen::VectorXd> before; for (auto& p : params) before.push_back(p.p->v);
    for (auto& p : params) p.p->v.setZero();                // scramble
    bool loaded = loadParams(params, "/tmp/dj_llm.bin");
    double maxd = 0; for (size_t i = 0; i < params.size(); i++) maxd = std::max(maxd, (params[i].p->v - before[i]).cwiseAbs().maxCoeff());
    bool okIO = loaded && maxd == 0.0;
    std::printf("  [%s] checkpoint restores weights exactly (max|Δ|=%.1e)\n", okIO ? "PASS" : "FAIL", maxd);
    if (!okIO) fails++;

    // ---- greedy continuation of a seen prefix reproduces the corpus (it LEARNED) ----
    std::printf("\n== greedy generation from a 12-char prompt (memorization check) ==\n");
    int promptLen = 12;
    std::vector<int> prompt(data.begin(), data.begin() + promptLen);
    std::mt19937_64 rng(123);
    std::vector<int> gen = gptGenerate(model, prompt, T - promptLen, 1.0, rng, /*greedy=*/true);
    int correct = 0; for (int i = 0; i < (int)gen.size(); i++) if (gen[i] == data[promptLen + i]) correct++;
    std::string shown; for (int id : prompt) shown += id2ch[id]; for (int id : gen) shown += id2ch[id];
    std::printf("  generated: \"%s\"\n", shown.c_str());
    double frac = gen.empty() ? 0.0 : (double)correct / gen.size();
    bool okGen = frac > 0.95;
    std::printf("  [%s] greedy continuation matches the corpus (%.1f%% of %d chars)\n",
                okGen ? "PASS" : "FAIL", 100.0 * frac, (int)gen.size());
    if (!okGen) fails++;

    // ---- temperature sampling: produces only valid in-vocab characters ----
    std::printf("\n== temperature sampling (creative generation) ==\n");
    std::mt19937_64 rng2(999);
    std::vector<int> sample = gptGenerate(model, prompt, 80, 0.6, rng2, /*greedy=*/false);
    std::string ssh; for (int id : prompt) ssh += id2ch[id]; for (int id : sample) ssh += id2ch[id];
    bool okVocab = true; for (int id : sample) if (id < 0 || id >= vocab) okVocab = false;
    std::printf("  sampled (T=0.6): \"%s\"\n", ssh.c_str());
    std::printf("  [%s] all sampled tokens are valid in-vocab characters\n", okVocab ? "PASS" : "FAIL");
    if (!okVocab) fails++;

    std::printf("\n%s  (%d failure%s)\n", fails ? "SOME FAILURES" : "ALL PASS", fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
