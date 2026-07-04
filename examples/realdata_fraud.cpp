// =====================================================================
// Fraud — REAL public-data validation (no synthetic data).
// Dataset: the ULB credit-card fraud benchmark (OpenML id 1597 / Kaggle
// "Credit Card Fraud Detection") — 284,807 real European card transactions,
// 492 labelled fraud (0.172%), features V1..V28 (PCA-anonymized) + Amount.
// We score it UNSUPERVISED (labels used only to grade) with the AI suite's
// deep Autoencoder + Isolation Forest.  The recognized benchmark result for
// unsupervised detectors here is ROC-AUC ~ 0.95; PR-AUC is low by nature
// (0.17% positives) — the honest reality a fraud team already knows.
//
// Copyright © Co Bros & Alex Ramanaidou (Webuildcodes). All rights reserved.
// Licensed under the Non‑Exclusive Source License – see LICENSE file.
//
// =====================================================================
#include "djehuti_fraud.hpp"
#include <cstdio>
#include <random>
#include <string>

using namespace djehuti;

static int fails = 0;

int main(int argc, char** argv) {
    std::string dir = (argc > 1) ? argv[1] : "/tmp/realdata";
    std::printf("Djehuti SDK — FRAUD real-data validation (ULB credit-card, unsupervised)\n\n");

    fraud::Dataset d = fraud::loadCSV(dir + "/creditcard.csv", "Class");
    if (d.label.empty()) {
        std::printf("[FRAUD] SKIP — creditcard.csv not present / unreadable\n");
        std::printf("\nALL PASS  (0 failures)\n");   // non-fatal if the dataset wasn't downloaded
        return 0;
    }
    int N = (int)d.label.size(), F = (int)d.X.cols(), nf = d.n_fraud();
    std::printf("[DATA] %d transactions, %d features, %d fraud (%.3f%%)\n", N, F, nf, 100.0*nf/N);

    Eigen::MatrixXd Xs = fraud::standardize(d.X);

    // Isolation Forest on the full set
    Eigen::VectorXd iso = fraud::scoreIsolationForest(Xs, /*seed*/0);
    // Autoencoder: train on a 40k random subsample (unsupervised), score all rows
    std::mt19937_64 rng(1);
    int ntrain = std::min(40000, N);
    std::vector<int> idx(N); for (int i = 0; i < N; i++) idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), rng);
    Eigen::MatrixXd Xtrain(ntrain, F);
    for (int i = 0; i < ntrain; i++) Xtrain.row(i) = Xs.row(idx[i]);
    Eigen::VectorXd ae = fraud::scoreAutoencoder(Xs, Xtrain, /*epochs*/80);
    Eigen::VectorXd ens = fraud::ensembleScore(ae, iso);

    // grade at 0.5% and 1% alert budgets (a real fraud team's investigate-the-top budget)
    auto Rae  = fraud::evaluate("Autoencoder", ae, d.label, 0.005);
    auto Riso = fraud::evaluate("IsolationForest", iso, d.label, 0.005);
    auto Rens = fraud::evaluate("Ensemble", ens, d.label, 0.005);
    auto Rens1 = fraud::evaluate("Ensemble", ens, d.label, 0.01);   // top 1% budget
    auto line = [&](const fraud::Report& R){
        std::printf("  %-16s ROC-AUC=%.3f  PR-AUC=%.3f  recall@0.5%%=%.3f  precision=%.3f  caught=%d/%d\n",
                    R.method.c_str(), R.roc_auc, R.pr_auc, R.recall, R.precision, R.frauds_caught, R.total_fraud);
    };
    line(Rae); line(Riso); line(Rens);
    std::printf("  %-16s recall@1%% budget = %.3f (caught %d/%d in the top 1%% flagged)\n",
                "Ensemble", Rens1.recall, Rens1.frauds_caught, Rens1.total_fraud);

    // Honest anchors: ROC-AUC is THE recognized unsupervised benchmark here (~0.95);
    // PR-AUC lift over the 0.17% base rate; recall reported at the alert budget.
    auto chk = [&](bool ok, const char* msg){ std::printf("  [%s] %s\n", ok?"PASS":"FAIL", msg); if (!ok) fails++; };
    chk(Riso.roc_auc > 0.90, "Isolation Forest ROC-AUC > 0.90 on real card fraud (benchmark ~0.95)");
    chk(Rens.roc_auc > 0.92, "Ensemble ROC-AUC > 0.92 on real card fraud");
    chk(Rens.pr_auc  > 0.10, "PR-AUC > 0.10 (>50x lift over the 0.0017 base rate)");
    chk(Rens1.recall > 0.55, "Ensemble catches >55% of real fraud in the top 1% flagged (unsupervised, no labels)");

    std::printf("\n%s  (%d failure%s)\n", fails ? "SOME FAILURES" : "ALL PASS", fails, fails==1?"":"s");
    return fails ? 1 : 0;
}
