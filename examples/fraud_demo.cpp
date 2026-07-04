// =====================================================================
// fraud_demo — unsupervised transaction-fraud detection on the AI suite.
// Generates a labeled fraud set (correlated legit + injected fraud), scores
// it with a deep Autoencoder and an Isolation Forest (NO labels used for
// training), and reports the fraud-team metrics: precision / recall / F1 /
// PR-AUC / ROC-AUC and "% of fraud caught".  Self-checks (>=80% recall at
// strong precision).  Optional: pass a real fraud CSV + label column to run
// the same pipeline on real public data.
//   ./fraud_demo [real_fraud.csv Class]
//
// Copyright © Co Bros & Alex Ramanaidou (Webuildcodes). All rights reserved.
// Licensed under the Non‑Exclusive Source License – see LICENSE file.
//
// =====================================================================
#include "djehuti_fraud.hpp"
#include <cstdio>

using namespace djehuti;

static int g_fail = 0;
static void CHECK(bool ok, const char* name, double got = 0, double want = 0) {
    std::printf("  [%s] %s", ok ? "PASS" : "FAIL", name);
    if (got != 0 || want != 0) std::printf("   (got %.4f, want %.4f)", got, want);
    std::printf("\n");
    if (!ok) g_fail++;
}

static void report(const fraud::Report& R) {
    std::printf("  %-16s  precision=%.3f  recall=%.3f  F1=%.3f  PR-AUC=%.3f  ROC-AUC=%.3f  caught=%d/%d (%.0f%%)\n",
                R.method.c_str(), R.precision, R.recall, R.f1, R.pr_auc, R.roc_auc,
                R.frauds_caught, R.total_fraud, R.pct_caught);
}

static void runPipeline(const fraud::Dataset& d, double alert_rate, bool assert_anchor) {
    Eigen::MatrixXd Xs = fraud::standardize(d.X);
    Eigen::VectorXd ae  = fraud::scoreAutoencoder(Xs);
    Eigen::VectorXd iso = fraud::scoreIsolationForest(Xs);
    Eigen::VectorXd ens = fraud::ensembleScore(ae, iso);
    auto Rae  = fraud::evaluate("Autoencoder", ae, d.label, alert_rate);
    auto Riso = fraud::evaluate("IsolationForest", iso, d.label, alert_rate);
    auto Rens = fraud::evaluate("Ensemble", ens, d.label, alert_rate);
    report(Rae); report(Riso); report(Rens);
    if (assert_anchor) {
        // PR-AUC / ROC-AUC are the threshold-free quality metrics; recall depends on
        // the alert budget, so anchor recall on the deployed ENSEMBLE.
        CHECK(Rae.pr_auc  >= 0.80, "Autoencoder PR-AUC >= 0.80", Rae.pr_auc, 0.80);
        CHECK(Riso.pr_auc >= 0.80, "Isolation Forest PR-AUC >= 0.80", Riso.pr_auc, 0.80);
        CHECK(Rens.pr_auc >= 0.85, "Ensemble PR-AUC >= 0.85", Rens.pr_auc, 0.85);
        CHECK(Rens.recall >= 0.80, "Ensemble catches >=80% of fraud at the alert budget", Rens.recall, 0.80);
        CHECK(Rens.precision >= 0.40, "Ensemble precision usable at the alert budget", Rens.precision, 0.40);
    }
}

int main(int argc, char** argv) {
    std::printf("=== fraud_demo (unsupervised fraud detection: Autoencoder + Isolation Forest) ===\n");

    // ---- generated labeled fraud set ----
    const int n_legit = 4000, n_fraud = 80;             // ~2% fraud (realistic imbalance)
    fraud::Dataset d = fraud::generate(n_legit, n_fraud, /*seed*/7);
    std::printf("Generated %d transactions, %d fraud (%.1f%%). Feature means (legit vs fraud):\n",
                (int)d.label.size(), d.n_fraud(), 100.0*d.n_fraud()/d.label.size());
    // human-readable contrast
    for (int j = 0; j < (int)d.feature_names.size(); j++) {
        double ml=0, mf=0; int nl=0, nf=0;
        for (int i = 0; i < (int)d.label.size(); i++) { if (d.label[i]) { mf += d.X(i,j); nf++; } else { ml += d.X(i,j); nl++; } }
        std::printf("    %-18s legit=%8.2f   fraud=%8.2f\n", d.feature_names[j].c_str(), ml/nl, mf/nf);
    }
    double alert_rate = 2.0 * (double)n_fraud / (n_legit + n_fraud);   // ~2x base rate (investigator budget)
    std::printf("Operating point: flag the top %.1f%% by anomaly score (~2x the fraud base rate)\n", 100*alert_rate);
    runPipeline(d, alert_rate, /*assert*/true);

    // ---- optional: a REAL public fraud CSV (evidence demo) ----
    if (argc > 2) {
        std::printf("\n[REAL DATA] %s  (label column '%s')\n", argv[1], argv[2]);
        fraud::Dataset r = fraud::loadCSV(argv[1], argv[2]);
        if (r.label.empty()) { std::printf("  could not load / no label column\n"); }
        else {
            double ar = std::max(1e-4, (double)r.n_fraud() / r.label.size());
            std::printf("  %d rows, %d features, %d fraud (%.3f%%)\n",
                        (int)r.label.size(), (int)r.X.cols(), r.n_fraud(), 100.0*r.n_fraud()/r.label.size());
            runPipeline(r, ar, /*assert*/false);   // report-only on real data
        }
    } else {
        std::printf("\n[REAL DATA] (skip — pass a CSV + label column to score a real public fraud set)\n");
    }

    std::printf("=== %s (%d failures) ===\n", g_fail ? "FAIL" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
