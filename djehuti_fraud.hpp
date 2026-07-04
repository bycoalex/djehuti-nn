// =====================================================================
// Djehuti SDK — fraud pack (on the AI suite).  Unsupervised transaction-
// fraud detection: the "already-in-business buyer" hook — fraud is a problem
// companies KNOW they have and pay for, and the few-labels reality is exactly
// what unsupervised anomaly detection is for.
//
//   * generate()  — a realistic LABELED transaction set: a correlated legit
//                   manifold + injected fraud (odd amounts, velocity bursts,
//                   off-hours, far-from-home).  For the self-test / demo.
//   * score with the AI suite's deep Autoencoder + Isolation Forest (no labels).
//   * report the metrics a fraud team actually buys on: precision / recall /
//     F1 / PR-AUC / ROC-AUC at an operating threshold, and "% of fraud caught".
//   * loadCSV() — ingest a REAL public fraud CSV for the live, evidence-logged demo.
//
// Reuses nn/djehuti_nn.hpp (Autoencoder, IsolationForest, binaryPRF, batchRows,
// Adam, mseLoss).  Header-only, deterministic under a fixed seed.
//
// Copyright © Co Bros & Alex Ramanaidou (Webuildcodes). All rights reserved.
// Licensed under the Non‑Exclusive Source License – see LICENSE file.
//
// =====================================================================
#pragma once
#include "djehuti_nn.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>

namespace djehuti {
namespace fraud {

struct Dataset {
    Eigen::MatrixXd X;                       // (n_txn × n_features)
    std::vector<int> label;                  // 1 = fraud, 0 = legit
    std::vector<std::string> feature_names;
    int n_fraud() const { int s = 0; for (int l : label) s += l; return s; }
};

// ---- PR-AUC (average precision) and ROC-AUC from anomaly scores (higher = more anomalous) ----
inline double rocAuc(const Eigen::VectorXd& score, const std::vector<int>& label) {
    int n = (int)label.size();
    std::vector<int> idx(n); for (int i = 0; i < n; i++) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return score(a) < score(b); });
    // average ranks (1..n) handling ties; Mann-Whitney U
    std::vector<double> rank(n);
    for (int i = 0; i < n; ) {
        int j = i; while (j < n && score(idx[j]) == score(idx[i])) j++;
        double r = 0.5 * ((i + 1) + j);      // average rank for the tie block
        for (int k = i; k < j; k++) rank[idx[k]] = r;
        i = j;
    }
    double sum_pos = 0; long npos = 0, nneg = 0;
    for (int i = 0; i < n; i++) { if (label[i]) { sum_pos += rank[i]; npos++; } else nneg++; }
    if (npos == 0 || nneg == 0) return 0.5;
    return (sum_pos - npos * (npos + 1) / 2.0) / ((double)npos * nneg);
}
inline double prAuc(const Eigen::VectorXd& score, const std::vector<int>& label) {
    int n = (int)label.size();
    std::vector<int> idx(n); for (int i = 0; i < n; i++) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return score(a) > score(b); });  // most anomalous first
    long P = 0; for (int l : label) P += l; if (P == 0) return 0.0;
    long tp = 0, fp = 0; double ap = 0; double prev_recall = 0;
    for (int i = 0; i < n; i++) {
        if (label[idx[i]]) tp++; else fp++;
        double recall = (double)tp / P;
        double precision = (double)tp / (tp + fp);
        ap += precision * (recall - prev_recall);     // area under PR (step) curve
        prev_recall = recall;
    }
    return ap;
}

struct Report {
    double precision = 0, recall = 0, f1 = 0, pr_auc = 0, roc_auc = 0, pct_caught = 0;
    int flagged = 0, frauds_caught = 0, total_fraud = 0;
    std::string method;
};

// Evaluate anomaly scores against labels at an operating point = flag the top
// `alert_rate` fraction by score (a fraud team's alert budget).
inline Report evaluate(const std::string& method, const Eigen::VectorXd& score,
                       const std::vector<int>& label, double alert_rate) {
    Report R; R.method = method;
    int n = (int)label.size();
    R.total_fraud = 0; for (int l : label) R.total_fraud += l;
    int k = std::max(1, (int)std::round(alert_rate * n));
    std::vector<int> idx(n); for (int i = 0; i < n; i++) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return score(a) > score(b); });
    std::vector<int> pred(n, 0); for (int i = 0; i < k; i++) pred[idx[i]] = 1;
    auto prf = nn::binaryPRF(pred, label, 1);
    R.precision = prf.precision; R.recall = prf.recall; R.f1 = prf.f1;
    R.flagged = k; R.pct_caught = prf.recall * 100.0;
    R.frauds_caught = 0; for (int i = 0; i < n; i++) if (pred[i] && label[i]) R.frauds_caught++;
    R.pr_auc = prAuc(score, label); R.roc_auc = rocAuc(score, label);
    return R;
}

// =====================================================================
// Synthetic labeled fraud set — correlated legit manifold + injected fraud.
// Interpretable features so the demo reads like a fraud report.
// =====================================================================
inline Dataset generate(int n_legit, int n_fraud, uint64_t seed = 7) {
    Dataset d;
    d.feature_names = {"log_amount","hour","velocity","log_distance_km","merchant_risk","account_age_days"};
    const int F = 6, N = n_legit + n_fraud;
    d.X.resize(N, F); d.label.assign(N, 0);
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> z(0, 1);
    auto clip = [](double x, double lo, double hi){ return std::min(hi, std::max(lo, x)); };

    // legit lies near a 2-factor manifold (a "spending" axis and an "habit" axis),
    // so an autoencoder has real structure to learn and reconstruct.
    auto legitRow = [&](int r){
        double f1 = z(rng), f2 = z(rng);                              // spending, habit
        d.X(r,0) = 3.0 + 0.8*f1 + 0.10*z(rng);                        // log amount  (~$20)
        d.X(r,2) = std::max(0.0, 1.0 + 0.5*f1 + 0.10*z(rng));         // velocity (corr w/ amount)
        d.X(r,4) = clip(0.12 + 0.05*f1 + 0.02*z(rng), 0, 1);         // merchant risk (corr)
        d.X(r,1) = clip(13 + 3.0*f2 + 0.30*z(rng), 0, 23);           // hour (business, habit)
        d.X(r,5) = std::max(30.0, 800 + 250*f2 + 30*z(rng));         // account age (habit)
        d.X(r,3) = 2.0 + 0.4*f2 + 0.20*z(rng);                       // log distance (~7km)
    };
    int row = 0;
    for (int i = 0; i < n_legit; i++, row++) { legitRow(row); d.label[row] = 0; }

    // fraud = DIVERSE off-manifold anomalies: start from a plausible legit row, then
    // push a RANDOM subset (2-3) of axes to fraud extremes (odd amount, off-hours,
    // velocity burst, far-from-home, high merchant risk, new account).  Scattered,
    // not a single tight cluster — which is both realistic and AE-detectable.
    std::uniform_int_distribution<int> naxes(2, 3);
    std::uniform_int_distribution<int> pick(0, 5);
    for (int i = 0; i < n_fraud; i++, row++) {
        legitRow(row);
        int k = naxes(rng); bool done[6] = {false,false,false,false,false,false};
        for (int a = 0; a < k; ) {
            int ax = pick(rng); if (done[ax]) continue; done[ax] = true; a++;
            switch (ax) {
                case 0: d.X(row,0) = 6.0 + 0.6*z(rng); break;                 // very high amount
                case 1: d.X(row,1) = clip(3 + 1.5*z(rng), 0, 23); break;      // off-hours
                case 2: d.X(row,2) = std::max(0.0, 9 + 2.5*z(rng)); break;    // velocity burst
                case 3: d.X(row,3) = 5.5 + 0.7*z(rng); break;                 // far from home
                case 4: d.X(row,4) = clip(0.78 + 0.10*z(rng), 0, 1); break;   // high merchant risk
                case 5: d.X(row,5) = std::max(5.0, 60 + 40*z(rng)); break;    // brand-new account
            }
        }
        d.label[row] = 1;
    }
    return d;
}

// z-score standardize columns (unsupervised: uses the full set, no labels).
inline Eigen::MatrixXd standardize(const Eigen::MatrixXd& X) {
    Eigen::RowVectorXd mean = X.colwise().mean();
    Eigen::MatrixXd Xc = X.rowwise() - mean;
    Eigen::RowVectorXd sd = (Xc.array().square().colwise().sum() / std::max<long>(1, X.rows()-1)).sqrt();
    for (int j = 0; j < X.cols(); j++) if (sd(j) < 1e-12) sd(j) = 1.0;
    return Xc.array().rowwise() / sd.array();
}

// ensemble of two anomaly scores: average after z-standardizing each (the
// production recommendation — combines the deep AE and the tree-based detector).
inline Eigen::VectorXd ensembleScore(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
    auto zs = [](const Eigen::VectorXd& v){
        double m = v.mean(); double s = std::sqrt((v.array() - m).square().mean());
        if (s < 1e-12) s = 1.0;
        return ((v.array() - m) / s).matrix().eval();
    };
    return (0.5 * (zs(a) + zs(b))).eval();
}

// ---- the two unsupervised scorers from the AI suite ----
inline Eigen::VectorXd scoreIsolationForest(const Eigen::MatrixXd& Xstd, uint64_t seed = 0) {
    nn::IsolationForest forest(150, 256, seed);
    forest.fit(Xstd);
    return forest.scores(Xstd);
}
// train a deep autoencoder on the (unlabeled) data; anomaly score = reconstruction error.
// Two robustness details that matter for UNSUPERVISED anomaly detection (no labels):
//   * WINSORIZE the standardized input to +/-clip so a few extreme anomalies don't
//     dominate the MSE and get themselves fitted (they'd then score as "normal").
//   * a TIGHT bottleneck (small latent) so the model can only represent the dominant
//     normal manifold — off-manifold fraud then reconstructs poorly (high score).
// train on Xtrain (a possibly-subsampled view), score every row of Xscore.
inline Eigen::VectorXd scoreAutoencoder(const Eigen::MatrixXd& Xscore, const Eigen::MatrixXd& Xtrain,
                                        int epochs = 200, double lr = 0.01, uint64_t seed = 1, double clip = 5.0) {
    int F = (int)Xscore.cols();
    Eigen::MatrixXd Tw = Xtrain.array().min(clip).max(-clip);   // winsorize (training)
    nn::Autoencoder ae(F, {8}, 3, seed);                        // bottleneck (latent 3)
    std::vector<int> tr((int)Tw.rows()); for (int i = 0; i < (int)Tw.rows(); i++) tr[i] = i;
    nn::Tensor T = nn::batchRows(Tw, tr);
    nn::Adam opt(ae.parameters(), lr);
    for (int e = 0; e < epochs; e++) {
        nn::Tensor recon = ae.forward(T);
        nn::Tensor loss = nn::mseLoss(recon, T);
        opt.zeroGrad(); loss.backward(); opt.step();
    }
    Eigen::MatrixXd Sw = Xscore.array().min(clip).max(-clip);
    std::vector<int> all((int)Sw.rows()); for (int i = 0; i < (int)Sw.rows(); i++) all[i] = i;
    nn::Tensor X = nn::batchRows(Sw, all);
    nn::Tensor recon = ae.forward(X);
    return nn::reconstructionError(recon, X);
}
inline Eigen::VectorXd scoreAutoencoder(const Eigen::MatrixXd& Xstd, int epochs = 200,
                                        double lr = 0.01, uint64_t seed = 1, double clip = 5.0) {
    return scoreAutoencoder(Xstd, Xstd, epochs, lr, seed, clip);
}

// ---- generic numeric-CSV loader for a REAL public fraud dataset ----
// header row required; `label_col` names the 0/1 fraud column; all other
// columns that parse as numbers become features.
inline Dataset loadCSV(const std::string& path, const std::string& label_col) {
    Dataset d; std::ifstream f(path); std::string line;
    // strip surrounding quotes/whitespace (OpenML CSV quotes fields, e.g. "V1", '0')
    auto clean = [](std::string s){
        size_t a = s.find_first_not_of(" \t\r\n\"'"); size_t b = s.find_last_not_of(" \t\r\n\"'");
        return (a == std::string::npos) ? std::string() : s.substr(a, b - a + 1);
    };
    if (!std::getline(f, line)) return d;
    std::vector<std::string> cols; { std::stringstream ss(line); std::string c; while (std::getline(ss, c, ',')) cols.push_back(clean(c)); }
    int lc = -1; for (int i = 0; i < (int)cols.size(); i++) if (cols[i] == label_col) lc = i;
    if (lc < 0) return d;
    std::vector<std::vector<double>> rows; std::vector<int> labs;
    while (std::getline(f, line)) {
        std::stringstream ss(line); std::string c; std::vector<double> feat; int j = 0; int lab = 0; bool bad = false;
        while (std::getline(ss, c, ',')) {
            std::string t = clean(c);
            if (j == lc) { try { lab = (int)std::lround(std::stod(t)); } catch (...) { bad = true; } }
            else { try { feat.push_back(std::stod(t)); } catch (...) { feat.push_back(0.0); } }
            j++;
        }
        if (!bad && !feat.empty()) { rows.push_back(feat); labs.push_back(lab ? 1 : 0); }
    }
    if (rows.empty()) return d;
    int N = (int)rows.size(), F = (int)rows[0].size();
    d.X.resize(N, F);
    for (int i = 0; i < N; i++) for (int k = 0; k < F && k < (int)rows[i].size(); k++) d.X(i,k) = rows[i][k];
    d.label = labs;
    for (int i = 0; i < (int)cols.size(); i++) if (i != lc) d.feature_names.push_back(cols[i]);
    return d;
}

} // namespace fraud
} // namespace djehuti
