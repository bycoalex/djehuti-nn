# Djehuti Neural Network API Reference

**Header-only C++17 Deep Learning, Reinforcement Learning & Anomaly Detection Library**

---
Version: Free Edition
Dependencies: C++17 compiler, Eigen 3 (header‑only)
License: Non‑exclusive source [LICENSE](LICENSE).
Copyright © Co Bros & Alex Ramanaidou (Webuildcodes).

Overview**

Djehuti NN provides a reverse‑mode autograd engine over dense tensors, a Module API (PyTorch‑style layers), and a suite of deep learning, reinforcement learning, and anomaly detection algorithms all in a single header. Every operation is gradient‑checked, ensuring correctness of backpropagation.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Core Concepts](#core-concepts)
   - [Tensor](#tensor)
   - [Tensor Operations](#tensor-operations)
3. [Primitive Ops](#primitive-ops)
   - [Elementwise Operations](#elementwise-operations)
   - [Matrix Operations](#matrix-operations)
   - [Activations](#activations)
   - [Reductions](#reductions)
   - [Shape Operations](#shape-operations)
   - [Losses](#losses)
4. [Module API](#module-api)
   - [Base Module](#base-module)
   - [Linear Layers](#linear-layers)
   - [Activation Layers](#activation-layers)
   - [Sequential Container](#sequential-container)
   - [Normalization Layers](#normalization-layers)
   - [Convolution Layers](#convolution-layers)
   - [Pooling Layers](#pooling-layers)
   - [Recurrent Layers](#recurrent-layers)
   - [Transformer Layers](#transformer-layers)
   - [Language Model](#language-model)
5. [Optimizers](#optimizers)
   - [SGD](#sgd)
   - [Adam](#adam)
6. [Training Utilities](#training-utilities)
   - [DataLoader](#dataloader)
   - [Batch Helper](#batch-helper)
   - [Gradient Clipping](#gradient-clipping)
   - [Learning Rate Schedules](#learning-rate-schedules)
   - [Early Stopping](#early-stopping)
   - [Checkpoints](#checkpoints)
7. [Metrics](#metrics)
   - [Accuracy](#accuracy)
   - [Binary PRF](#binary-prf)
   - [R² Score](#r-score)
8. [Anomaly Detection](#anomaly-detection)
   - [PCA](#pca)
   - [Autoencoder](#autoencoder)
   - [VAE](#vae)
   - [Isolation Forest](#isolation-forest)
9. [Reinforcement Learning](#reinforcement-learning)
   - [Environments](#environments)
   - [Tabular Agents](#tabular-agents)
   - [Deep Agents](#deep-agents)
10. [Build & Compilation](#build--compilation)
11. [Examples](#examples)

---

## System Requirements

### Minimum Requirements

| Component | Specification |
|-----------|---------------|
| **CPU** | x86-64 with SSE4.2 support (Intel Nehalem / AMD Bulldozer or newer) |
| **RAM** | 512 MB |
| **OS** | Linux (kernel 3.10+), macOS 10.15+, or Windows 10+ (MSYS2/WSL) |
| **Compiler** | GCC 8+ or Clang 7+ with C++17 support |
| **Build Tools** | Make or CMake 3.14+, pthreads |
| **Dependencies** | Eigen 3.3.7+ (header-only, not bundled) |

### Recommended Requirements

| Component | Specification |
|-----------|---------------|
| **CPU** | x86-64 with AVX2 & FMA support (Intel Haswell / AMD Excavator or newer) |
| **RAM** | 4 GB+ |
| **OS** | Linux (kernel 5.0+) or macOS 12+ |
| **Compiler** | GCC 11+ or Clang 15+ with OpenMP support |
| **Build Tools** | CMake 3.20+, Ninja, pthreads |
| **Dependencies** | Eigen 3.4.0+ (header-only, not bundled) |

> **Note:** For maximum performance, pair recommended hardware with `-march=native` to leverage AVX2/FMA instructions.

---

## Core Concepts

### Tensor

The fundamental data structure for all computations.

```cpp
class Tensor {
public:
    // Constructors & Factories
    Tensor();
    explicit Tensor(TensorPtr ptr);
    
    static Tensor zeros(const std::vector<int>& shape, bool requires_grad = false);
    static Tensor from(const std::vector<int>& shape, const Eigen::VectorXd& vals, bool requires_grad = false);
    static Tensor scalar(double x, bool requires_grad = false);
    static Tensor randn(const std::vector<int>& shape, double std_dev, uint64_t seed, bool requires_grad = true);
    
    // Accessors
    const std::vector<int>& shape() const;
    int size() const;
    int rank() const;
    double item() const;
    
    Eigen::VectorXd& value();
    Eigen::VectorXd& grad();
    Eigen::Map<RowMat> mat() const;
    Eigen::Map<RowMat> gmat() const;
    
    // Gradients
    void zeroGrad();
    void backward();  // root must be scalar
    
    // Static Helpers
    static int numel(const std::vector<int>& s);
};
```

**Example:**
```cpp
// Create a 2x3 tensor
Tensor t = Tensor::zeros({2, 3}, true);
t.value() << 1, 2, 3, 4, 5, 6;

// Scalar tensor
Tensor s = Tensor::scalar(3.14);
double val = s.item();  // 3.14

// Random initialization (Kaiming/He)
Tensor w = Tensor::randn({100, 50}, std::sqrt(2.0/100), 12345, true);
```

---

### Tensor Operations

All operations return new `Tensor` objects with proper gradients wired.

```cpp
// Elementwise operations (same shape)
Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, double s);
Tensor operator*(double s, const Tensor& a);

// Matrix operations
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor addBias(const Tensor& x, const Tensor& b);
Tensor transpose2d(const Tensor& a);

// Activations
Tensor relu(const Tensor& a);
Tensor sigmoidT(const Tensor& a);
Tensor tanhT(const Tensor& a);
Tensor leakyRelu(const Tensor& a, double alpha = 0.01);
Tensor gelu(const Tensor& a);  // exact erf-based GELU
Tensor silu(const Tensor& a);  // Swish: x * sigmoid(x)
Tensor expT(const Tensor& a);

// Softmax & friends
Tensor softmaxRow(const Tensor& a);
Tensor logSoftmaxRow(const Tensor& a);

// Reductions (return scalar)
Tensor sum(const Tensor& a);
Tensor mean(const Tensor& a);

// Shape operations
Tensor reshape(const Tensor& a, const std::vector<int>& newshape);
Tensor sliceCols(const Tensor& a, int c0, int c1);
Tensor sliceRows(const Tensor& a, int r0, int r1);
Tensor concatCols(const Tensor& a, const Tensor& b);
Tensor concatRows(const Tensor& a, const Tensor& b);

// Special ops
Tensor gatherColumns(const Tensor& X, const std::vector<int>& idx);
Tensor clampT(const Tensor& a, double lo, double hi);
Tensor minT(const Tensor& a, const Tensor& b);
Tensor layerNorm(const Tensor& x, const Tensor& gamma, const Tensor& beta, double eps = 1e-5);
Tensor embeddingLookup(const Tensor& W, const std::vector<int>& idx);

// Convolution
Tensor conv2d(const Tensor& x, const Tensor& W, const Tensor& b,
              int Cin, int H, int Wd, int Cout, int kh, int kw,
              int stride = 1, int pad = 0);

// Pooling
Tensor maxPool2d(const Tensor& x, int C, int H, int Wd, int k, int stride);
Tensor avgPool2d(const Tensor& x, int C, int H, int Wd, int k, int stride);
```

**Example:**
```cpp
Tensor a = Tensor::zeros({3, 4});
Tensor b = Tensor::zeros({3, 4});
Tensor c = a + b;  // elementwise add
Tensor d = matmul(a, b.transpose());  // matrix multiply

// Activation with gradient
Tensor x = Tensor::randn({10, 100}, 0.1, 1, true);
Tensor y = relu(x);
```

---

## Primitive Ops

### Elementwise Operations

| Operation | Function | Gradient Rule |
|-----------|----------|---------------|
| Addition | `a + b` | `dA += dY`, `dB += dY` |
| Subtraction | `a - b` | `dA += dY`, `dB -= dY` |
| Multiplication | `a * b` | `dA += dY * b`, `dB += dY * a` |
| Scalar Multiplication | `a * s` | `dA += dY * s` |

### Matrix Operations

| Operation | Function | Gradient Rule |
|-----------|----------|---------------|
| Matrix Multiply | `matmul(a, b)` | `dA = dC * Bᵀ`, `dB = Aᵀ * dC` |
| Bias Addition | `addBias(x, b)` | `dX = dY`, `db = sum_rows(dY)` |
| Transpose | `transpose2d(a)` | `dA = dYᵀ` |

### Activations

| Activation | Function | Notes |
|------------|----------|-------|
| ReLU | `relu(x)` | `max(0, x)` |
| Sigmoid | `sigmoidT(x)` | `1/(1 + exp(-x))` |
| Tanh | `tanhT(x)` | `tanh(x)` |
| LeakyReLU | `leakyRelu(x, alpha=0.01)` | `x > 0 ? x : alpha*x` |
| GELU | `gelu(x)` | `x * Φ(x)` (exact erf) |
| SiLU/Swish | `silu(x)` | `x * σ(x)` |
| Exp | `expT(x)` | `exp(x)` |

### Reductions

| Operation | Function | Returns |
|-----------|----------|---------|
| Sum | `sum(a)` | Scalar tensor |
| Mean | `mean(a)` | Scalar tensor |

### Shape Operations

| Operation | Function | Behavior |
|-----------|----------|----------|
| Reshape | `reshape(a, newshape)` | Identity on flat buffer |
| Slice Columns | `sliceCols(a, c0, c1)` | (N, c1-c0) |
| Slice Rows | `sliceRows(a, r0, r1)` | (r1-r0, M) |
| Concat Columns | `concatCols(a, b)` | Both (N, *) |
| Concat Rows | `concatRows(a, b)` | Both (*, M) |

### Losses

| Loss | Function | Returns |
|------|----------|---------|
| Mean Squared Error | `mseLoss(pred, target)` | Scalar |
| Cross Entropy (logits) | `crossEntropyLoss(logits, targets)` | Scalar |
| Binary CE (logits) | `bceWithLogits(logits, targets)` | Scalar |

**Example:**
```cpp
Tensor pred = model.forward(x);
Tensor loss = crossEntropyLoss(pred, labels);
loss.backward();  // computes gradients for all requires_grad tensors
```

---

## Module API

### Base Module

All layers inherit from `Module`.

```cpp
struct Module {
    virtual ~Module() = default;
    virtual Tensor forward(const Tensor& x) = 0;
    virtual std::vector<Tensor> parameters() { return {}; }
    
    Tensor operator()(const Tensor& x) { return forward(x); }
};
using ModulePtr = std::shared_ptr<Module>;
```

### Linear Layers

#### Linear (Fully Connected)

```cpp
class Linear : public Module {
public:
    Tensor W, b;
    
    Linear(int in_features, int out_features, uint64_t seed = 1);
    
    Tensor forward(const Tensor& x) override;
    std::vector<Tensor> parameters() override;
};
```

**Example:**
```cpp
auto fc = std::make_shared<Linear>(128, 64);
Tensor x = Tensor::randn({32, 128}, 0.1, 1, false);
Tensor y = fc->forward(x);  // (32, 64)
```

---

### Activation Layers

```cpp
class ReLU : public Module {
    Tensor forward(const Tensor& x) override;
};

class Tanh : public Module {
    Tensor forward(const Tensor& x) override;
};

class Sigmoid : public Module {
    Tensor forward(const Tensor& x) override;
};
```

**Example:**
```cpp
auto relu = std::make_shared<ReLU>();
Tensor y = relu->forward(x);
```

---

### Sequential Container

```cpp
class Sequential : public Module {
public:
    std::vector<ModulePtr> mods;
    
    Sequential();
    explicit Sequential(std::vector<ModulePtr> m);
    void add(ModulePtr m);
    
    Tensor forward(const Tensor& x) override;
    std::vector<Tensor> parameters() override;
};
```

**Example:**
```cpp
Sequential net;
net.add(std::make_shared<Linear>(784, 256));
net.add(std::make_shared<ReLU>());
net.add(std::make_shared<Linear>(256, 128));
net.add(std::make_shared<ReLU>());
net.add(std::make_shared<Linear>(128, 10));
```

---

### Normalization Layers

#### LayerNorm

```cpp
class LayerNorm : public Module {
public:
    Tensor gamma, beta;
    double eps;
    
    explicit LayerNorm(int d, double eps_ = 1e-5);
    
    Tensor forward(const Tensor& x) override;
    std::vector<Tensor> parameters() override;
};
```

**Example:**
```cpp
auto ln = std::make_shared<LayerNorm>(512);
Tensor y = ln->forward(x);
```

---

### Convolution Layers

#### Conv2d (Single Image)

```cpp
class Conv2d : public Module {
public:
    int Cin, Cout, kh, kw, stride, pad;
    Tensor W, b;
    
    Conv2d(int Cin_, int Cout_, int k, int stride_ = 1, int pad_ = 0, uint64_t seed = 1);
    
    Tensor forward(const Tensor& x) override;  // x: (Cin, H, W)
    std::vector<Tensor> parameters() override;
};
```

**Example:**
```cpp
auto conv = std::make_shared<Conv2d>(3, 16, 3, 1, 1);
Tensor x = Tensor::randn({3, 32, 32}, 0.1, 1, false);
Tensor y = conv->forward(x);  // (16, 32, 32)
```

---

### Pooling Layers

```cpp
class MaxPool2d : public Module {
public:
    int k, stride;
    explicit MaxPool2d(int k_, int stride_ = -1);
    Tensor forward(const Tensor& x) override;
};

class AvgPool2d : public Module {
public:
    int k, stride;
    explicit AvgPool2d(int k_, int stride_ = -1);
    Tensor forward(const Tensor& x) override;
};
```

**Example:**
```cpp
auto pool = std::make_shared<MaxPool2d>(2, 2);
Tensor y = pool->forward(x);  // (C, H/2, W/2)
```

---

### Recurrent Layers

#### RNN

```cpp
class RNN : public Module {
public:
    int in_dim, hid;
    Tensor Wx, Wh, b;
    
    RNN(int in_, int hid_, uint64_t seed = 1);
    
    Tensor forward(const Tensor& x) override;  // x: (T, in_dim) -> (T, hid)
    std::vector<Tensor> parameters() override;
};
```

#### LSTM

```cpp
class LSTM : public Module {
public:
    int in_dim, hid;
    Tensor Wx, Wh, b;
    
    LSTM(int in_, int hid_, uint64_t seed = 1);
    
    Tensor forward(const Tensor& x) override;  // x: (T, in_dim) -> (T, hid)
    std::vector<Tensor> parameters() override;
};
```

#### GRU

```cpp
class GRU : public Module {
public:
    int in_dim, hid;
    // Separate weight/bias for each gate
    Tensor Wxz, Whz, bz, Wxr, Whr, br, Wxn, Whn, bn;
    
    GRU(int in_, int hid_, uint64_t seed = 1);
    
    Tensor forward(const Tensor& x) override;  // x: (T, in_dim) -> (T, hid)
    std::vector<Tensor> parameters() override;
};
```

**Example:**
```cpp
auto lstm = std::make_shared<LSTM>(128, 256);
Tensor x = Tensor::randn({10, 32, 128}, 0.1, 1, false);  // (seq_len, batch, features)
Tensor h = lstm->forward(x);  // (seq_len, batch, 256)
```

---

### Transformer Layers

#### MultiHeadAttention

```cpp
class MultiHeadAttention : public Module {
public:
    int d_model, n_heads, d_head;
    bool causal;
    Tensor Wq, bq, Wk, bk, Wv, bv, Wo, bo;
    
    MultiHeadAttention(int d_model_, int n_heads_, bool causal_ = false, uint64_t seed = 1);
    
    Tensor forward(const Tensor& x) override;  // x: (seq, d_model)
    std::vector<Tensor> parameters() override;
};
```

#### TransformerBlock (Pre-norm)

```cpp
class TransformerBlock : public Module {
public:
    MultiHeadAttention attn;
    LayerNorm ln1, ln2;
    Linear ff1, ff2;
    
    TransformerBlock(int d_model, int n_heads, int d_ff, bool causal = false, uint64_t seed = 1);
    
    Tensor forward(const Tensor& x) override;
    std::vector<Tensor> parameters() override;
};
```

**Example:**
```cpp
auto block = std::make_shared<TransformerBlock>(512, 8, 2048, true);
Tensor x = Tensor::randn({64, 512}, 0.1, 1, false);
Tensor y = block->forward(x);  // (64, 512)
```

---

### Language Model

#### GPT (Causal Language Model)

```cpp
class GPT : public Module {
public:
    int vocab, d_model, n_layers, n_heads, d_ff, max_seq;
    Embedding tok;
    Tensor pos;  // learned positional embedding (max_seq, d_model)
    std::vector<std::shared_ptr<TransformerBlock>> blocks;
    LayerNorm lnf;
    Linear head;
    
    GPT(int vocab_, int d_model_, int n_layers_, int n_heads_, int d_ff_, int max_seq_, uint64_t seed = 1);
    
    Tensor forwardIds(const std::vector<int>& ids);  // (T, vocab) logits
    Tensor forward(const Tensor& x) override;  // use forwardIds()
    std::vector<Tensor> parameters() override;
};
```

**Autoregressive Generation:**
```cpp
std::vector<int> gptGenerate(GPT& model, std::vector<int> ctx, int n_new,
                             double temperature, std::mt19937_64& rng,
                             bool greedy = false);
```

**Example:**
```cpp
// Train a character-level GPT
GPT model(27, 64, 2, 4, 128, 256);
std::vector<int> ids = {0, 1, 2, 3, 4};  // token ids
Tensor logits = model.forwardIds(ids);   // (5, 27)

// Generate
std::mt19937_64 rng(42);
auto output = gptGenerate(model, {0}, 100, 0.8, rng);
```

---

## Optimizers

### SGD

```cpp
class SGD {
public:
    SGD(std::vector<Tensor> params, double lr, double momentum = 0.0);
    
    void zeroGrad();
    void step();
    void setLR(double v);
};
```

**Example:**
```cpp
auto params = net.parameters();
SGD optimizer(params, 0.01, 0.9);

// Training loop
optimizer.zeroGrad();
loss.backward();
optimizer.step();
```

### Adam

```cpp
class Adam {
public:
    Adam(std::vector<Tensor> params, 
         double lr = 1e-3, 
         double b1 = 0.9, 
         double b2 = 0.999,
         double eps = 1e-8, 
         double weight_decay = 0.0);
    
    void zeroGrad();
    void step();
    void setLR(double v);
};
```

**Example:**
```cpp
Adam optimizer(net.parameters(), 0.001, 0.9, 0.999, 1e-8, 0.01);  // AdamW with weight decay
```

---

## Training Utilities

### DataLoader

```cpp
class DataLoader {
public:
    DataLoader(int n, int batch_size, bool shuffle = true, uint64_t seed = 0);
    std::vector<std::vector<int>> epoch();  // Returns batch indices
};
```

**Example:**
```cpp
DataLoader loader(10000, 64, true, 123);
auto batches = loader.epoch();
for (auto& batch : batches) {
    // batch contains indices for this mini-batch
}
```

### Batch Helper

```cpp
Tensor batchRows(const Eigen::MatrixXd& X, const std::vector<int>& idx);
```

**Example:**
```cpp
Eigen::MatrixXd X(10000, 784);
auto batch_idx = loader.epoch()[0];
Tensor x_batch = batchRows(X, batch_idx);
```

### Gradient Clipping

```cpp
void clipGradNorm(std::vector<Tensor>& params, double max_norm);
```

**Example:**
```cpp
auto params = net.parameters();
clipGradNorm(params, 1.0);  // Clip to max norm of 1.0
```

### Learning Rate Schedules

```cpp
double lrStep(double base, int epoch, int step_size, double gamma);
double lrExp(double base, int epoch, double gamma);
double lrCosine(double base, int epoch, int total);
double lrWarmupCosine(double base, int epoch, int warmup, int total);
```

**Example:**
```cpp
for (int epoch = 0; epoch < 100; ++epoch) {
    double lr = lrWarmupCosine(0.001, epoch, 5, 100);
    optimizer.setLR(lr);
    // train...
}
```

### Early Stopping

```cpp
class EarlyStopping {
public:
    explicit EarlyStopping(int patience = 10);
    bool step(double val_loss);
    double best() const;
};
```

**Example:**
```cpp
EarlyStopping early_stop(10);
for (int epoch = 0; epoch < 1000; ++epoch) {
    double val_loss = validate();
    if (early_stop.step(val_loss)) {
        std::cout << "Early stopping!" << std::endl;
        break;
    }
}
```

### Checkpoints

```cpp
void saveParams(const std::vector<Tensor>& params, const std::string& path);
bool loadParams(std::vector<Tensor>& params, const std::string& path);
```

**Example:**
```cpp
// Save
auto params = net.parameters();
saveParams(params, "model.bin");

// Load
auto params = net.parameters();
if (loadParams(params, "model.bin")) {
    std::cout << "Model loaded!" << std::endl;
}
```

---

## Metrics

### Accuracy

```cpp
double accuracy(const Tensor& logits, const std::vector<int>& targets);
```

**Example:**
```cpp
Tensor logits = model.forward(X);
double acc = accuracy(logits, labels);
```

### Binary PRF

```cpp
struct PRF { double precision, recall, f1; };
PRF binaryPRF(const std::vector<int>& pred, const std::vector<int>& truth, int positive = 1);
```

**Example:**
```cpp
auto metrics = binaryPRF(predictions, ground_truth);
std::cout << "Precision: " << metrics.precision
          << ", Recall: " << metrics.recall
          << ", F1: " << metrics.f1 << std::endl;
```

### R² Score

```cpp
double r2Score(const Eigen::VectorXd& pred, const Eigen::VectorXd& y);
```

**Example:**
```cpp
double r2 = r2Score(predictions, targets);
```

---

## Anomaly Detection

### PCA

```cpp
class PCA {
public:
    Eigen::MatrixXd components;          // (D, k) principal axes
    Eigen::VectorXd mean;                // (D) feature means
    Eigen::VectorXd explained_variance;  // (k) variance along components
    
    void fit(const Eigen::MatrixXd& X, int k);
    Eigen::MatrixXd transform(const Eigen::MatrixXd& X) const;
    Eigen::MatrixXd reconstruct(const Eigen::MatrixXd& X) const;
    Eigen::VectorXd reconstructionError(const Eigen::MatrixXd& X) const;
};
```

**Example:**
```cpp
PCA pca;
pca.fit(X_train, 50);
auto X_reduced = pca.transform(X_test);
auto X_recon = pca.reconstruct(X_test);
auto errors = pca.reconstructionError(X_test);
```

### Autoencoder

```cpp
class Autoencoder : public Module {
public:
    Sequential enc, dec;
    
    Autoencoder(int d_in, std::vector<int> hidden, int d_latent, uint64_t seed = 1);
    
    Tensor encode(const Tensor& x);
    Tensor forward(const Tensor& x) override;  // reconstruction
    std::vector<Tensor> parameters() override;
};

// Reconstruction error (anomaly score)
Eigen::VectorXd reconstructionError(const Tensor& recon, const Tensor& x);
```

**Example:**
```cpp
Autoencoder ae(784, {256, 128}, 32);
Tensor x = Tensor::randn({100, 784}, 0.1, 1, false);
Tensor recon = ae.forward(x);
auto scores = reconstructionError(recon, x);  // Higher = more anomalous
```

### VAE (Variational Autoencoder)

```cpp
class VAE : public Module {
public:
    Sequential enc, dec;
    int latent;
    
    VAE(int d_in, int hidden, int latent_, uint64_t seed = 1);
    
    struct Out { 
        Tensor recon, kl, mu, logvar, z; 
    };
    Out run(const Tensor& x, const Tensor& eps);
    
    Tensor forward(const Tensor& x) override;  // use run()
    std::vector<Tensor> parameters() override;
};
```

**Example:**
```cpp
VAE vae(784, 256, 32);
Tensor x = Tensor::randn({100, 784}, 0.1, 1, false);
Tensor eps = Tensor::randn({100, 32}, 1.0, 42, false);
auto out = vae.run(x, eps);

// ELBO = reconstruction loss + beta * KL
Tensor total_loss = mseLoss(out.recon, x) + out.kl * 0.1;
```

### Isolation Forest

```cpp
class IsolationForest {
public:
    IsolationForest(int n_trees = 100, int sample_size = 128, uint64_t seed = 0);
    
    void fit(const Eigen::MatrixXd& X);
    double score(const Eigen::VectorXd& x) const;  // returns (0,1), 1 = anomalous
    Eigen::VectorXd scores(const Eigen::MatrixXd& X) const;
};
```

**Example:**
```cpp
IsolationForest iforest(100, 256, 123);
iforest.fit(X_train);
auto anomaly_scores = iforest.scores(X_test);  // Higher = more anomalous
```

---

## Reinforcement Learning

### Environments

#### GridWorld (Discrete, Deterministic)

```cpp
class GridWorld {
public:
    GridWorld(int r = 4, int c = 4, int start_ = 0, int goal_ = -1, int maxSteps_ = 100);
    
    int nStates() const;
    int nActions() const;
    int obsDim() const;
    
    Eigen::VectorXd obs() const;
    Eigen::VectorXd reset();
    StepOut step(int a);  // {obs, reward, done}
};
```

#### CartPole (Continuous, Classic)

```cpp
class CartPole {
public:
    explicit CartPole(int maxSteps = 200, uint64_t seed = 0);
    
    int obsDim() const;
    int nActions() const;
    
    Eigen::VectorXd reset();
    StepOut step(int a);  // returns +1 reward per surviving step
};
```

**Example:**
```cpp
GridWorld env(4, 4, 0, 15, 100);
auto obs = env.reset();
auto result = env.step(0);  // Up
std::cout << "Reward: " << result.reward << ", Done: " << result.done << std::endl;
```

---

### Tabular Agents

#### Q-Learning

```cpp
class QLearning {
public:
    Eigen::MatrixXd Q;
    double alpha, gamma;
    std::mt19937_64 rng;
    
    QLearning(int S, int A, double alpha_ = 0.5, double gamma_ = 1.0, uint64_t seed = 0);
    
    int greedy(int s) const;
    int epsGreedy(int s, double eps);
    void update(int s, int a, double r, int sp, bool done);
};
```

#### SARSA (On-Policy)

```cpp
class SARSA {
public:
    Eigen::MatrixXd Q;
    double alpha, gamma;
    std::mt19937_64 rng;
    
    SARSA(int S, int A, double alpha_ = 0.5, double gamma_ = 1.0, uint64_t seed = 0);
    
    int greedy(int s) const;
    int epsGreedy(int s, double eps);
    void update(int s, int a, double r, int sp, int ap, bool done);
};
```

#### Double Q-Learning

```cpp
class DoubleQLearning {
public:
    Eigen::MatrixXd Qa, Qb;
    double alpha, gamma;
    std::mt19937_64 rng;
    
    DoubleQLearning(int S, int A, double alpha_ = 0.5, double gamma_ = 1.0, uint64_t seed = 0);
    
    int greedy(int s) const;
    int epsGreedy(int s, double eps);
    void update(int s, int a, double r, int sp, bool done);
};
```

---

### Deep Agents

#### DQN (Deep Q-Network)

```cpp
class DQNAgent {
public:
    std::shared_ptr<Sequential> online, target;
    std::unique_ptr<Adam> opt;
    ReplayBuffer buf;
    int obsDim, nActions;
    double gamma;
    bool doubleDQN;
    std::mt19937_64 rng;
    
    DQNAgent(int obsDim_, int nActions_, std::vector<int> hidden, 
             double lr = 1e-3, double gamma_ = 0.99, 
             int bufCap = 20000, bool doubleDQN_ = true, uint64_t seed = 0);
    
    int act(const Eigen::VectorXd& s, double eps);
    void remember(const Transition& t);
    void syncTarget();
    double trainStep(int batch_size);
};
```

**Example:**
```cpp
DQNAgent agent(4, 2, {64, 64}, 0.001, 0.99, 10000, true, 123);
auto obs = env.reset();
int action = agent.act(obs, 0.1);
agent.remember({obs, action, reward, next_obs, done});
double loss = agent.trainStep(32);
agent.syncTarget();
```

#### REINFORCE (Policy Gradient)

```cpp
class ReinforceAgent {
public:
    std::shared_ptr<Sequential> policy;
    std::unique_ptr<Adam> opt;
    int obsDim, nActions;
    double gamma, entCoef;
    bool baseline;
    std::mt19937_64 rng;
    
    ReinforceAgent(int obsDim_, int nActions_, std::vector<int> hidden,
                   double lr = 1e-2, double gamma_ = 0.99, 
                   bool baseline_ = true, uint64_t seed = 0, double entCoef_ = 0.01);
    
    Eigen::VectorXd probs(const Eigen::VectorXd& s);
    int sample(const Eigen::VectorXd& s);
    double update(const std::vector<Eigen::VectorXd>& S,
                  const std::vector<int>& A,
                  const std::vector<double>& R);
};
```

**Example:**
```cpp
ReinforceAgent agent(4, 2, {64, 64}, 0.01, 0.99, true, 123);
std::vector<Eigen::VectorXd> states;
std::vector<int> actions;
std::vector<double> rewards;

// Collect episode...
double loss = agent.update(states, actions, rewards);
```

#### PPO (Proximal Policy Optimization)

```cpp
class PPOAgent {
public:
    std::shared_ptr<Sequential> actor, critic;
    std::unique_ptr<Adam> optA, optC;
    int obsDim, nActions;
    double gamma, lam, clip, entCoef;
    int epochs;
    std::mt19937_64 rng;
    
    PPOAgent(int obsDim_, int nActions_, std::vector<int> hidden,
             double lr = 3e-3, double gamma_ = 0.99, double lam_ = 0.95,
             double clip_ = 0.2, double entCoef_ = 0.01, 
             int epochs_ = 4, uint64_t seed = 0);
    
    Eigen::VectorXd probs(const Eigen::VectorXd& s);
    int sample(const Eigen::VectorXd& s);
    double value(const Eigen::VectorXd& s);
    void update(const std::vector<Eigen::VectorXd>& S,
                const std::vector<int>& A,
                const std::vector<double>& R,
                const std::vector<char>& done);
};
```

**Example:**
```cpp
PPOAgent agent(4, 2, {64, 64}, 0.003, 0.99, 0.95, 0.2, 0.01, 4, 123);
std::vector<Eigen::VectorXd> states;
std::vector<int> actions;
std::vector<double> rewards;
std::vector<char> done;

// Collect trajectory...
agent.update(states, actions, rewards, done);
```

---

## Build & Compilation

### Using the Provided Script

```bash
# Auto-tuned for your CPU
./compile-native.sh

# IEEE-reproducible math
STRICT=1 ./compile-native.sh

# Exact std::transcendentals (research-grade)
EXACT=1 ./compile-native.sh

# Custom Eigen path
EIGEN_INCLUDE=/path/to/eigen ./compile-native.sh
```

### Manual Compilation

```bash
g++ -std=c++17 -O3 -march=native -mtune=native -flto -fopenmp \
    -funroll-loops -ftree-vectorize -fprefetch-loop-arrays \
    -I. -I/usr/include/eigen3 \
    your_source.cpp -o your_program -lpthread -lm
```

### Compiler Flags Explained

| Flag | Description |
|------|-------------|
| `-march=native` | Optimize for your CPU |
| `-mtune=native` | Tune for your CPU |
| `-flto` | Link-time optimization |
| `-fopenmp` | Enable parallel processing |
| `-funroll-loops` | Unroll loops |
| `-ftree-vectorize` | Auto-vectorize |
| `-fprefetch-loop-arrays` | Prefetch data |
| `-ffast-math` | Fast math (default) |
| `-fno-fast-math` | IEEE-strict math |
| `-DDJEHUTI_EXACT_MATH` | Exact std::transcendentals |

---

## Examples

### Example 1: Simple MLP Classification

```cpp
#include "djehuti_nn.hpp"
using namespace djehuti::nn;

int main() {
    // Create model
    Sequential net;
    net.add(std::make_shared<Linear>(784, 256));
    net.add(std::make_shared<ReLU>());
    net.add(std::make_shared<Linear>(256, 128));
    net.add(std::make_shared<ReLU>());
    net.add(std::make_shared<Linear>(128, 10));

    // Optimizer
    Adam opt(net.parameters(), 0.001);

    // Training loop
    for (int epoch = 0; epoch < 100; ++epoch) {
        Tensor x = ...;  // (batch, 784)
        Tensor labels = ...;  // (batch, 10)
        
        Tensor logits = net.forward(x);
        Tensor loss = crossEntropyLoss(logits, labels);
        
        opt.zeroGrad();
        loss.backward();
        opt.step();
        
        std::cout << "Epoch " << epoch << " Loss: " << loss.item() << std::endl;
    }
}
```

### Example 2: Training a GPT Language Model

```cpp
#include "djehuti_nn.hpp"
using namespace djehuti::nn;

int main() {
    // Small GPT model
    GPT model(/*vocab*/ 27, /*d_model*/ 64, /*layers*/ 2, 
              /*heads*/ 4, /*d_ff*/ 128, /*max_seq*/ 256);
    
    Adam opt(model.parameters(), 0.001);
    
    // Training
    for (int epoch = 0; epoch < 50; ++epoch) {
        std::vector<int> ids = get_batch();  // (T) token ids
        std::vector<int> target_ids = get_next_tokens();
        
        Tensor logits = model.forwardIds(ids);  // (T, vocab)
        Tensor loss = crossEntropyLoss(logits, target_ids);
        
        opt.zeroGrad();
        loss.backward();
        opt.step();
    }
    
    // Generate
    std::mt19937_64 rng(42);
    std::vector<int> prompt = {1, 2, 3, 4};
    auto output = gptGenerate(model, prompt, 100, 0.8, rng);
}
```

### Example 3: Autoencoder for Anomaly Detection

```cpp
#include "djehuti_nn.hpp"
using namespace djehuti::nn;

int main() {
    // Build autoencoder
    Autoencoder ae(784, {256, 128}, 32);
    Adam opt(ae.parameters(), 0.001);
    
    // Train on normal data only
    for (int epoch = 0; epoch < 50; ++epoch) {
        Tensor x = get_normal_batch();
        Tensor recon = ae.forward(x);
        Tensor loss = mseLoss(recon, x);
        
        opt.zeroGrad();
        loss.backward();
        opt.step();
    }
    
    // Score anomalies
    Tensor test_data = get_test_batch();
    Tensor recon = ae.forward(test_data);
    auto scores = reconstructionError(recon, test_data);
    
    // Higher score = more anomalous
}
```

### Example 4: DQN for CartPole

```cpp
#include "djehuti_nn.hpp"
using namespace djehuti::nn;
using namespace djehuti::nn::rl;

int main() {
    CartPole env;
    DQNAgent agent(4, 2, {64, 64}, 0.001, 0.99, 10000, true, 123);
    
    for (int episode = 0; episode < 1000; ++episode) {
        auto obs = env.reset();
        double episode_reward = 0;
        bool done = false;
        
        while (!done) {
            int action = agent.act(obs, 0.1 * (1 - episode/1000));
            auto result = env.step(action);
            agent.remember({obs, action, result.reward, result.obs, result.done});
            
            obs = result.obs;
            episode_reward += result.reward;
            done = result.done;
            
            if (agent.buf.size() > 32) {
                agent.trainStep(32);
            }
        }
        
        if (episode % 10 == 0) {
            agent.syncTarget();
            std::cout << "Episode " << episode 
                      << " Reward: " << episode_reward << std::endl;
        }
    }
}
```

---

## Additional Notes

### Memory Management
- `Tensor` uses `shared_ptr` for automatic memory management
- `backward()` traverses the computational graph and frees intermediate nodes
- Gradients accumulate by default; call `zeroGrad()` before each backward pass

### Thread Safety
- Most operations are not thread-safe
- Use separate `Tensor` objects for different threads
- OpenMP is used internally for parallel matrix operations

### Performance Tips
1. Use `-march=native` for best SIMD performance
2. Enable `-fopenmp` for multi-threaded operations
3. Set `-flto` for link-time optimization
4. Use `Tensor::randn` with fixed seed for reproducibility
5. Batch operations are more efficient than per-sample loops
6. Use `matmul` instead of manual loops
7. Avoid creating unnecessary `Tensor` objects in tight loops

### Numerical Precision
- All operations use `double` precision
- `STRICT=1` enables IEEE-strict math (slower, more reproducible)
- `EXACT=1` uses `std::` transcendentals instead of fast approximations
- Gradient checking is done with tolerance ~1e-10 and can be lowered

---

## License

Non-exclusive source license with OEM/embed rights. See [LICENSE](LICENSE) file for details.

*Documentation generated from source code. For questions or issues, please open an issue on GitHub.*


## Information

**Webuildcodes** | [Website](https://www.webuild.codes) | [Gumroad](https://webuildcodes.gumroad.com) | [GitHub](https://github.com/bycoalex)

---
