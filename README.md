# Djehuti NN – Self‑contained C++17 Deep Learning Library

[![License](https://img.shields.io/badge/License-Non--exclusive-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)
[![Eigen](https://img.shields.io/badge/Eigen-3.4-blue.svg)](https://eigen.tuxfamily.org/)

**Copyright © Co Bros & Alex Ramanaidou (Webuildcodes). All rights reserved.**

**Header‑only, zero Python, zero external deps (except Eigen).**  
Autograd, Transformers, GPT, RL, Anomaly Detection, and Unsupervised transaction-fraud detection, all in one file.

> **This is source‑available software, not open source.**  
> You can use, modify, and embed it in your own products, but you may not resell the source as a standalone SDK. See [LICENSE](LICENSE).

## Quick Start

```cpp
#include "djehuti_nn.hpp"
using namespace djehuti::nn;

Sequential net;
net.add(std::make_shared<Linear>(2, 16));
net.add(std::make_shared<ReLU>());
net.add(std::make_shared<Linear>(16, 3));

Adam opt(net.parameters(), 0.01);
Tensor logits = net.forward(X);
Tensor loss = crossEntropyLoss(logits, labels);
loss.backward();
opt.step();
```

## Features

- Autograd Core – Reverse‑mode automatic differentiation
- Layers – Linear, Conv2d, RNN/LSTM/GRU, MultiHeadAttention, TransformerBlock, GPT
- Optimizers – SGD, Adam (with AdamW)
- Anomaly Detection – Autoencoder, VAE, Isolation Forest, PCA
- Reinforcement Learning – DQN, REINFORCE, PPO (with GridWorld & CartPole)
- Training Utilities – DataLoader, EarlyStopping, LR schedules, checkpointing
- SIMD Accelerated – AVX2/AVX‑512 auto‑detected
- Unsupervised transaction-fraud detection 

## Build
```bash
./compile-native.sh          # Auto-tuned for your CPU
STRICT=1 ./compile-native.sh # IEEE-reproducible math
EXACT=1 ./compile-native.sh  # Exact std::transcendentals
```
## Documentation

Full API reference in NN_API.md.

## License

Non‑exclusive source with OEM/embed rights – see [LICENSE](LICENSE).

## Information

[Webuildcodes](https://www.webuild.codes) | [Gumroad](https://webuildcodes.gumroad.com) | [GitHub](https://github.com/bycoalex)

---
