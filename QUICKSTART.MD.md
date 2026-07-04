# Djehuti NN – Quick Start

This guide gets you up and running with the free C++ deep‑learning library in a few minutes.  
No Python, no cloud, no GPU required – just a C++17 compiler and the Eigen header library.

----

1. Prerequisites

- **C++17 compiler** (GCC 8+ or Clang 7+).  
  On Ubuntu/Debian:
```bash
  sudo apt install g++ make

    Eigen 3 (header‑only).
    bash

    sudo apt install libeigen3-dev
```
    Or download from eigen.tuxfamily.org and set EIGEN_INCLUDE.

    OpenMP (optional, for multi‑threading).
    Usually included with GCC/Clang; if not, install libomp-dev.

    Linux/macOS (Windows via WSL/MSYS2 works too).

2. Get the Code

Clone the repository or download the source zip.
```bash

git clone https://github.com/bycoalex/djehuti_nn_free
cd djehuti_nn_free
```
Your folder should look like this:
text

djehuti_nn_free/
├── compile-native.sh
├── CONTRIBUTING.md
├── djehuti_fraud.hpp
├── djehuti_nn_core.hpp
├── djehuti_nn.hpp
├── exact_math_test
├── examples
│   ├── exact_math_test.cpp
│   ├── fraud_demo.cpp
│   ├── nn_demo.cpp
│   ├── nn_llm_demo.cpp
│   ├── nn_recovery.cpp
│   └── realdata_fraud.cpp
├── FAQ.md
├── fraud_demo
├── LICENSE
├── NN_API.md
├── nn_demo
├── nn_llm_demo
├── nn_recovery
├── QUICKSTART.MD.md    <===== You are HERE
├── README.md
├── realdata_files
│   └── Fraud-ULB-credit-card-sourcefile.txt <===== to download the csv file to do the FRAUD test, once downloaded please put the csv inside the folder.
└── verify_nn.sh


3. Build the Quick Demo

The compile-native.sh script auto‑detects your CPU and sets optimal compiler flags (AVX2/AVX‑512, cache sizes, OpenMP threads).
```bash

./compile-native.sh
```
This builds the default demo (examples/nn_demo.cpp) and creates an executable nn_demo.
If Eigen is not in /usr/include/eigen3, set EIGEN_INCLUDE:
```bash

EIGEN_INCLUDE=/path/to/eigen3 ./compile-native.sh
```
4. Run the Quick Demo
```bash

./nn_demo
```
You should see output like:
```text

Djehuti SDK — AI suite quick demo (train a net + detect anomalies, zero deps)

== a 2->16->3 MLP classifies three clusters ==
  loss 5.602 -> 0.000   train accuracy = 100.0%
  [PASS] the net learns the classes (>95%)

== an Isolation Forest flags off-distribution points ==
  mean anomaly score: normal=0.477  outliers=0.726
  [PASS] outliers score higher than normal data

ALL PASS — that's the entry suite. Pro adds Transformers, deep RL, and a self-hosted LLM.
```
This trains a small classifier and runs anomaly detection – all within a second.

5. Build Other Demos

Use the DEMO environment variable to select any .cpp file from the examples/ folder.
Demo	 | Command | What it does
MLP classification + Isolation Forest |	./compile-native.sh (default) |	Quick sanity check
Full gradient‑check + learning anchors |	DEMO=nn_recovery ./compile-native.sh	 | Runs 70+ tests; takes ~30s
Char‑level GPT training & generation	| DEMO=nn_llm_demo ./compile-native.sh	| Trains a small Transformer on text; ~10s
SIMD transcendental accuracy	 | DEMO=exact_math_test ./compile-native.sh | Verifies fast exp/log/erf approximations
Fraud detection (synthetic)	| DEMO=fraud_demo ./compile-native.sh |	Autoencoder + Isolation Forest on fraud data

Run any demo:
```bash

DEMO=nn_recovery ./compile-native.sh && ./nn_recovery
```

6. Run the Full Verification Suite

The verify_nn.sh script builds and runs the core tests: nn_recovery, nn_llm_demo, exact_math_test (both FAST and EXACT profiles), and fraud_demo.
```bash

./verify_nn.sh
```
All tests should pass with ALL TESTS PASSED. This confirms the library works correctly on your machine.

7. (Optional) Test on Real Fraud Data

If you have the ULB credit‑card fraud dataset (creditcard.csv from OpenML/Kaggle), you can run the real‑data validation:
```bash

DEMO=realdata_fraud ./compile-native.sh
# pass the FOLDER that contains creditcard.csv (not the file itself):
./realdata_fraud realdata_files
```
You'll see ROC‑AUC and PR‑AUC scores, and self‑checks (≥0.90 ROC‑AUC, catches >55% of fraud in the top 1% alerts).

8. Next Steps

- Read the API Reference in NN_API.md to learn about Tensor, layers, optimizers, and RL.
- Check the FAQ if you run into license or usage questions.
- Browse the source – the library is header‑only and heavily commented.

Enjoy building offline, air‑gap AI systems!


## License

Non-exclusive source license with OEM/embed rights. See [LICENSE](LICENSE) file for details.

*Documentation generated from source code. For questions or issues, please open an issue on GitHub.*


## Information

Need help? Open an issue on GitHub or reach out via Webuildcodes.

**Webuildcodes** | [Website](https://www.webuild.codes) | [Gumroad](https://webuildcodes.gumroad.com) | [GitHub](https://github.com/bycoalex)
