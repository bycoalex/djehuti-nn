# Frequently Asked Questions – Djehuti NN

## General

### Is this open source?

No. The source code is **publicly available**, but it is not Open Source in the OSI sense.  
You are free to view, modify, and use it under the terms of our [Non‑Exclusive Source License](LICENSE).

### What can I do with the code?

- ✅ Use it in your own projects (startups, enterprises, research, side projects).
- ✅ Modify it for your own needs.
- ✅ Ship compiled binaries of your own product that include this library (OEM).
- ❌ Resell the source code as a standalone SDK, or repackage it as a competing product.

### Can I use it in my commercial product?

Absolutely. The license explicitly grants OEM rights – you may embed the compiled library inside your commercial application and distribute it to your customers.

### Do I have to pay anything?

No, this free edition is gratis. The Licensor may offer paid versions (Pro, enterprise support, custom modules) separately.

## Technical

### What are the system requirements?

- C++17 compiler (GCC 11+ / Clang 13+).
- Eigen 3 (header‑only, installed separately).
- x86‑64 CPU (AVX2 recommended, but SSE4.2 works).
- Linux/macOS (Windows via WSL/MSYS2 works too).

### How do I build it?

Use `./compile-native.sh` – it auto‑detects your CPU and sets the best flags.

### Does it support GPU?

No, it is CPU‑only. SIMD (AVX‑512/AVX2) and OpenMP provide strong performance on modern CPUs.

### How do I contribute?

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Licensing

### Can I fork the repo and modify it?

Yes, you may fork and modify for your own use. If you submit improvements back, they will be licensed under the same terms (see CONTRIBUTING.md).

### Can I remove the copyright notice?

No, the license requires you to keep the copyright headers and notices intact.

### What happens if I breach the license?

Your rights terminate immediately; you must stop using and delete all copies of the source.