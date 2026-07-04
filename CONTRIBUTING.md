# Contributing to Djehuti NN

First off, thank you for considering contributing! This project thrives on community input.

## How to Contribute

- **Report bugs** – open an issue with a clear description and steps to reproduce.
- **Suggest features** – open an issue and explain the use case.
- **Submit code** – fork the repo, create a branch, and open a pull request (PR).

## Contribution License Agreement (CLA)

By submitting a pull request, you agree that:

- Your contribution is your original work (or you have the right to submit it).
- You grant the Licensor (Webuildcodes) a perpetual, worldwide, non‑exclusive, royalty‑free license to use, modify, and distribute your contribution under the same terms as the project’s license (the [Non‑Exclusive Source License](LICENSE)).
- You will not assert any patent claims against the Licensor or other users based on your contribution.

## Code Style

- Use C++17, stick to the existing indentation (2 spaces).
- Keep header‑only; avoid adding new dependencies.
- Add gradient‑checks for any new primitive op (the `nn_recovery` test suite will catch it).
- Update `NN_API.md` if you add new public API.

## Testing

Run `./verify_nn.sh` to ensure all tests pass before submitting a PR.

## Questions?

Open an issue or reach out via [Webuildcodes](https://www.webuild.codes).

Thank you for helping make Djehuti NN better!