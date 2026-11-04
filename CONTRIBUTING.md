# Contributing to Simulation of Gene Regulatory Networks Driving Shape Homeostasis

Thank you for your interest in contributing to this research software project. Whether you are extending the mathematical model, optimizing the reaction-diffusion solvers, adding evolutionary topologies, or improving documentation, we welcome well-scoped contributions.

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## 1. Maintainer Capacity & Project Scope

This repository is maintained independently alongside ongoing research and engineering work. Please keep the following scope and maintainer considerations in mind:

- **Supported Platforms**: We currently support **macOS (`Darwin`)** and **Linux (`Linux`)** using `Python 3.11+` (`uv`) and **Podman** or **Docker**. Native Windows scripts are out of scope (Windows users should use WSL2 or containers).
- **Discuss Before Large Changes**: Because maintainer review bandwidth is finite, please check [`docs/history-log.md`](docs/history-log.md) and open a GitHub Issue before starting large architectural refactors or adding new third-party dependencies.
- **Keep Pull Requests Focused**: Small, single-purpose Pull Requests linked to an Issue are reviewed and merged much faster than sweeping multi-topic changes.

---

## 2. Opening Issues (Bug Reports & Feature Requests)

Before submitting a Pull Request for a new feature or non-trivial fix, create a GitHub Issue (or reference an existing one from the [roadmap](docs/history-log.md)):

1. **Title Convention**: Use a Conventional Commit style prefix in the issue title so its scope is immediately clear:
   - `feat(evolution): ...`
   - `perf(core): ...`
   - `refactor(core): ...`
   - `fix(visualization): ...`
   - `docs(architecture): ...`
2. **Issue Body Structure**:
   - **Context / Motivation**: Link to the scientific concept, mathematical equation, or current limitation.
   - **Current State & Affected Files**: List the exact modules (`core/`, `evolution/`, `visualization/`, `analysis/`, `config/`) involved.
   - **Proposed Solution**: Describe the intended change concisely.
   - **Acceptance Criteria**: Provide a checklist (`- [ ] ...`) of testable outcomes.

---

## 3. Local Development Setup

Clone the repository and synchronize the development environment using [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/userravs/simulation-of-gene-regulatory-networks-driving-shape-homeostasis.git
cd simulation-of-gene-regulatory-networks-driving-shape-homeostasis

# Install runtime and development dependencies (ruff, pytest)
uv sync --dev
```

To run the application in a container via **Podman** (including Podman Desktop) or **Docker**:

```bash
./scripts/run_container.sh
```

---

## 4. Trunk-Based Development (TBD) & Branch Hygiene

We strictly follow **Trunk-Based Development (TBD)** with short-lived topic branches off `main` and **Squash-Merge** Pull Requests:

1. **Pre-Flight Check**: Always start from a clean, up-to-date `main` branch. Never stack new work on a stale or already-merged topic branch:
   ```bash
   git switch main
   git pull --ff-only origin main
   git status -sb
   ```
2. **Create a Short-Lived Topic Branch**: Name your branch `<type>/<short-slug>`:
   ```bash
   git switch -c feat/short-descriptive-slug
   ```
3. **Squash-Merge & Post-Merge Cleanup**: Pull Requests are Squash-Merged into `main` to keep the trunk history linear and atomic. After your PR is merged, immediately clean up your local branch:
   ```bash
   git switch main && git pull --ff-only origin main && git branch -D <merged-branch>
   ```

---

## 5. Conventional Commits Specification

All commit messages and Pull Request titles must follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```text
<type>(<scope>): <concise imperative summary>
```

### Allowed Types
| Type | Purpose |
|---|---|
| `feat` | New simulation capability, evolutionary operator, topology, or UI feature |
| `fix` | Bug fix in cellular automata rules, PDE solvers, evolution, or plotting |
| `perf` | Performance optimization (e.g., Numba `@njit`, linear algebra, multiprocessing) |
| `refactor` | Code restructuring without changing external mathematical behavior |
| `docs` | Documentation, mathematical explanations, citation metadata, or changelog updates |
| `test` | Adding or updating `pytest` unit tests in `tests/` |
| `ci` | GitHub Actions workflows (`.github/workflows/ci.yml`) or container build checks |
| `chore` | Dependency updates (`pyproject.toml`, `uv.lock`) or repository maintenance |
| `style` | PEP 8 / Ruff formatting and lint cleanups |

### Common Scopes
`core`, `evolution`, `visualization`, `analysis`, `config`, `scripts`, `architecture`, `history`, `release`.

---

## 6. Coding, Documentation & Versioning Standards

- **PEP 8 & Ruff Formatting**:
  - Maximum line length is **100 characters** (`[tool.ruff]` in `pyproject.toml`).
  - Do not leave commented-out legacy code in Python modules.
- **Namespace Packages**:
  - Do not add empty `__init__.py` files; the project uses PEP 420 implicit namespace packages configured in `pyproject.toml`.
- **Clean Scientific Prose**:
  - Write all source code comments, docstrings, Markdown documentation, commit messages, and issue/PR descriptions 100% in English without decorative emojis.
- **Version & Release Synchronization**:
  - When bumping the semantic version, update [`VERSION`](VERSION), [`pyproject.toml`](pyproject.toml), [`CITATION.cff`](CITATION.cff), and [`CHANGELOG.md`](CHANGELOG.md) together, and tag the release on `main` using an annotated Git tag (`vX.Y.Z`).

---

## 7. Pre-PR Verification Checklist

Before opening a Pull Request, run the full local verification suite (matching `.github/workflows/ci.yml`):

```bash
# 1. Validate shell scripts
bash -n scripts/run_container.sh scripts/run_web_app.sh

# 2. Run Ruff linter
uv run ruff check .

# 3. Verify Python bytecode compilation
uv run python -m compileall -q core evolution visualization analysis examples tests

# 4. Run unit test suite
uv run pytest -v
```
