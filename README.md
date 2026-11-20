# DeepGate-FaceRec

**Deep Neural Networks — Homework Assignment: Face Recognition via Deep Embedding Learning**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/package%20manager-uv-8A2BE2.svg)](https://docs.astral.sh/uv/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

> **Assignment Objective:** Move beyond traditional softmax classification. In this homework, you will train a Siamese/Triplet architecture that learns an embedding into the metric space $\mathbb{R}^d$ where faces of the same identity cluster closely together, and different identities are pushed apart by a margin $\alpha$. You will then use these embeddings to simulate an automated **Smart Access Gate**.

---

## 1. Homework Overview & Learning Milestones

Instead of training a neural network to predict a closed-set categorical label (which fails whenever an unseen person arrives), modern biometric systems train an **embedding function** $f_\theta(\mathbf{x}) \in \mathbb{R}^d$.

To guide your implementation, the homework is organized into six progressive milestones:

| Milestone | Topic | Description | Deliverables |
|---|---|---|---|
| **Part 1** | **Exploratory Data Analysis** | Investigate the provided dataset and measure its quality. | `notebooks/01_eda.ipynb` || **Part 2** | **Embedding Architecture** | Adapt a pretrained vision backbone (ResNet/MobileNet) with an embedding projection head and $L_2$ hypersphere normalization. | `src/models.py` |
| **Part 3** | **Identity Batch Sampling** | Implement a $P \times K$ identity sampler (sample $P$ subjects, each with $K$ distinct face images) to enable informative in-batch comparisons. | `src/data.py` |
| **Part 4** | **Online Triplet Mining** | Implement batch-hard and semi-hard negative mining strategies using pairwise distance matrices. | `src/mining.py` |
| **Part 5** | **Triplet Loss from Scratch** | Implement vectorised Triplet Margin Loss: $\mathcal{L}(a, p, n) = \max\left(0, \|\mathbf{z}_a - \mathbf{z}_p\|_2^2 - \|\mathbf{z}_a - \mathbf{z}_n\|_2^2 + \alpha\right)$ without calling `torch.nn.TripletMarginLoss`. | `src/losses.py` |
| **Part 6** | **Smart Gate Simulation** | Enroll staff identities, evaluate True/False Acceptance Rates (TAR/FAR), and simulate real-time gate pass/deny decisions on open-set query faces. | `src/gate.py`, `notebooks/03_smart_gate_demo.ipynb` |
| **Part 7** | **Scientific Report** | Write a conference-style report documenting mathematical formulations, mining comparisons, metric curves, and threshold trade-offs. | `docs/main.pdf` |

---

## Repository Structure

```text
deepgate-facerec/
├── .github/
│   └── workflows/
│       ├── ci.yml                     # Automated linting, formatting, and type checking
│       └── tests.yml                  # Automated pytest suite execution
├── .pre-commit-config.yaml            # Local Git hooks (ruff, trailing-whitespace, YAML checks)
├── pyproject.toml                     # Build configuration, dependencies, and tool settings
├── README.md                          # Project overview, setup, and usage guide
├── scripts/
│    └── download_data.py              # Dataset download and train/val/test split script
├── data/
│   ├── README.md                      # Dataset overview
│   └── split/                         # Disjoint identity splits (ignored by git) and extracted by the `scripts\download_data.py`
│       ├── train/
│       ├── val/
│       └── test/
├── docs/
│   └── report/                            # LaTeX report template and bibliography
│       ├── README.md                      # Report compilation and Overleaf guide
│       ├── deepgate_report.sty            # NeurIPS-adapted LaTeX style package
│       ├── main.tex                       # Report master document
│       └── references.bib                 # Bibliography references
├── src/
│   ├── __init__.py                    # Top-level package exports
│   ├── data.py                        # Image transforms, identity mapping, and PKBatchSampler
│   ├── models.py                      # Pretrained backbone + L2-normalized projection head
│   ├── losses.py                      # TripletMarginLoss using squared Euclidean distance
│   ├── mining.py                      # Pairwise distance matrix, PK masks, and semi-hard mining
│   ├── trainer.py                     # MetricTrainer loop, differential LRs, and checkpointing
│   ├── metrics.py                     # 1:1 verification pair sampling, ROC-AUC, and EER
│   └── gate.py                        # SmartGate 1:N enrollment, verification, and threshold sweeps
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory Data Analysis
│   ├── 02_model_training.ipynb        # Model training, learning curves, and t-SNE comparison
│   └── 03_smart_gate_demo.ipynb       # SmartGate enrollment, FAR/TAR calibration, and probe demos
└── tests/
   ```
---

## 3. Engineering Stack & Tooling

To ensure scientific rigor and clean software engineering habits, this project uses:

### A. Environment & Package Management: [`uv`](https://docs.astral.sh/uv/)

<details>

<summary>Click here to learn more about uv</summary>

### `uv` overview
uv is an ultra-fast Python package and project manager built in Rust by Astral.
It manages Python versions, creates a local .venv virtual environment, and resolves reproducible dependencies from `pyproject.toml`
replacing `pip`, `venv`, `pip-tools`, and `pyenv` with a single tool.

### Most important commands:

```bash
# 1. Create `.venv` and install all runtime & dev dependencies from pyproject.toml
uv sync

# 2. Run commands, tests, or notebooks inside `.venv` (no manual activation required)
uv run pytest
uv run jupyter notebook

# 3. Add or remove packages in pyproject.toml and update uv.lock
uv add <package>
uv add --dev <dev-package>
uv remove <package>

# 4. Download and manage Python interpreter versions
uv python install 3.12

# 5. Upgrade uv to the latest release (standalone installer only)
uv self update

```

> **Tip:** Prefixing commands with `uv run` automatically uses the project's `.venv`, so you do not need to manually run `source .venv/bin/activate` or `.venv\Scripts\activate`.

#### Video Overviews

* [ArjanCodes — This Tool Replaces pip, Poetry, pyenv, and More](https://www.youtube.com/watch?v=qh98qOND6MI) *(high-level introduction to `uv`)*
* [ArjanCodes — UV Just Got A Serious Upgrade](https://www.youtube.com/watch?v=CJJzrtHeLGI) *(modern `uv` workflows & build backends)*
* [Corey Schafer — Python Tutorial: UV](https://www.youtube.com/watch?v=AMdG7IjgSPM) *(step-by-step beginner walkthrough)*

#### Official Documentation

* [uv Overview & Getting Started](https://docs.astral.sh/uv/)
* [Installation Methods & Shell Autocompletion](https://docs.astral.sh/uv/getting-started/installation/)
* [Managing Projects (`uv sync`, `uv add`, `uv run`)](https://docs.astral.sh/uv/guides/projects/)
* [CLI Command Reference](https://docs.astral.sh/uv/reference/cli/)

</details>



### B. Fast Linting & Formatting: [`ruff`](https://docs.astral.sh/ruff/)
* Google-style docstrings and [PEP 8](https://peps.python.org/pep-0008/) standards are enforced. Line length is set to **96 characters**.

<details>

<summary>Click here to learn more about Ruff</summary>

#### What is Ruff?
[Ruff](https://docs.astral.sh/ruff/) is an ultra-fast Python linter and code formatter written in Rust by Astral. It replaces multiple standalone tools—such as Flake8, Black, `isort`, `pydocstyle`, and `pyupgrade`—with a single binary configured directly in `pyproject.toml`.

Enforcing Ruff ensures that all code across `src/` and `tests/` follows consistent formatting, clean import ordering, and safe Python idioms before your changes reach GitHub CI.

#### Essential Commands

```bash
# 1. Check all files for lint, import-order, and docstring violations
uv run ruff check .

# 2. Automatically fix safe issues (e.g., unused imports, import sorting)
uv run ruff check --fix .

# 3. Format code to consistent 88-character style (Black-compatible)
uv run ruff format .

# 4. Verify formatting without modifying files (used in CI)
uv run ruff format --check .

# 5. Explain a specific rule code and see a fix example
uv run ruff rule F401

```

#### Examples of What Ruff Flags

1. **Unused Imports (`F401`) & Unused Variables (`F841`):**
```python
# Ruff complains: `os` imported but unused (F401)
import os
import torch
```


*Fix:* Run `uv run ruff check --fix .` to remove unused imports automatically.
2. **Unsorted Imports (`I001`):**
```python
# Ruff complains: Import block is un-sorted or un-formatted (I001)
from src.models import FaceEmbeddingNet
import torch
```


*Fix:* Ruff groups and alphabetizes standard library, third-party, and local `src` imports when running `ruff check --fix`.
3. **Mutable Default Arguments (`B006`):**
```python
# Ruff complains: Do not use mutable data structures for argument defaults (B006)
def enroll_batch(identities: list[str] = []) -> None: ...
```


*Fix:* Use `identities: list[str] | None = None` and initialize inside the function.
4. **Missing Docstrings in Library Modules (`D100`–`D103`):**
Public classes, and functions in `src/` require concise docstrings (relaxed in `tests/` via `[tool.ruff.lint.per-file-ignores]` in `pyproject.toml`).
5. **Line Length (`E501`):**
Lines exceeding 96 characters are wrapped automatically by `uv run ruff format .`.

### Documentation:

* [Ruff Overview](https://docs.astral.sh/ruff/)
* [Getting Started Tutorial](https://docs.astral.sh/ruff/tutorial/)
* [Linter Rules](https://docs.astral.sh/ruff/linter/)
* [Formatter Guide](https://docs.astral.sh/ruff/formatter/)
* [Configuration Settings](https://docs.astral.sh/ruff/settings/)

#### Video Overviews

* [Corey Schafer — Python Tutorial: Ruff (Linter & Formatter)](https://www.youtube.com/watch?v=828S-DMQog8) *(complete walkthrough of linting, formatting, and IDE integration)*
* [BugBytes — Ultra-Fast Python Linting & Formatting with Ruff](https://www.youtube.com/watch?v=yLYaQq6KbKs) *(practical project setup and CLI usage)*
* [InfoWorld — Ruff: A Faster Linter for Python, Written in Rust](https://www.youtube.com/watch?v=5qY8HwCsSYQ) *(quick overview of Ruff's architecture and speed)*

</details>


### C. Static Type Checking: [`mypy`](https://mypy-lang.org/)
* Enforces type annotations across model inputs, loss functions, to catch mistakes early and to document interfaces concisely.

<details>

<summary>Click here to learn more about mypy</summary>

#### What is Mypy?
[Mypy](https://mypy.readthedocs.io/) is a static type checker for Python. Using [PEP 484](https://peps.python.org/pep-0484/) type annotations (such as `torch.Tensor`, `Path | None`, or `list[str]`), Mypy analyzes your code *without running it* to catch `None`-dereferences path types, and invalid return types before they fail at runtime or in GitHub CI.

#### Essential Commands

```bash
# 1. Type-check the entire `src/` package using settings in pyproject.toml
uv run mypy src

# 2. Type-check a single module while iterating
uv run mypy src/gate.py

# 3. Run with strict type-checking flags enabled
uv run mypy --strict src

# 4. Show error codes (e.g., [union-attr], [arg-type]) to look up fixes
uv run mypy --show-error-codes src

```

#### Examples of What Mypy Flags & How to Fix Them

1. **Accessing Attributes on `None` / Union Types (`[union-attr]`):**
When a variable is typed as `T | None`, Mypy blocks attribute access until `None` is ruled out.
```python
# Mypy error: Item "None" of "Tensor | None" has no attribute "shape" [union-attr]
def gallery_size(embeddings: torch.Tensor | None) -> int:
    return embeddings.shape[0]
```


*Fix:* Narrow the type with an explicit check or `assert` before accessing attributes:
```python
def gallery_size(embeddings: torch.Tensor | None) -> int:
    if embeddings is None:
        raise RuntimeError("Gallery is empty.")
    return embeddings.shape[0]
```


2. **Returning `Any` from Typed Functions (`[no-any-return]`):**
Third-party helpers (such as `torchvision.transforms.Compose`) are typed as returning `Any`, which violates a stricter return annotation like `-> torch.Tensor`.
```python
# Mypy error: Returning Any from function declared to return "Tensor" [no-any-return]
def preprocess(img: Image.Image, transform: transforms.Compose) -> torch.Tensor:
    return transform(img).unsqueeze(0)
```


*Fix:* Narrow the type with `isinstance(...)` so Mypy can verify the return type:
```python
def preprocess(img: Image.Image, transform: transforms.Compose) -> torch.Tensor:
    tensor = transform(img)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}.")
    return tensor.unsqueeze(0)
```


3. **Incompatible Argument Types (`[arg-type]`):**
Passing a `tuple` or `Path | None` into a parameter annotated as `list[Path]` or `Path` triggers `[arg-type]`.
```python
# Mypy error: Argument 1 to "load_paths" has incompatible type "tuple[Path, ...]"; expected "list[Path]" [arg-type]
def load_paths(paths: list[Path]) -> int:
    return len(paths)


load_paths((Path("a.jpg"), Path("b.jpg")))
```


*Fix:* Annotate read-only collection parameters with `collections.abc.Sequence`:
```python
from collections.abc import Sequence


def load_paths(paths: Sequence[Path]) -> int:
    return len(paths)
```


4. **Incompatible Variable Reassignment (`[assignment]`):**
Mypy infers a variable's type from its first assignment; assigning a different type later fails.
```python
# Mypy error: Incompatible types in assignment (expression has type "device", variable has type "str") [assignment]
device = "cpu"
device = torch.device("cuda")
```


*Fix:* Declare the union type explicitly upfront (or keep a single consistent type):
```python
device: str | torch.device = "cpu"
device = torch.device("cuda")
```


5. **Missing Library Type Stubs (`[import-untyped]`):**
Third-party libraries without bundled type annotations raise stub errors.
```python
# Mypy error: Library stubs not installed for "yaml" [import-untyped]
import yaml
```


*Fix:* Install the official `types-*` stub package (`uv add --dev types-PyYAML`) or ignore missing imports for that library in `pyproject.toml`.

> **Tip on Silencing False Positives:** If a third-party library genuinely lacks accurate types on a single line, suppress only the specific error code with `# type: ignore[code]` (for example, `# type: ignore[attr-defined]`) rather than a blanket `# type: ignore` so real bugs on the same line are still caught.

#### Documentation:

* [Mypy Overview](https://mypy.readthedocs.io/)
* [Python 3 Type Hints Cheat Sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)
* [Error Codes Guide](https://mypy.readthedocs.io/en/stable/error_codes.html)
* [Default Error Codes Reference](https://mypy.readthedocs.io/en/stable/error_code_list.html)


#### Video Overviews

* [Corey Schafer — Type Hinting vs Type Checking vs Data Validation](https://www.youtube.com/watch?v=fM4O9bModsE) *(explains how static checking with Mypy differs from runtime validation)*
* [Corey Schafer — Type Hints: From Basic Annotations to Advanced Generics](https://www.youtube.com/watch?v=RwH2UzC2rIo) *(covers annotations, unions, generics, and modern typing patterns)*
* [Carberra — Static Type Checking with Mypy](https://www.youtube.com/watch?v=9gNnhNxra3E) *(step-by-step walkthrough of configuring and running Mypy)*


</details>


### D. Automated Quality Gate: [`pre-commit`](https://pre-commit.com/)
* Automatically validates formatting, type contracts, and tests before any commit is finalized.

<details>

<summary> Click here to learn more about pre-commit </summary>

### `pre-commit` overview

Git supports **hooks**—scripts that run automatically when certain Git events occur, such as right before a commit is recorded. [`pre-commit`](https://pre-commit.com/) is a framework that manages and runs these checks automatically in isolated environments so every contributor enforces the same quality standards without manual effort.

Once enabled, **every `git commit` will run commands defined in `.pre-commit-config.yaml`, like `ruff format`**. If a check fails or auto-formats a file, the commit is paused so you can review and stage the fix—preventing style drift, broken configuration files, or accidental checkpoint uploads before they reach GitHub CI.


### Checks supported within the current project

1. **File Hygiene & Large-File Guard (`pre-commit-hooks`):**
* `trailing-whitespace` & `end-of-file-fixer`: Strips trailing spaces and ensures every file ends with a single newline.
* `check-yaml` & `check-toml`: Validates syntax in YAML and TOML files (such as `pyproject.toml` and `.github/workflows/`).
* `check-added-large-files` (`--maxkb=5000`): Blocks staging files larger than **5 MB**, preventing accidental commits of model checkpoints (`.pt`), datasets, or `.zip` archives.


2. **Linting & Formatting (`ruff-pre-commit` on `src/` and `tests/`):**
* `ruff check (auto-fix)`: Checks Python files against the rules in `pyproject.toml` and automatically fixes safe lint issues.
* `ruff format`: Enforces consistent line formatting across `src/` and `tests/`.

> **Workflow Tip:** When an auto-fixing hook (`ruff`, `ruff-format`, `trailing-whitespace`, or `end-of-file-fixer`) modifies a file during `git commit`, Git aborts the commit so you can inspect the diff. Stage the updated files (`git add -u`) and run `git commit` again.

### Documentation:**

* [pre-commit Official Guide](https://pre-commit.com/)
* [Built-in File Hooks](https://github.com/pre-commit/pre-commit-hooks)
* [Ruff pre-commit Integration](https://docs.astral.sh/ruff/integrations/#pre-commit)

</details>

---

### F. Continuous Integration with GitHub Actions

<details>

<summary> Click here to learn more about Guthub Actions </summary>


#### What is GitHub Actions?
[GitHub Actions](https://docs.github.com/en/actions) is GitHub's built-in Continuous Integration (CI) platform. Whenever you push commits or open a pull request, GitHub spins up a clean Linux runner (`ubuntu-latest`) and executes the automated pipelines defined in `.github/workflows/*.yml`.

This guarantees that every submission is tested in an identical, reproducible environment—catching formatting mistakes, type errors, or failing unit tests automatically and displaying green/red status checks directly on your commits and pull requests.

### Where to Find Results

#### 1. GitHub Actions CI Results (Linting, Type Checks & Tests)

- **Actions Tab (Full Logs & Coverage):**
  1. Open your repository on GitHub and click the **Actions** tab in the top navigation bar.
  2. Select **Code-quality** or **Unit Tests** from the left sidebar, then click the latest workflow run.
  3. Click a job (**Lint & Type Check** or **Run Unit Tests & Coverage**) and expand any step to view Ruff diagnostics, Mypy type errors, or the `pytest` terminal summary showing test pass/fail counts and line-by-line coverage (`Missing` column).
- **Pull Requests:** Scroll to the **Checks** panel at the bottom of a Pull Request and click **Details** next to any check to inspect its output.
- **Commit History:** Click the status icon (**✓** green check or **✕** red cross) next to any commit hash in the repository's commit list to view the status of each workflow run for that revision.


#### Reproducing CI Checks Locally

Before pushing to GitHub, you can run the exact same commands executed by our two CI workflows locally:
```bash
# 1. Code-quality workflow checks (formatting, linting, and static types)
uv run ruff format --check .
uv run ruff check .
uv run mypy src/

# 2. Unit Tests workflow check (pytest + missing-line coverage report)
uv run pytest --cov=src --cov-report=term-missing --cov-report=xml
```

#### Documentation:
* [GitHub Actions Overview](https://docs.github.com/en/actions/get-started/understand-github-actions)
* [Workflow YAML Syntax](https://docs.github.com/en/actions/writing-workflows/workflow-syntax-for-github-actions)
* [Building & Testing Python](https://docs.github.com/en/actions/use-cases-and-examples/building-and-testing/building-and-testing-python)
* [`astral-sh/setup-uv` Action](https://github.com/astral-sh/setup-uv)

#### Video Overviews

* [Timnology — Setup a Python Code Quality CI Pipeline with UV & Ruff](https://www.youtube.com/watch?v=Y6D2XaFV3Cc) *(walks through configuring `uv`, Ruff, type checking, and `pytest` coverage in GitHub Actions)*
* [TechWorld with Nana — GitHub Actions Tutorial: Basic Concepts and CI/CD Pipeline](https://www.youtube.com/watch?v=R8_veQiYBjI) *(explains triggers, runners, jobs, and YAML workflow structure from scratch)*



</details>


### Prerequisites
* **uv**: Install via Astral's official script:
  ```bash
  # macOS / Linux
  curl -LsSf https://astral.sh/uv/install.sh | sh

  # Windows (PowerShell)
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```


### Local Setup in 4 Commands

1. **Clone the repository**:
   ```bash
   git clone https://github.com/adendek/DeepGate-FaceRec.git
   cd DeepGate-FaceRec
   ```

2. **Sync the virtual environment and all dependencies**:
   ```bash
   uv sync --all-extras --group dev
   ```

<details>

<summary>Click here to learn more about uv sync </summary>

* **Reproducible Environment:** Updates (or creates) the virtual environment in this folder (`.venv`) with this project's exact Python and dependency versions (declared in `pyproject.toml` and pinned for reproducibility in `uv.lock`).
* **Included Groups:** `--all-extras` installs optional runtime features, and `--group dev` installs testing, linting, and notebook tools (`pytest`, `ruff`, `mypy`, `ipykernel`).
* **Disk Space:** Uses ~6 GB of disk space for `.venv` (plus `uv`'s global cache).

</details>

3. **Install Git pre-commit hooks**:
   ```bash
   uv run pre-commit install --install-hooks
   ```

3. Run all checks manually across every file in the repository:
```bash
uv run pre-commit run --all-files
```


4. **Download and prepare the face dataset**:
   ```bash
   uv run python data/download_data.py
   ```

The expected zip hash:

```
sha256:6bd700f9eebe5dad3561bcad79fd949b6db25b3759974faef16d54fe5f1cdefd
```

---

## 5. Development & Testing Command Reference

All commands must be executed using `uv run` to guarantee they run inside the isolated environment:

| Task | Command | Description |
|---|---|---|
| **Run All Unit Tests** | `uv run pytest` | Runs all unit tests with verbose test reporting |
| **Test Specific Module** | `uv run pytest tests/test_loss.py` | Runs tests only for your Triplet Loss implementation |
| **Check Test Coverage** | `uv run pytest --cov=src` | Computes test coverage across your implementation |
| **Lint Code** | `uv run ruff check .` | Checks for linting issues and code style bugs |
| **Auto-fix Lint** | `uv run ruff check . --fix` | Automatically fixes safe linting errors |
| **Format Code** | `uv run ruff format .` | Formats all files according to project style |
| **Type Check** | `uv run mypy src/` | Runs strict static type validation |
| **Run Pre-Commit** | `uv run pre-commit run --all-files` | Manually triggers all pre-commit hooks |
| **Start Notebook** | `uv run jupyter lab` | Launches JupyterLab to run the visual demos |

---

## 6. Written Report & Overleaf Template

Alongside the working code, you must submit a concise, conference-style scientific report (3–4 pages). The report format is adapted from the NeurIPS conference style.

### Direct Overleaf Template Link
You can open and copy the official course report template directly in your browser:
**[Open DeepGate-FaceRec Report Template on Overleaf](https://www.overleaf.com/project/6ae88d3923251984d11a96a5/share#2edcd42a0d900d6fa592d1db61885d531af4d654656f98ee)**

*(Inside Overleaf, click **Menu** $\rightarrow$ **Copy Project** to create your own editable team copy).*

### Required Report Content
1. **Mathematical Formulations**: State the exact loss formula, distance metric, and how gradients backpropagate through normalized embeddings.
2. **Mining Analysis**: Empirical comparison between Random, Semi-Hard, and Batch-Hard mining (loss convergence, percentage of non-zero loss triplets).
3. **Verification Performance**: ROC curve, AUC, Equal Error Rate (EER), and distance distribution histograms for positive vs. negative pairs.
4. **Smart Gate Decision Trade-offs**: Analysis of False Acceptance Rate (FAR) vs. False Rejection Rate (FRR) as a function of the threshold $\tau$, explaining the security implications of choosing conservative vs. permissive thresholds.

See [`docs/report/README.md`](docs/report/README.md) for full compilation instructions, styling details, and formatting tips.

---

## 7. Submission Guidelines

Your final submission consists of two deliverables:

1. **Codebase**:
   - All tests in `tests/` pass (`uv run pytest`).
   - Clean code hygiene passing Ruff and Mypy (`uv run ruff check .` and `uv run mypy src/`).
   - Push your branch to the course GitHub repository.

2. **Compiled Report (`report.pdf`)**:
   - Compiled PDF from the Overleaf template placed in `docs/report.pdf` together with the codebase.

----

<div align="center">

![Academic Integrity](https://img.shields.io/badge/ACADEMIC_INTEGRITY-NO_LLM_CODE_GENERATION-dc2626?style=for-the-badge)

### **Generative AI & LLM Usage Policy**

**Using generative AI or LLMs to write assignment code is strictly forbidden.**

**Permitted Exception (Plotting Boilerplate Only):**
You may use LLMs to generate `matplotlib` / `seaborn` plotting and styling calls, provided that **all code preparing, filtering, or transforming the underlying data is written without AI**.

</div>

----
