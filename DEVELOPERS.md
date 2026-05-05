# DEVELOPERS.md

Guidance for developers working in this repo.

## Repository layout

- `R/` — R package (cmdstanr backend).
- `Python/` — Python package (cmdstanpy backend).
- `commons/stan_files/` — single source of truth for `.stan` files.
- `commons/extdata/` — example data shared by both packages.
- `Python/hbayesdm/common/{stan_files,extdata}` are **symlinks** into `commons/`.
  When you edit a Stan file, edit it in `commons/` (or anywhere — they're the
  same file via the symlink).
- `R/inst/stan_files/` and `R/inst/extdata/` are also symlinks into `commons/`.

## Toolchain

- **Stan backend**: CmdStan via cmdstanr (R) and cmdstanpy (Python). Models
  compile on first use and the binaries are cached next to the `.stan` file.
  No install-time precompilation.
- **R**: ≥ 4.4. Required in `R/DESCRIPTION` (`Depends`).
- **Python**: ≥ 3.13. Required in `Python/pyproject.toml` (`requires-python`).
  Build/dependency manager is
  **uv** with **hatchling** (`Python/pyproject.toml`).
- **Stan syntax**: canonical (`array[N, T] real x`, not `real x[N, T]`; `abs`,
  not `fabs`). If you add a new model, write it in canonical form.

## Common commands

### Python

```bash
cd Python
uv sync --all-groups                # install deps + dev tools
uv run pytest tests/                # full test suite
uv run pytest tests/test_user_facing.py -v   # API-surface tests
uv run sphinx-build -W docs docs/_build/html
```

First-time CmdStan install: `uv run python -c 'import cmdstanpy; cmdstanpy.install_cmdstan()'`.

### R

```bash
cd R
R CMD INSTALL --no-docs --no-test-load .
Rscript -e 'roxygen2::roxygenize(".")'                 # regen man/*.Rd + NAMESPACE
NOT_CRAN=true Rscript -e 'testthat::test_dir("tests/testthat")'
NOT_CRAN=true Rscript -e 'testthat::test_file("tests/testthat/test_user_facing.R")'
Rscript -e 'rmarkdown::render("README.Rmd", output_format = "github_document")'
```

First-time CmdStan install (R): `Rscript -e 'cmdstanr::install_cmdstan()'`.

## API contract

Both packages expose the same fitting interface per task model. The result
object surfaces:

| Attribute (Py) | Slot (R) | Type |
|---|---|---|
| `fit` | `$fit` | `CmdStanMCMC` (or `CmdStanVB` when `vb=True`) |
| `idata` | — | `xarray.DataTree` (arviz) |
| `all_ind_pars` | `$allIndPars` | `pandas.DataFrame` / `data.frame` |
| `par_vals` | `$parVals` | dict / list of posterior draws |
| `model` | `$model` | model name string |
| `model_regressor` | `$modelRegressor` | dict of regressor arrays (only when requested) |

Diagnostics live in `hbayesdm.diagnostics` (Python) and at top level (R):
`rhat`, `print_fit`/`printFit`, `extract_ic`, `hdi`/`HDIofMCMC`,
`plot_hdi`/`plotHDI`, `plot_ind`/`plotInd`.

## Conventions

### Comments

Comments describe **current behavior**, not history. Don't write things like
"used to use rstan", "was a bug in 1.x", "after the refactor", "now uses X
instead of Y". If a reader doesn't know what changed, the comment is noise; if
they do, `git log` / `NEWS.md` is the source of truth.

Single short comments are fine when the *why* is non-obvious (e.g. "Bind
`posterior::rhat` locally so cmdstanr's `$summary()` looks it up by value, not
by name — string lookup would find this function and recurse"). Default to no
comment.

### Stan files

Edit in `commons/stan_files/`. Re-canonicalize with stanc 2.28.2's
`--print-canonical` if you ever paste in legacy syntax.

### R `additional_args`

When defaulting per-model args, use **single-bracket list assignment** so a
`NULL` default is preserved:

```r
args[nm] <- list(additional_args[[nm]])    # keeps NULL
# args[[nm]] <- NULL                        # would *delete* the entry
```

### R `rhat()` name collision

`rhat` in this package shadows `posterior::rhat`. When you need posterior's
version (e.g. inside `cmdstanr::CmdStanFit$summary()`), bind it to a local
variable and pass that — never the string `"rhat"`.

### Python VB path

`cmdstanpy.CmdStanModel.variational()` accepts `inits` only as a perturbation
scalar (`Optional[float]`), not as dict/JSON. The fit dispatch drops dict
inits on the VB branch. `CmdStanVB.stan_variables()` returns variational
*means* by default; pass `mean=False` when you want the n_draws sample matrix
(used in `idata` construction).

### Roxygen / NAMESPACE

`roxygen2::roxygenize(".")` regenerates `R/man/*.Rd` and `R/NAMESPACE` from
the source roxygen tags. Don't hand-edit either output. The shared model doc
template is `R/man-roxygen/model-documentation.R` — edit that to change the
boilerplate that appears on every task model's man page.

## Tests

- Per-task tests in `Python/tests/test_<model>.py` and
  `R/tests/testthat/test_<model>.R` are minimal smoke tests (just verify the
  fit doesn't crash with `niter=10, nwarmup=5, nchain=1`).
- `test_user_facing.{py,R}` cover the result-object API: properties, plotting
  dispatch, diagnostics, IC, HDI helpers, `model_regressor`, `vb=TRUE`.
- R tests skip on CRAN by default; set `NOT_CRAN=true` to run locally.

## Pitfalls

- **Don't run `find /` patterns** — search from `.`.
- **Don't `R CMD check` without `NOT_CRAN=true`** if you want the model-fit
  tests to actually execute; otherwise they all skip.
- **Symlinks in `Python/hbayesdm/common/`**: when collecting wheel contents,
  hatchling follows the symlinks. If you `rm` the symlink, recreate it as
  `ln -s ../../commons/stan_files Python/hbayesdm/common/stan_files`.
- **First-fit compile**: ~30 s per model. Tests that use a never-before-fit
  model will look like they hang. Pre-compile with `cmdstanr::cmdstan_model()`
  / `cmdstanpy.CmdStanModel()` if you need predictable timing.
- **arviz API surface is mid-flux**: `idata`'s exact return type may shift
  across arviz minor versions until the post-1.0 split stabilizes. Test
  against the version in `pyproject.toml` / `requirements.txt`, not against
  whatever's on PyPI.

## Workflows with Claude Code

### When delegating to subagents

This repo has two parallel implementations (R + Python) that share `commons/`.
When asking Claude to make a behavioral change, be explicit about whether it
should propagate to both languages, and whether the Stan files in `commons/`
need editing too. A change to `commons/stan_files/foo.stan` affects every
package that imports it.

For broad codebase exploration ("where is X used across both packages?"),
prefer the `Explore` subagent — it'll search both `R/R/` and
`Python/hbayesdm/` in one pass without burning the main context window.

### Running the dev loop

The fast feedback cycle is:

1. Edit code (Python or R or `commons/`).
2. **Python**: `uv run pytest tests/test_user_facing.py -v` (~30 s, exercises
   plotting + diagnostics + VB + regressors).
3. **R**: `R CMD INSTALL --no-docs --no-test-load .` then
   `NOT_CRAN=true Rscript -e 'testthat::test_file("tests/testthat/test_user_facing.R")'`.
4. If you changed roxygen tags: `Rscript -e 'roxygen2::roxygenize(".")'`
   before reinstalling.

The full per-model smoke suite (`tests/test_*`) is slow (~one cmdstan compile
per model on first run); skip it during iteration and run it before pushing.

## Reference docs

- `hbayesdm_2_0_docs_update_plan.md` — staged plan for finishing the docs work
  (vignettes, migration guide).
- `R/NEWS.md` — user-facing changelog and 1.x → 2.0 migration notes.
