# 2.0 docs update plan

After the 2.0 stack migration (pystan/rstan → cmdstanpy/cmdstanr, py 3.13+, R 4.4+), code passes smoke tests but documentation still describes the 1.x toolchain. This plan stages the doc work into two passes so the cheap-but-load-bearing edits ship first and the expensive vignette rewrite waits until the 2.0 user-facing API is stable (covariates + growth modeling).

## Pass 1 — load-bearing source updates (~20 min)

Update the small number of *source* files that everything else is generated or stamped from, then regenerate. Don't hand-edit `R/man/*.Rd` — those are roxygen2 output.

### R

1. **`R/man-roxygen/model-documentation.R`** — the shared roxygen template applied to every task model. Update:
   - `@return` description of `fit`: `class 'stanfit'` → `class 'CmdStanMCMC' (or 'CmdStanVB' if vb=TRUE)`.
   - Any prose about `rstan::extract()`, `rstan_options`, etc.
   - One line on the new compile-on-first-use behavior so users aren't surprised by the ~30s wait on first fit.

2. **Per-task `.R` files** — sweep `@return` blocks that hand-rolled `\code{'stanfit'}` instead of using the template. Quick `grep -l "stanfit" R/R/*.R` and bulk-edit.

3. **`R/README.Rmd`** — installation section. Replace rstan/StanHeaders setup with:
   ```r
   install.packages("cmdstanr", repos = c("https://stan-dev.r-universe.dev", getOption("repos")))
   cmdstanr::install_cmdstan()
   install.packages("hBayesDM")
   ```
   Re-knit to `R/README.md`.

4. **`R/NEWS.md`** — add a 2.0 entry: backend swap, R≥4.4, removal of install-time precompile, breaking changes (e.g., `fit` is now CmdStanMCMC, not stanfit; methods like `extract()` no longer apply directly).

5. **`R/cran-comments.md`** — refresh to describe the 2.0 submission story (CmdStan as a system dep, no compiled C++ in the package).

6. **Regenerate** — `roxygen2::roxygenize("R")` (or `devtools::document()` once devtools installs cleanly). This rewrites all 68 `R/man/*.Rd` files from the updated roxygen sources. Verify with `R CMD check R/`.

### Python

7. **`Python/README.rst`** — installation section. Replace pystan setup with `uv add hbayesdm` (or `pip install hbayesdm`) plus the `cmdstanpy.install_cmdstan()` one-liner. Mention python ≥ 3.13.

8. **`Python/docs/requirements.txt`** — repin to current sphinx + sphinx-autodoc-typehints + sphinx-rtd-theme that already live in `pyproject.toml [dependency-groups].dev`. Or delete the file and switch the docs build to `uv sync --group dev` so versions stay aligned.

9. **`Python/docs/index.rst`, `models.rst`, `diagnostics.rst`** — quick prose pass:
   - Wherever `pystan`, `StanModel`, `sampling()`, `extract()` are described, replace with cmdstanpy equivalents.
   - The `fit` attribute is now a `CmdStanMCMC` (or `CmdStanVB`); the `idata` attribute on `TaskModel` exposes an `arviz.InferenceData` for diagnostics.
   - Update python version classifiers if mentioned in prose.

10. **`.readthedocs.yml`** — bump to py 3.13 build, switch dep install command to uv if RTD supports it (else `pip install .[dev]` / `pip install -r docs/requirements.txt`).

### Top-level

11. **`README.md`** — light pass; current version was clean on grep but worth re-reading once the R/Python READMEs are updated so the top-level pitch matches.

### Pass 1 acceptance

- `R CMD check R/` passes with no doc-related WARNINGs/NOTEs.
- `cd Python && uv run sphinx-build -W docs docs/_build/html` builds clean.
- README install instructions, copy-pasted, get a fresh user from zero to a successful `ra_prospect("example", ...)` fit.

## Pass 2 — vignettes (with the 2.0 user-facing work)

Defer until the covariate + growth APIs are landed. Rewriting the tutorials twice (once for the backend swap, again for the new modeling features) is wasted churn.

### R vignettes

- **`R/vignettes/getting_started.Rmd`** — full rewrite around cmdstanr idioms. Add a section on the first-fit compile cost and the `precompile_models()` helper if we ship one (see `hbayesdm_covariate_plus_growth_plan.md`).
- **`R/vignettes/hgf_tutorial.Rmd`** — same, plus updates for any HGF-specific API changes that shake out of the 2.0 work.

### Python docs

- New tutorial(s) under `Python/docs/` mirroring the R vignettes' content, since 1.x didn't have prose tutorials at parity. Probably one "getting started" notebook and one task-specific deep dive.

### Pass 2 acceptance

- Vignettes build under `R CMD build` without errors.
- Each vignette has been run end-to-end on a clean machine and the printed outputs match the rendered narrative.
- Both languages cover: install → load example data → fit → diagnostics → posterior plots → model comparison.

## Cross-cutting items to track

- **Migration guide**: a single `MIGRATION_1_TO_2.md` (or a section in `NEWS.md`) listing breaking changes for existing users — fit class change, removed VB return shape, extract API differences, install steps. Worth writing once Pass 1 is done so it can link to authoritative sections in the regenerated docs.
- **Travis → GitHub Actions**: out of scope for docs but referenced from CI badges in READMEs. Once GH Actions exists, swap the badge URLs in Pass 1's README edits.
- **arviz `MigrationWarning`**: arviz 0.21 is mid-transition to xarray DataTree. Diagnostics docs should not pretend this is settled — add a one-line note that the `idata` accessor's exact return type may shift across arviz minor versions until they stabilize.
