# Adding `lm`-style covariates to hBayesDM: implementation plan

## 1. What changes, conceptually

Today, every hBayesDM model is "intercept-only" at the group level. For a parameter `theta` (after Matt-trick reparameterization on the `_pr` scale):

```
theta_pr[i] = mu_pr + sigma * z[i]
```

You want to allow:

```
theta_pr[i] = X[i,] %*% gamma + sigma * z[i]
```

where `X` is a design matrix built from a user-supplied formula like `theta ~ 1 + age + group`. When the formula is `~ 1`, this collapses exactly to the current model (intercept = `mu_pr`, single column of ones), so the extension is **strictly backward-compatible** if we make the formula default to `~ 1` for every parameter.

The good news: the codebase is well-suited to this change. The `commons/` directory is the source of truth — YAML model specs plus a Stan template that gets converted into per-language wrappers via `convert-to-r.py` / `convert-to-py.py`. The covariate machinery is mostly **generic** (parse formulas, build design matrices, pass `X`/`D`/`D_start`/`D_end` to Stan); the only per-model work is editing the Stan file's parameter block and the linear predictor lines.

## 2. Mapping your prototype onto the hBayesDM idiom

Your prototype contains two orthogonal extensions glued together: (a) **covariates on person-level parameters**, and (b) **longitudinal sessions with growth curves**. For the hBayesDM extension you only want (a). Sessions/longitudinal are a separate (much bigger) extension that touches every preprocess function and every Stan file's data block — out of scope for this plan, though I'll note the design decisions that keep the door open.

For the covariate-only case, the Stan idiom collapses to:

```stan
data {
  int<lower=1> N;
  int<lower=1> D;                 // total number of design columns across all params
  array[N, D] real X;             // design matrix (single-session)
  array[P] int D_start;           // P = number of person-level parameters
  array[P] int D_end;
  // ... existing behavioral data ...
}
parameters {
  vector[D] gamma;                // all linear-predictor coefs concatenated
  vector<lower=0>[P] sigma;
  matrix[P, N] z_pr;              // raw subject-level (Matt trick)
}
transformed parameters {
  matrix[N, P] theta_pr;
  vector<lower=L1, upper=U1>[N] Arew;   // example bounded transform
  // ...
  for (p in 1:P)
    theta_pr[, p] = to_matrix(X[, D_start[p]:D_end[p]]) * gamma[D_start[p]:D_end[p]]
                  + sigma[p] * to_vector(z_pr[p, :]);
  Arew = Phi_approx(theta_pr[, 1]);   // unchanged transforms
  // ...
}
```

This is a cleaner version of the pattern in your `igt_orl_playpass_conditional_growth.stan` — no time slope, no session dimension — but the key construct (`X[, D_start[p]:D_end[p]] * gamma[D_start[p]:D_end[p]]`) is identical.

## 3. Implementation phases

I'd stage this as four phases. Each is independently testable and shippable.

### Phase 1 — Generic infrastructure (R + Python)

**Goal:** put covariate parsing, design-matrix construction, and the new Stan data fields in the engine, behind a feature flag that defaults to "intercept-only" (current behavior).

R side:

- Add a `parse_formula()` helper to a new file `R/R/formula_utils.R`. Same idea as your `parse_formula` in `utils.R` but lifted into a package function with proper roxygen, error messages referencing parameter names from the model spec, and stricter validation.
- Add a `build_design_matrix()` helper in the same file. It takes a named list of formulas, a per-subject covariate `data.frame`, and the parameter list; returns `list(X, D, D_start, D_end)`. Two important wrinkles relative to your prototype: (i) factor handling — `model.matrix` will dummy-code factors, so column counts per parameter aren't known until covariate data is in hand; (ii) collinearity check — warn if any design matrix column is constant or has rank < ncol.
- Modify `hBayesDM_model.R` to accept a new argument `formulas = NULL` (the list, post-parsing). When not NULL, the engine: (i) extracts a per-subject covariate frame from `raw_data` (one row per subject; numeric columns averaged, factor/character columns taken as the first non-NA — this matches your `covar_data` summarization), (ii) calls `build_design_matrix`, (iii) injects `X, D, D_start, D_end` into `data_list` before passing to Stan, (iv) appends `"gamma"` to the parameter-extraction list `pars`. When `formulas` is NULL, fall through to current behavior unchanged.
- Touch the `init` machinery (`gen_init` closure, lines ~466-493 of `hBayesDM_model.R`). Currently it inits `mu_pr` and `sigma` from plausible parameter values via `qnorm`. With covariates, the equivalent is initializing `gamma[D_start[p]]` (the intercept column) the same way and the rest at zero. Keep `sigma` init logic the same.

Python side: parallel changes in `Python/hbayesdm/base.py`:

- Add a `formula_utils.py` module. Use `formulaic` (preferred — modern, well-maintained, supports `~ 1 + a + b` and factor expansion) or `patsy` (older but widespread) to handle the formula → design matrix conversion. Both libraries match R's `model.matrix` semantics closely enough for this use case.
- In `TaskModel.__init__`, accept `formulas: Optional[Dict[str, str]] = None`. Wire it through `_run` the same way as R: build covariate frame, build design matrix, inject into `data_dict`.

What gets touched in this phase:
- `R/R/hBayesDM_model.R` — add `formulas` arg and the design-matrix block
- `R/R/formula_utils.R` — new file
- `Python/hbayesdm/base.py` — parallel changes
- `Python/hbayesdm/formula_utils.py` — new file
- `commons/templates/R_CODE_TEMPLATE.txt` and `PY_CODE_TEMPLATE.txt` — add `formulas` to the wrapper signatures so codegen produces it for every model

Nothing about Stan files changes yet. Existing models still work.

### Phase 2 — Per-model Stan refactor (one model at a time)

For each model, the Stan file gets edited so that the parameter block uses the design-matrix form. The key insight: **this is mostly mechanical** because the change is local to the `parameters` and `transformed parameters` blocks; the `model` and `generated quantities` blocks (which contain the task-specific likelihood) are untouched.

The diff for `igt_orl.stan` (worked out fully in §4 below):

- Add to `data`: `int<lower=1> D;`, `array[N, D] real X;`, `array[5] int D_start;`, `array[5] int D_end;` (5 parameters).
- Replace `vector[5] mu_pr; vector<lower=0>[5] sigma;` with `vector[D] gamma; vector<lower=0>[5] sigma;`.
- Replace each `mu_pr[k] + sigma[k] * Arew_pr[i]` with `dot_product(to_vector(X[i, D_start[k]:D_end[k]]), gamma[D_start[k]:D_end[k]]) + sigma[k] * Arew_pr[i]`.
- Adjust priors: `gamma ~ normal(0, 1);` (or finer-grained per-parameter, but keep simple for now).
- Generated quantities `mu_Arew` etc.: drop them, OR keep them for backward compatibility by computing them at the mean of `X` (i.e. `Phi_approx(mean(X[, D_start[1]:D_end[1]] * gamma[D_start[1]:D_end[1]]))`). I'd drop them in v1 and add a `posterior_means()` post-hoc helper instead — more flexible and avoids baking covariate-conditional decisions into every model.

Each model's YAML doesn't need to change — the parameter list, regressors, postpreds, etc. all stay the same. The Stan file edit is the entire per-model effort.

### Phase 3 — Per-model R/Python wrapper updates

Once a model's Stan file is updated, regenerate its R/Python wrapper from YAML (`commons/generate-codes.sh`). The templates in Phase 1 added `formulas` to the function signature, so this gives every model an `lm`-style entry point automatically.

For users, the migration is: pass `formulas = list(Arew = ~ 1, Apun = ~ 1, ...)` to get the old behavior, or pass non-trivial formulas to get the new behavior. For convenience, when `formulas = NULL` the engine should auto-fill all formulas as `~ 1` so the default call still works.

### Phase 4 — Helpers and polish

- `extract_gamma()` post-hoc helper that returns a tidy data.frame / DataFrame of `gamma` draws with column names that match the formula terms (e.g. `Arew_(Intercept)`, `Arew_age`, `Apun_(Intercept)`, ...). This is the equivalent of your `relabel_map` but auto-generated from the design matrix's `colnames(X[[par]])`.
- `predict()` method that takes new covariate values and returns posterior predictive draws of person-level parameters (with optional marginalization over `sigma`, as in your `marginal_mean_growth`).
- Vignette / documentation update with one fully worked example.
- Tests: a small simulated dataset where the true `gamma` is known, and a regression test that the `formulas = NULL` path matches the legacy fits exactly.

## 4. Worked example: `igt_orl` end-to-end

Here's exactly what changes for one model. I'll show Stan, R, and Python.

### 4.1 Stan: `commons/stan_files/igt_orl.stan`

Diff against the current file. Five parameters: `Arew, Apun, K, betaF, betaP` (= P in the generic discussion). Behavioral data block, model block, generated quantities are unchanged except for the `mu_*` outputs.

```stan
#include /pre/license.stan

data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=1, upper=T> Tsubj[N];
  int choice[N, T];
  real outcome[N, T];
  real sign_out[N, T];

  // NEW: covariate machinery
  int<lower=1> D;                  // total design columns across all 5 params
  array[N, D] real X;              // person-level design matrix
  array[5] int D_start;            // start index in gamma for each param
  array[5] int D_end;              // end index in gamma for each param
}

transformed data {
  vector[4] initV;
  initV = rep_vector(0.0, 4);
}

parameters {
  // Group-level coefficients (was: vector[5] mu_pr;)
  vector[D] gamma;
  vector<lower=0>[5] sigma;

  // Subject-level raw parameters (Matt trick) — unchanged
  vector[N] Arew_pr;
  vector[N] Apun_pr;
  vector[N] K_pr;
  vector[N] betaF_pr;
  vector[N] betaP_pr;
}

transformed parameters {
  vector<lower=0, upper=1>[N] Arew;
  vector<lower=0, upper=1>[N] Apun;
  vector<lower=0, upper=5>[N] K;
  vector[N]                   betaF;
  vector[N]                   betaP;

  // Each person-level parameter is X[i, cols_p] * gamma[cols_p] + sigma[p] * z_p[i]
  // Computed via matrix multiplication for vectorization.
  {
    vector[N] lp_Arew  = to_matrix(X[, D_start[1]:D_end[1]]) * gamma[D_start[1]:D_end[1]];
    vector[N] lp_Apun  = to_matrix(X[, D_start[2]:D_end[2]]) * gamma[D_start[2]:D_end[2]];
    vector[N] lp_K     = to_matrix(X[, D_start[3]:D_end[3]]) * gamma[D_start[3]:D_end[3]];
    vector[N] lp_betaF = to_matrix(X[, D_start[4]:D_end[4]]) * gamma[D_start[4]:D_end[4]];
    vector[N] lp_betaP = to_matrix(X[, D_start[5]:D_end[5]]) * gamma[D_start[5]:D_end[5]];

    for (i in 1:N) {
      Arew[i] = Phi_approx(lp_Arew[i]  + sigma[1] * Arew_pr[i]);
      Apun[i] = Phi_approx(lp_Apun[i]  + sigma[2] * Apun_pr[i]);
      K[i]    = Phi_approx(lp_K[i]     + sigma[3] * K_pr[i]) * 5;
      betaF[i] = lp_betaF[i] + sigma[4] * betaF_pr[i];
      betaP[i] = lp_betaP[i] + sigma[5] * betaP_pr[i];
    }
  }
}

model {
  // Priors
  gamma ~ normal(0, 1);
  sigma[1:3] ~ normal(0, 0.2);
  sigma[4:5] ~ cauchy(0, 1.0);

  Arew_pr  ~ normal(0, 1);
  Apun_pr  ~ normal(0, 1);
  K_pr     ~ normal(0, 1);
  betaF_pr ~ normal(0, 1);
  betaP_pr ~ normal(0, 1);

  // ... (likelihood block UNCHANGED — everything from `for (i in 1:N) { ... }` onward) ...
}

generated quantities {
  // For log likelihood + posterior predictives — UNCHANGED
  real log_lik[N];
  real y_pred[N, T];

  // Drop the old mu_Arew, mu_Apun, mu_K, mu_betaF, mu_betaP scalars.
  // Users get coefficient-level summaries from `gamma` directly,
  // and a post-hoc helper computes "marginal mean" parameters at chosen X values.

  // ... rest unchanged ...
}
```

Two notes on this diff:

- The `to_matrix(X[, a:b]) * gamma[a:b]` is a single matrix-vector multiply of size `N × k`, computed once per leapfrog step. This is faster than the per-subject `dot_product` form in your prototype because Stan's autodiff handles the matmul more efficiently.
- Dropping `mu_Arew` etc. is a deliberate API break for v1. They're only meaningful in the intercept-only case, and computing them with covariates requires choosing a reference X value, which is a user decision. The post-hoc `posterior_means()` helper described in Phase 4 handles this cleanly.

### 4.2 R: per-subject covariate extraction

The bulk of the new R code lives in `formula_utils.R` and `hBayesDM_model.R`. Here's what `formula_utils.R` looks like:

```r
#' Parse a list of formulas describing covariate effects on person-level parameters
#'
#' @param formulas Named list of one-sided formulas, e.g.
#'   \code{list(Arew = ~ 1 + age + group, Apun = ~ 1)}.
#'   Names must be a subset of \code{parameters}; missing entries default to \code{~ 1}.
#' @param parameters Character vector of allowed parameter names (from the model spec).
#' @return Named list of formulas, one per parameter, in the order of \code{parameters}.
#' @keywords internal
validate_formulas <- function(formulas, parameters) {
  if (is.null(formulas)) {
    formulas <- list()
  }
  if (!is.list(formulas)) {
    stop("`formulas` must be a named list of one-sided formulas.")
  }
  bad_names <- setdiff(names(formulas), parameters)
  if (length(bad_names) > 0) {
    stop("Unknown parameter(s) in formulas: ",
         paste(bad_names, collapse = ", "), ". ",
         "Allowed: ", paste(parameters, collapse = ", "))
  }
  # Default any missing parameter to intercept-only
  out <- list()
  for (p in parameters) {
    f <- if (p %in% names(formulas)) formulas[[p]] else as.formula("~ 1")
    if (!inherits(f, "formula") || length(f) != 2L) {
      stop("Formula for `", p, "` must be a one-sided formula like `~ 1 + age`.")
    }
    out[[p]] <- f
  }
  out
}

#' Build per-subject design matrices and pack them into Stan-ready arrays
#'
#' @param formulas Validated named list (from \code{validate_formulas}).
#' @param subj_covars data.frame with one row per subject, in subject order.
#' @return List with elements X (matrix N x D), D (int), D_start, D_end (named int vectors).
#' @keywords internal
build_design_matrices <- function(formulas, subj_covars) {
  Xs <- lapply(formulas, function(f) {
    mm <- stats::model.matrix(f, data = subj_covars)
    if (any(is.na(mm))) {
      stop("Design matrix contains NAs after model.matrix(). ",
           "Check that covariates have no missing values for the included subjects.")
    }
    mm
  })
  D_per <- vapply(Xs, ncol, integer(1))
  D_end <- cumsum(D_per)
  D_start <- c(1L, D_end[-length(D_end)] + 1L)
  names(D_start) <- names(D_end) <- names(formulas)
  X_full <- do.call(cbind, Xs)
  # Useful for downstream labeling — column names like "Arew_(Intercept)", "Arew_age", ...
  colnames(X_full) <- unlist(lapply(names(Xs), function(p) {
    paste(p, colnames(Xs[[p]]), sep = "_")
  }))
  list(
    X = X_full,
    D = sum(D_per),
    D_start = unname(D_start),
    D_end = unname(D_end),
    coef_names = colnames(X_full)
  )
}

#' Aggregate raw_data to one row per subject for covariate extraction
#' @keywords internal
extract_subj_covars <- function(raw_data, subjs, covar_cols) {
  raw_data <- as.data.frame(raw_data)
  if (length(covar_cols) == 0) {
    return(data.frame(subjid = subjs))
  }
  # Numeric: subject mean (handles within-subject jitter / per-trial repeats)
  # Non-numeric: first non-NA value
  agg <- lapply(subjs, function(s) {
    sub <- raw_data[raw_data$subjid == s, , drop = FALSE]
    out <- list(subjid = s)
    for (col in covar_cols) {
      v <- sub[[col]]
      if (is.numeric(v)) {
        out[[col]] <- mean(v, na.rm = TRUE)
      } else {
        nn <- v[!is.na(v)]
        out[[col]] <- if (length(nn)) nn[1] else NA
      }
    }
    out
  })
  do.call(rbind.data.frame, c(agg, stringsAsFactors = FALSE))
}
```

And the patch to `hBayesDM_model.R` (just the relevant insertion, not the whole file):

```r
# ... existing code through `general_info <- list(...)` ...

# NEW: validate formulas and pull covariate column names from the union of RHS terms
formulas <- validate_formulas(formulas, names(parameters))
covar_cols <- unique(unlist(lapply(formulas, function(f) all.vars(f))))
# Filter out anything that's actually a behavioral / required column
covar_cols <- setdiff(covar_cols, tolower(gsub("_", "", data_columns, fixed = TRUE)))

# Existing preprocess call
data_list <- preprocess_func(raw_data, general_info, ...)

# NEW: build design matrix and merge into data_list
subj_covars <- extract_subj_covars(raw_data, subjs, covar_cols)
dm <- build_design_matrices(formulas, subj_covars)
data_list$X        <- dm$X
data_list$D        <- dm$D
data_list$D_start  <- dm$D_start
data_list$D_end    <- dm$D_end

# Save coefficient names for downstream labeling
attr(data_list, "gamma_coef_names") <- dm$coef_names

# ... existing pars assembly, with `mu_*` removed and "gamma" added ...
pars <- character()
if (model_type != "single") {
  pars <- c(pars, "gamma", "sigma")
}
pars <- c(pars, names(parameters), "log_lik")
# ... rest unchanged ...
```

And the user's call:

```r
library(hBayesDM)

# Old (still works — formulas defaults to NULL → all ~ 1)
fit_old <- igt_orl(data = "example", niter = 2000, nchain = 4)

# New
fit <- igt_orl(
  data     = my_igt_data,        # must contain age, group columns at subject level
  formulas = list(
    Arew  = ~ 1 + age + group,
    Apun  = ~ 1 + age + group,
    K     = ~ 1,
    betaF = ~ 1 + group,
    betaP = ~ 1
  ),
  niter = 2000, nchain = 4
)

# Inspect coefficients with sensible names
gamma_draws <- rstan::extract(fit$fit, "gamma")$gamma  # iter x D matrix
colnames(gamma_draws) <- attr(fit$data_list, "gamma_coef_names")
# e.g. columns: Arew_(Intercept), Arew_age, Arew_groupB, Apun_(Intercept), ...
```

### 4.3 Python: equivalent

In `Python/hbayesdm/formula_utils.py`:

```python
"""Formula parsing and design-matrix construction for hBayesDM."""
from collections import OrderedDict
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from formulaic import Formula
    _BACKEND = "formulaic"
except ImportError:
    from patsy import dmatrix
    _BACKEND = "patsy"


def validate_formulas(
    formulas: Optional[Mapping[str, str]],
    parameters: Sequence[str],
) -> "OrderedDict[str, str]":
    """Fill in missing entries with '~ 1' and reject unknown keys."""
    formulas = dict(formulas or {})
    bad = set(formulas) - set(parameters)
    if bad:
        raise ValueError(
            f"Unknown parameter(s) in formulas: {sorted(bad)}. "
            f"Allowed: {list(parameters)}"
        )
    out = OrderedDict()
    for p in parameters:
        f = formulas.get(p, "~ 1")
        if not isinstance(f, str) or "~" not in f:
            raise ValueError(
                f"Formula for `{p}` must be a string like '~ 1 + age', got {f!r}"
            )
        out[p] = f
    return out


def _design_matrix(formula: str, data: pd.DataFrame) -> pd.DataFrame:
    """Single-formula design matrix; returns a DataFrame so we keep column names."""
    if _BACKEND == "formulaic":
        # formulaic accepts '~ rhs' but wants 'lhs ~ rhs' for some methods;
        # we treat as model-matrix-only by stripping the LHS.
        rhs = formula.split("~", 1)[1].strip()
        return Formula(f"~ {rhs}").get_model_matrix(data)
    else:  # patsy
        return pd.DataFrame(dmatrix(formula, data, return_type="dataframe"))


def build_design_matrices(
    formulas: "OrderedDict[str, str]",
    subj_covars: pd.DataFrame,
) -> Dict:
    """Stack per-parameter design matrices into a single (N, D) array."""
    blocks = OrderedDict()
    for p, f in formulas.items():
        mm = _design_matrix(f, subj_covars)
        if mm.isna().any().any():
            raise ValueError(
                f"Design matrix for `{p}` contains NAs. "
                "Check covariate completeness."
            )
        blocks[p] = mm

    d_per = np.array([mm.shape[1] for mm in blocks.values()], dtype=int)
    d_end = np.cumsum(d_per)
    d_start = np.concatenate([[1], d_end[:-1] + 1])  # 1-indexed for Stan
    X = np.concatenate([mm.to_numpy() for mm in blocks.values()], axis=1)
    coef_names = [f"{p}_{c}" for p, mm in blocks.items() for c in mm.columns]

    return {
        "X": X,
        "D": int(X.shape[1]),
        "D_start": d_start.astype(int),
        "D_end": d_end.astype(int),
        "coef_names": coef_names,
    }


def extract_subj_covars(
    raw_data: pd.DataFrame,
    subjs: Sequence,
    covar_cols: Sequence[str],
) -> pd.DataFrame:
    """One row per subject. Numeric → mean, others → first non-NA."""
    if not covar_cols:
        return pd.DataFrame({"subjid": list(subjs)})
    rows = []
    for s in subjs:
        sub = raw_data.loc[raw_data["subjid"] == s]
        row = {"subjid": s}
        for col in covar_cols:
            v = sub[col]
            if pd.api.types.is_numeric_dtype(v):
                row[col] = float(v.mean())
            else:
                nn = v.dropna()
                row[col] = nn.iloc[0] if len(nn) else None
        rows.append(row)
    return pd.DataFrame(rows)
```

In `Python/hbayesdm/base.py`, the `_run` method gets a few new lines around the existing `_preprocess_func` call:

```python
# In TaskModel.__init__, accept formulas
def __init__(self, ..., formulas: Optional[Dict[str, str]] = None, **kwargs):
    self.__formulas = formulas
    # ... existing code ...

# In _run, after _preprocess_func:
data_dict = self._preprocess_func(raw_data, general_info, additional_args)

# NEW
from hbayesdm.formula_utils import (
    validate_formulas, build_design_matrices, extract_subj_covars,
)
formulas = validate_formulas(self.__formulas, list(self.parameters))
covar_cols = sorted({
    v for f in formulas.values()
    for v in _vars_in_formula(f)  # tiny helper that pulls names from RHS
}) - set(self._get_insensitive_data_columns())
subj_covars = extract_subj_covars(
    raw_data, general_info["subjs"], list(covar_cols)
)
dm = build_design_matrices(formulas, subj_covars)
data_dict["X"]       = dm["X"]
data_dict["D"]       = dm["D"]
data_dict["D_start"] = dm["D_start"]
data_dict["D_end"]   = dm["D_end"]
self._gamma_coef_names = dm["coef_names"]

# In _prepare_pars: replace any "mu_*" with "gamma"
```

User-facing call:

```python
from hbayesdm.models import igt_orl

fit = igt_orl(
    data=my_df,
    formulas={
        "Arew":  "~ 1 + age + group",
        "Apun":  "~ 1 + age + group",
        "K":     "~ 1",
        "betaF": "~ 1 + group",
        "betaP": "~ 1",
    },
    niter=2000, nchain=4,
)

import pandas as pd
gamma_draws = pd.DataFrame(
    fit.par_vals["gamma"], columns=fit._gamma_coef_names
)
```

## 5. Things I'd flag before you start

A few real design choices to make explicitly:

**Centering / standardizing covariates.** In your prototype, `gamma0[1]` is the intercept on the `_pr` scale, which is interpretable as the population mean only when all other covariates are zero. For continuous covariates this is unintuitive (mean age = 0?). Consider auto-mean-centering numeric columns and warning the user — or at minimum, documenting it loudly. Your `~ 1 + age` will give very different posteriors depending on whether `age` is centered.

**Prior on `gamma`.** `gamma ~ normal(0, 1)` is reasonable on the `_pr` scale, but the intercept and slopes have different reasonable scales. A common pattern is `gamma[D_start[p]] ~ normal(prior_mean_p, 1)` for the intercept and `gamma[non_intercept] ~ normal(0, 0.5)` for slopes. You could expose this as an additional argument or just hardcode reasonable defaults.

**Identifiability with default `mu_pr ~ normal(0, 1)`.** Your prototype uses `gamma0 ~ normal(-1, 1)` and `gamma1 ~ normal(0, 0.5)` — different scales for intercept vs slope. The hBayesDM convention is `mu_pr ~ normal(0, 1)`, which is fine for the intercept but a bit wide for slopes once factors are dummy-coded. Worth picking a default and documenting.

**Single-subject (`model_type = "single"`) and multipleB models.** Skip them in v1. `single` has no person dimension so covariates are meaningless. `multipleB` adds a block dimension that's orthogonal to the covariate question, but it complicates the design matrix story (do covariates vary by block? probably not, but the data layout is `[N, B]`). I'd ship covariates for `model_type = ""` only in the first release, and add the others in a follow-up.

**Backward compatibility.** Two paths to choose from:

1. *Hard cutover.* Every Stan file is rewritten in Phase 2; users must pass `formulas = NULL` (which engine fills with `~ 1`) to get old behavior. The old `mu_Arew` etc. outputs go away. Cleaner long-term but breaks downstream code.
2. *Side-by-side.* Keep old Stan files; add new ones as `igt_orl_cov.stan` etc.; gate on whether `formulas` is provided. More code paths to maintain but zero-risk migration.

I'd lean (1), with a clear changelog entry and a release-note example showing how to recover the old means via `posterior_means(fit)`. The codegen pipeline makes (1) tractable — regenerating wrappers from YAML means you don't manually edit 60+ R files.

**Tests.** Two regression tests are essential:
- `formulas = NULL` on the example dataset should produce the same posterior means as the legacy fit (within MCMC noise).
- Simulated data with known `gamma` should recover the truth at modest N.

## 6. Effort estimate

Rough breakdown for a single experienced developer:

- Phase 1 (R + Python infrastructure + templates): ~1 week. The R side is straightforward; Python takes a bit longer because of `formulaic` vs `patsy` decisions and tests.
- Phase 2 (per-model Stan refactor): ~1-2 days per model for the first 3-5 models (figuring out idioms, edge cases like `multipleB`, `single`). ~30-60 min per model after that. Total ~2-3 weeks for all ~60 models if doing them all; or pick the 10 most-used and ship those first.
- Phase 3 (regenerate wrappers): ~half a day, mostly running scripts and spot-checking.
- Phase 4 (helpers, tests, vignettes): ~1 week.

**Total**: ~4-6 weeks for a clean v1 covering the most-used models, ~8-10 weeks for full coverage. The path to a useful prototype is much shorter — Phase 1 plus `igt_orl` alone is probably 3-4 days and gives you a complete demo.
