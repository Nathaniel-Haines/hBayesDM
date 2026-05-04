# Alternate plan: covariates **+** longitudinal growth modeling

## TL;DR — what's different from the original plan

The original plan added covariates to the existing cross-sectional models. This plan extends the same idea to longitudinal data, where each subject is observed across multiple sessions and the per-session person-level parameters can drift over time. The two extensions are **partially independent and partially layered**, which means we can ship them in stages without painting ourselves into a corner.

The right architectural choice is to introduce a new `model_type = "growth"` alongside the existing `""` (default hierarchical), `"single"`, and `"multipleB"`. This piggybacks on infrastructure that already handles model-type-specific data shapes, required columns (`block` for `multipleB`), and Stan file selection. Each task gets at most two Stan files — `igt_orl.stan` (cross-sectional) and `igt_orl_growth.stan` (longitudinal) — both supporting covariates via the same formula DSL.

**Why not collapse them?** A single unified Stan file with `S=1` as a degenerate case would duplicate every model's likelihood block under an extra session loop, costing a few percent runtime in the common single-session case for no benefit. Two model-types is cleaner and matches users' mental model: "I have one wave of data" vs. "I have repeated measures."

**Why not three or more variants per task?** We've already seen the codebase doesn't enjoy a model variant explosion (the bandit family already has 6+ variants per task and the YAML codegen barely keeps up). Plan 2 is the ceiling.

## Recap: the two orthogonal extensions

To set up the staging clearly:

- **Extension A — covariates on person-level parameters.** Lets users write `Arew ~ 1 + age + group` to estimate how person-level parameters depend on subject characteristics. Covered in the original plan. Lives in `model_type = ""` (and eventually `multipleB`, `growth`).
- **Extension B — longitudinal growth.** Adds a session dimension, time-indexed person-level parameters, growth coefficients (intercept + slope) per parameter, and session-level random noise. Lives in a new `model_type = "growth"`.

Extension B *requires* Extension A as a foundation — you don't want a longitudinal version that can't take covariates, because the whole point of repeated measures is to ask conditional questions like "does anxiety predict steeper decline in `Arew`?" But Extension A is fully usable on its own.

## Recommended staging

Three stages, each independently shippable:

1. **Stage 1 — covariates only** (model_type = ""). Same as the original plan.
2. **Stage 2 — growth scaffolding** (model_type = "growth", unconditional). Add the new model_type, the session/time data preprocessing, and the growth Stan template with `~ 1` formulas only (i.e., your `FIT_CONDITIONAL <- FALSE` branch). One model — `igt_orl_growth` — as the demo.
3. **Stage 3 — conditional growth** (model_type = "growth", with covariates). Wire the formula DSL through to the growth template, supporting both intercept formulas and slope formulas. Roll out across the model catalog.

Stage 1 alone is useful and shippable. Stages 2+3 should ship together since unconditional growth without covariates has limited research value beyond a "does the model work" demo.

## Stage 1 — covariates (cross-sectional)

Identical to the original plan. The only forward-compatibility detail to keep in mind: the formula DSL we design here needs to extend cleanly to `par_int ~ ...; par_slope ~ ...` later. Two adjustments to the original plan to lock that in:

- Validate formula LHS as a member of `parameters` *or* a `<param>_int` / `<param>_slope` suffix (only the former is meaningful in stage 1, but the parser shouldn't reject the latter outright — a clearer error is "growth-only LHS in a cross-sectional model").
- The internal representation of formulas is keyed by `(parameter, term)` where `term ∈ {"int", "slope"}`. In stage 1 only `term = "int"` is populated. This means `D_start`/`D_end` are length `P` for cross-sectional and length `2P` for growth (matching your prototype's array layout).

Other than that, stage 1 = original plan.

## Stage 2 — growth scaffolding

This is the meat of the new work. I'll cover what changes in three places: data shape, Stan templates, and the engine.

### 2.1 Data column requirements

`model_type = "growth"` requires three things in the input data:

- A subject identifier column (`subjID`) — already required everywhere.
- A **session** column (integer, 1..S) identifying which session each trial belongs to.
- A **time** value associated with each session. Two reasonable conventions:
  - *Time-as-session*: time is just `session - 1` (linear in session number). User passes nothing extra.
  - *Time-as-elapsed*: time is a separate column (e.g., months from baseline) that varies across subjects. User passes the column name via a new YAML field or function argument.

I'd support both. Default to time-as-session for simplicity, with a `time_var` argument that overrides to a named column. Your prototype uses time-as-elapsed (`"session"` as the time variable name passed to `make_stan_data_growth`), and that's the more flexible default once we offer it.

This adds `data_columns: { session: ... }` to the YAML for any growth model. The engine's existing `data_columns` validation logic (checking `complete.cases` on insensitive column names) handles it.

### 2.2 Stan template structure

The growth Stan template has more moving parts than the cross-sectional one. Here's the structural blueprint for `igt_orl_growth.stan`, distilled from your prototype but cleaned up to fit hBayesDM idioms:

```stan
#include /pre/license.stan

data {
  int<lower=1> N;                 // subjects
  int<lower=1> S;                 // max sessions
  int<lower=1> T;                 // max trials (across all subject-sessions, flattened)

  // Trial-level behavioral data — flat layout matching prototype
  int<lower=1> Tsubj[N];                  // total trials per subject across sessions
  int<lower=0, upper=1> session_start[N, T];  // 1 where a new session begins
  int choice[N, T];
  real outcome[N, T];
  real sign_out[N, T];

  // Time variable per (subject, session)
  real time[N, S];

  // Covariate / design machinery — same shape as cross-sectional but indexed by session
  // for time-varying covariates. Most users will have time-invariant covariates,
  // in which case X[i, s, :] is constant across s.
  int<lower=1> D;
  array[N, S, D] real X;

  // 2P entries: first P for intercept formulas, last P for slope formulas
  // P = 5 for igt_orl (Arew, Apun, K, betaF, betaP)
  array[10] int D_start;
  array[10] int D_end;

  int<lower=0, upper=1> prior_only;
}

transformed data {
  vector[4] initV = rep_vector(0.0, 4);
  int D_int   = D_end[5];           // number of intercept design columns
  int D_slope = D - D_int;          // number of slope design columns
}

parameters {
  // Group-level coefficients
  vector[D_int]   gamma_int;        // intercept linear-predictor coefs
  vector[D_slope] gamma_slope;      // slope    linear-predictor coefs

  // Subject-level random effects: correlated intercept + slope, per parameter
  array[5] matrix[2, N] beta_pr;
  array[5] vector<lower=0>[2] sigma_beta;
  array[5] cholesky_factor_corr[2] R_chol;

  // Session-level idiosyncratic noise (after fixed + linear-trend predictions)
  array[5] matrix[N, S] theta;
  array[5] real<lower=0> sigma_theta;
}

transformed parameters {
  array[5] matrix[2, N] beta_tilde;       // post-Cholesky person-level deviations
  matrix[N, S] Arew;
  matrix[N, S] Apun;
  matrix[N, S] K;
  matrix[N, S] betaF;
  matrix[N, S] betaP;

  // Apply Cholesky factor to get correlated intercept/slope deviations
  for (p in 1:5)
    beta_tilde[p] = diag_pre_multiply(sigma_beta[p], R_chol[p]) * beta_pr[p];

  // Linear predictor + random effects + session noise, then bound transforms
  for (s in 1:S) {
    for (i in 1:N) {
      // For each parameter p: linear_predictor_int + linear_predictor_slope * time
      // + person random intercept + person random slope * time
      // + session-level noise

      real eta_Arew =
        dot_product(to_vector(X[i, s, D_start[1]:D_end[1]]),       gamma_int[D_start[1]:D_end[1]])
      + dot_product(to_vector(X[i, s, D_start[6]:D_end[6]]),       gamma_slope[(D_start[6]-D_int):(D_end[6]-D_int)]) * time[i, s]
      + beta_tilde[1, 1, i]
      + beta_tilde[1, 2, i] * time[i, s]
      + sigma_theta[1] * theta[1, i, s];
      Arew[i, s] = Phi_approx(eta_Arew);

      // ... same pattern for Apun, K, betaF, betaP ...
    }
  }
}

model {
  // Priors
  gamma_int   ~ normal(0, 1);
  gamma_slope ~ normal(0, 0.5);

  for (p in 1:5) {
    R_chol[p] ~ lkj_corr_cholesky(2);
    to_vector(beta_pr[p]) ~ std_normal();
    sigma_beta[p] ~ normal(0, 0.2);
    to_vector(theta[p]) ~ std_normal();
    sigma_theta[p] ~ normal(0, 0.2);
  }

  if (!prior_only) {
    for (i in 1:N) {
      int session = 0;
      // Existing igt_orl trial loop — but with parameters indexed [i, session]
      // and a session-reset whenever session_start[i, t] == 1
      for (t in 1:Tsubj[i]) {
        if (session_start[i, t] == 1) {
          session += 1;
          // ... initialize ev, ef, pers, util ...
        }
        // ... same likelihood as cross-sectional igt_orl, but using
        //     Arew[i, session] instead of Arew[i] etc. ...
      }
    }
  }
}

generated quantities {
  // Random effect intercept-slope correlations (one 2x2 matrix per parameter)
  corr_matrix[2] R_Arew  = R_chol[1] * R_chol[1]';
  corr_matrix[2] R_Apun  = R_chol[2] * R_chol[2]';
  // ...

  real log_lik[N, T];
  real y_pred[N, T];
  // ... posterior predictive loop, same structure as cross-sectional ...
}
```

A few structural choices baked into this:

- **Flat trial array** (`[N, T]`) with a `session_start` flag rather than nested `[N, S, T]`. Matches your prototype, avoids the worst-case memory footprint when sessions have very different trial counts (a 200-trial session next to a 5-trial drop-out doesn't waste a 200x dimension on the dropout). It also keeps the trial-level loop nearly identical to the cross-sectional version.
- **Correlated intercept-slope random effects** (the `R_chol[p]` Cholesky factors). This matches your prototype and is the more general default. If correlation is uninformative, the LKJ(2) prior pulls toward zero correlation gracefully.
- **Session-level noise** (`sigma_theta[p] * theta[p, i, s]`). Captures occasion-specific deviation from the linear trend. Without it, a subject who had a bad day session 3 would get all that variation pushed into their slope estimate.
- **`prior_only` flag** for prior predictive checks. Your prototype has this; the existing hBayesDM Stan files don't. Worth standardizing as a default in the growth template (and considering for cross-sectional too in stage 1).
- **`R_chol[p]` correlation matrices in generated quantities**. These are user-facing — researchers want to know how intercepts and slopes correlate at the population level.

### 2.3 Engine changes

The cross-sectional engine handles formulas with one term-type ("int") per parameter. The growth engine extends this to two term-types ("int" and "slope") per parameter, with covariates indexed by `[N, S, D]` instead of `[N, D]`.

In R, the new helpers in `formula_utils.R` extend cleanly:

```r
# Formulas now look like:
#   list(Arew_int = ~ 1 + anxiety, Arew_slope = ~ 1, Apun_int = ~ 1, ...)
# The validator splits on the suffix.

validate_growth_formulas <- function(formulas, parameters) {
  expected_lhs <- c(paste0(parameters, "_int"), paste0(parameters, "_slope"))
  formulas <- formulas %||% list()
  bad <- setdiff(names(formulas), expected_lhs)
  if (length(bad) > 0) {
    stop("Unknown LHS in growth formulas: ", paste(bad, collapse = ", "), ". ",
         "Expected one of: ", paste(expected_lhs, collapse = ", "))
  }
  out <- list()
  for (lhs in expected_lhs) {
    out[[lhs]] <- formulas[[lhs]] %||% as.formula("~ 1")
  }
  # Order: all intercepts in parameter order, then all slopes
  out[expected_lhs]
}
```

The design matrix construction is more involved because covariates can be time-varying. The per-subject covariate frame becomes a per-(subject, session) frame:

```r
extract_subj_session_covars <- function(raw_data, subjs, n_sessions, covar_cols, time_var) {
  # One row per (subject, session). Numeric → mean within (subj, session),
  # non-numeric → first non-NA. Carry forward if a session is missing covariate data
  # (matches your prototype's `fill(everything(), .direction = "down")` behavior).
  out <- expand.grid(subjid = subjs, session = seq_len(n_sessions))
  for (col in c(covar_cols, time_var)) {
    out[[col]] <- NA
    for (i in seq_along(subjs)) {
      for (s in seq_len(n_sessions)) {
        sub <- raw_data[raw_data$subjid == subjs[i] & raw_data$session == s, ]
        if (nrow(sub) == 0) next
        v <- sub[[col]]
        out[out$subjid == subjs[i] & out$session == s, col] <-
          if (is.numeric(v)) mean(v, na.rm = TRUE)
          else { nn <- v[!is.na(v)]; if (length(nn)) nn[1] else NA }
      }
    }
  }
  out <- dplyr::group_by(out, subjid) %>%
    tidyr::fill(dplyr::everything(), .direction = "down") %>%
    dplyr::ungroup()
  as.data.frame(out)
}

build_growth_design <- function(formulas, subj_session_covars, n_subj, n_sessions) {
  # For each formula, build a model matrix on the long (subj, session) frame,
  # then reshape to a 3D array [N, S, ncol].
  mms <- lapply(formulas, function(f)
    stats::model.matrix(f, data = subj_session_covars))
  d_per <- vapply(mms, ncol, integer(1))
  d_end <- cumsum(d_per)
  d_start <- c(1L, d_end[-length(d_end)] + 1L)

  D <- sum(d_per)
  X <- array(0, c(n_subj, n_sessions, D))
  for (j in seq_along(mms)) {
    cols <- d_start[j]:d_end[j]
    mm <- mms[[j]]
    for (i in seq_len(n_subj)) {
      for (s in seq_len(n_sessions)) {
        row_idx <- which(subj_session_covars$subjid == subjs[i] &
                         subj_session_covars$session == s)
        if (length(row_idx) == 1)
          X[i, s, cols] <- mm[row_idx, ]
      }
    }
  }
  list(X = X, D = D, D_start = d_start, D_end = d_end,
       coef_names = unlist(Map(function(p, mm) paste(p, colnames(mm), sep = "_"),
                               names(mms), mms)))
}
```

The `hBayesDM_model.R` engine routes on `model_type`:

```r
if (model_type == "growth") {
  formulas <- validate_growth_formulas(formulas, names(parameters))
  covar_cols <- unique(unlist(lapply(formulas, all.vars)))
  covar_cols <- setdiff(covar_cols, insensitive_data_columns)
  ssc <- extract_subj_session_covars(raw_data, subjs, n_sessions, covar_cols, time_var)
  dm <- build_growth_design(formulas, ssc, n_subj, n_sessions)
  # ... merge X, D, D_start, D_end, time, session_start, S into data_list ...
} else {
  # cross-sectional path from stage 1
}
```

The pars to extract change too: `gamma_int`, `gamma_slope`, `sigma_beta`, `sigma_theta`, `R_chol` (and maybe `R_Arew` etc. from generated quantities) replace the simpler `gamma`, `sigma` of stage 1.

### 2.4 Python parallel

Same logic in Python. Two notes:

- `formulaic` and `patsy` both handle the `~ 1` case fine and produce DataFrames with named columns, so the per-formula model matrix construction is identical to R.
- The 3D `X` array is just `numpy.zeros((N, S, D))` filled in a loop. No special library needed.

The main subtlety is the long-format covariate aggregation, which is a `groupby(['subjID', 'session']).agg(...)` followed by reindexing onto a complete `(subj, session)` grid and forward-filling. Pandas handles this cleanly:

```python
def extract_subj_session_covars(raw_data, subjs, n_sessions, covar_cols, time_var):
    cols_to_agg = list(covar_cols) + [time_var]
    agg_dict = {c: ('mean' if pd.api.types.is_numeric_dtype(raw_data[c])
                    else lambda x: x.dropna().iloc[0] if x.dropna().any() else None)
                for c in cols_to_agg}
    long = raw_data.groupby(['subjid', 'session'])[cols_to_agg].agg(agg_dict).reset_index()
    full = pd.MultiIndex.from_product([subjs, range(1, n_sessions + 1)],
                                      names=['subjid', 'session']).to_frame(index=False)
    long = full.merge(long, on=['subjid', 'session'], how='left')
    long = long.sort_values(['subjid', 'session']).groupby('subjid').ffill().reset_index(drop=True)
    return long
```

## Stage 3 — conditional growth (the full thing)

By the time stages 1 and 2 are complete, stage 3 is mostly testing and rollout. The Stan template from stage 2 already supports covariates on both intercepts and slopes — what's missing is just user-facing documentation, helpers, and applying the pattern across the model catalog.

Two new helpers to write:

- **`predict_growth()`**: takes a fit object and a covariate value (or grid of values), returns posterior predictive trajectories of person-level parameters across sessions. This is your `marginal_mean_growth` function generalized — it should marginalize over `sigma_beta` and `sigma_theta` to produce the actual posterior expected mean (since `Phi_approx` of a normal is right-skewed). For unbounded parameters (`betaF`, `betaP`), the marginalization is trivial; for bounded ones (`Arew`, `Apun`, `K`), it's a Monte Carlo integral over the random effects, which is what your prototype does.
- **`tidy_gamma_growth()`**: returns a data.frame / DataFrame with columns `parameter`, `term_type` (`"int"` or `"slope"`), `coefficient` (e.g. `"(Intercept)"`, `"anxiety"`), and the posterior summaries. Replaces your prototype's `relabel_map` with auto-generated labels.

## Per-model rollout strategy

For stage 3, not every model needs a `_growth` variant. Realistic priorities:

- **High value**: `igt_orl`, `igt_pvl_decay`, `igt_pvl_delta`, `igt_vpp` (Iowa gambling family — your home turf, naturally longitudinal in clinical work), `gng_m1` through `gng_m4` (go/no-go, common in development studies), `prl_*` (probabilistic reversal, used in clinical RL studies), `dd_*` (delay discounting, often longitudinal).
- **Lower priority**: bandit models (rarely longitudinal), `cgt_cm` (Cambridge gambling, single-session by design), `cra_*` (choice under risk, usually one wave), `*_single` (single-subject types are explicitly excluded).

Reasonable v1: 8-12 high-priority models get `_growth` variants. The rest stay cross-sectional with covariate support from stage 1.

## Things that need explicit decisions

A few design choices to make before coding starts:

- **Time-as-session vs time-as-elapsed default.** Recommend exposing a `time_var` argument that defaults to NULL (= use `session - 1`). When set to a column name, that column becomes the time variable. Document loudly that scaling matters: if time is in months from baseline, a "slope of 0.1" means 0.1 unit change per month, which compounds over a year-long study.
- **Time centering.** Same issue as covariate centering in Plan 1, but worse. With time-as-session and S=5, the intercept is the value at session 1, which is fine. With time-as-elapsed in months, where baseline is a non-trivial time point, the user might want time centered or not depending on whether the intercept means "value at study start" or "value at average session time". Default to no centering, document the consequences.
- **Time-varying vs time-invariant covariates.** The `[N, S, D]` design matrix supports both transparently — if a covariate is time-invariant, the user just supplies the same value across sessions and the engine handles it. But documentation should be explicit that the formula DSL treats the design matrix uniformly; if you want to interact a covariate with time, you do it via the `_slope` formula, not via an interaction term in the `_int` formula. (i.e., `Arew_slope ~ 1 + anxiety` is "anxiety predicts slope of Arew", which is *not* the same as `Arew_int ~ 1 + anxiety:time`.)
- **Missing sessions.** Your prototype handles dropout via the `Tsubj[i]` count and `session_start` markers — a subject with only 3 of 5 sessions just contributes 3 trial-loops. The engine needs to mirror this: don't error on subjects with missing sessions, do warn if covariate values are missing for sessions where the subject has trials (and either error or last-observation-carry-forward, with the choice exposed).
- **Identifiability of the random effects covariance.** With small S (say S=2 or S=3), the intercept-slope correlation can be poorly identified. The LKJ(2) prior helps but not always enough. For very small S, consider falling back to independent random effects. Worth a runtime warning if S < 3.
- **Interaction with Stage 1 covariate centering.** If we auto-center continuous covariates in stage 1, we should do the same for intercept-formula covariates in growth. For slope-formula covariates, the centering question is the same as for any interaction — center the main effect to make the intercept interpretable as "slope at average covariate value."

## Effort estimate (revised)

For a single experienced developer:

- **Stage 1** (covariates, cross-sectional): ~1 week infrastructure + ~2-3 weeks per-model rollout. Same as original plan.
- **Stage 2** (growth scaffolding + 1 demo model): ~2 weeks. The engine work is more substantial than stage 1 because of the long-format covariate aggregation, time variable handling, and the new Stan template structure. Expect debugging time on the trial-loop / session-reset interaction.
- **Stage 3** (conditional growth + 8-12 high-priority models): ~3-4 weeks. Each model's growth Stan file is bigger work than its covariate-only refactor — ~1-2 days per model rather than 30-60 minutes — because the trial loop has to be re-read carefully to ensure parameter indexing changes from `[i]` to `[i, session]` everywhere.

**Total**: ~7-10 weeks for a v1 covering both extensions across the most-used models. The natural shipping cadence is:

- Month 1: ship stage 1 (covariates only) — useful on its own, gets a wide user base testing the formula DSL.
- Month 2: ship stage 2 (`igt_orl_growth` as a tech preview behind a feature flag).
- Month 3: ship stage 3 (full conditional growth across the priority model list).

## What I'd flag, on top of the original plan's flags

The original plan's design issues all still apply (centering, prior choice, single-subject exclusion, backward compatibility). Growth-specific additions:

- **Sample size requirements are real.** A minimum of ~30 subjects with ~3 sessions each is a soft floor for the conditional growth model to identify both random-effect variances (`sigma_beta`, `sigma_theta`) and the population-level coefficients (`gamma_int`, `gamma_slope`). Below that, the prior dominates. Document this and ideally print a warning when N or S falls below thresholds.
- **`prior_only` mode is essential and currently absent from hBayesDM.** Growth models need prior predictive checks more than cross-sectional ones because the priors interact in non-obvious ways (LKJ on correlations × normal on slopes × half-normal on session noise). Add `prior_only` as an argument to all growth model functions; consider backporting to cross-sectional models for consistency.
- **Posterior predictive checks need redesign.** The cross-sectional `y_pred[N, T]` is sufficient for a single session, but for growth models researchers usually want session-stratified PPCs ("does the model recover the within-subject change from session 1 to 5?"). The `y_pred` array is already correctly dimensioned (since it's flat `[N, T]` with `session_start` markers), but the helper code that summarizes PPCs needs to be growth-aware to stratify by session.
- **MCMC efficiency.** Your prototype uses 8 chains × 500 sampling iterations × 200 warmup. The growth Stan model has many more parameters than the cross-sectional one (about `2P` extra variance terms plus the `R_chol` Cholesky factors plus the `theta[p, i, s]` array). Default `niter`/`nwarmup` may be too low. Consider raising defaults for growth models to `niter = 2000`, `nwarmup = 1000`, and use `init = 0` rather than random initialization (which can wander into bad geometry on these models — your prototype uses `init = 0` for this reason).
- **The chrono of design decisions.** If you ship stage 1 before stage 2, users will start writing code against the cross-sectional formula DSL. If the growth DSL needs slightly different syntax (the `_int` / `_slope` suffix), that's a breaking change. Mitigation: in stage 1, accept the suffix syntax silently for `model_type = ""` (treating `Arew_int` as an alias for `Arew`), so users can write longitudinal-ready formulas even on cross-sectional models. This locks in the syntax across both shipping stages.
