context("Test user-facing API (plots, diagnostics, IC, HDI helpers)")
library(hBayesDM)

# Fit one small hierarchical model and reuse across assertions to avoid
# repeated cmdstanr compile cost.
fitted <- suppressWarnings(suppressMessages(
  dd_hyperbolic(data  = "example",
                   niter = 300, nwarmup = 150,
                   nchain = 2, ncore = 2)
))

test_that("result object has expected structure", {
  skip_on_cran()
  expect_s3_class(fitted, "hBayesDM")
  expect_true(inherits(fitted$fit, "CmdStanMCMC"))
  expect_s3_class(fitted$allIndPars, "data.frame")
  expect_true(is.list(fitted$parVals))
  expect_true("log_lik" %in% names(fitted$parVals))
  expect_equal(fitted$model, "dd_hyperbolic")
})

test_that("plot dispatcher accepts trace and simple", {
  skip_on_cran()
  expect_silent(p_trace <- plot(fitted, type = "trace"))
  expect_silent(p_simple <- plot(fitted, type = "simple"))
  expect_true(inherits(p_trace, "ggplot") || inherits(p_trace, "gg"))
  expect_true(inherits(p_simple, "ggplot") || inherits(p_simple, "gg"))
})

test_that("plotInd returns a ggplot for an individual-level parameter", {
  skip_on_cran()
  # Pick the first per-subject parameter that actually has draws stored.
  ind_par <- names(fitted$parVals)[
    !startsWith(names(fitted$parVals), "mu_") &
      !names(fitted$parVals) %in% c("sigma", "log_lik")
  ][1]
  p <- plotInd(fitted, ind_par)
  expect_true(inherits(p, "ggplot") || inherits(p, "gg"))
})

test_that("rhat returns data.frame, and threshold form returns logical", {
  skip_on_cran()
  rd <- rhat(fitted)
  expect_s3_class(rd, "data.frame")
  expect_true("Rhat" %in% colnames(rd))
  res <- suppressWarnings(rhat(fitted, less = 1e9))
  expect_true(isTRUE(res))
})

test_that("extract_ic returns LOOIC by default", {
  skip_on_cran()
  ic <- extract_ic(fitted)
  expect_true("LOOIC" %in% names(ic))
  expect_true(is.finite(ic$LOOIC$estimates[3, 1]))
})

test_that("printFit returns a data.frame with weights column", {
  skip_on_cran()
  tab <- printFit(fitted)
  expect_s3_class(tab, "data.frame")
  expect_equal(nrow(tab), 1L)
  expect_true(any(grepl("Weights", colnames(tab))))
})

test_that("HDIofMCMC returns 2-vector from samples", {
  set.seed(1)
  hdi_lim <- HDIofMCMC(rnorm(2000), credMass = 0.5)
  expect_length(hdi_lim, 2)
  expect_true(hdi_lim[1] < hdi_lim[2])
})

test_that("modelRegressor=TRUE extracts regressors for gng_m1", {
  skip_on_cran()
  m <- suppressWarnings(suppressMessages(
    gng_m1(data = "example", niter = 40, nwarmup = 20,
           nchain = 1, ncore = 1, modelRegressor = TRUE)
  ))
  reg <- m$modelRegressor
  expect_true(is.list(reg))
  expect_setequal(names(reg), c("Qgo", "Qnogo", "Wgo", "Wnogo"))
  for (arr in reg) {
    expect_true(is.numeric(arr))
    expect_gte(length(dim(arr)), 2L)
    expect_true(all(is.finite(arr)))
  }
})

test_that("modelRegressor=TRUE errors when model has no regressors", {
  skip_on_cran()
  expect_error(
    dd_hyperbolic(data = "example", niter = 20, nwarmup = 10,
                  nchain = 1, ncore = 1, modelRegressor = TRUE),
    "regressors"
  )
})

test_that("vb=TRUE returns a CmdStanVB fit with usable summaries", {
  skip_on_cran()
  m <- suppressWarnings(suppressMessages(
    dd_hyperbolic(data = "example", niter = 20, nwarmup = 10,
                  nchain = 1, ncore = 1, vb = TRUE)
  ))
  expect_s3_class(m, "hBayesDM")
  expect_true(inherits(m$fit, "CmdStanVB"))
  expect_s3_class(m$allIndPars, "data.frame")
  expect_true(is.list(m$parVals))
})

test_that("plotDist and plotHDI return ggplot objects", {
  set.seed(1)
  s <- rnorm(500)
  p1 <- plotDist(s, binSize = 30)
  expect_true(inherits(p1, "ggplot") || inherits(p1, "gg"))
  p2 <- suppressMessages(plotHDI(s))
  expect_true(inherits(p2, "ggplot") || inherits(p2, "gg"))
})
