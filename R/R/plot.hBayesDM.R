#' General Purpose Plotting for hBayesDM. This function plots hyper parameters.
#'
#' @keywords internal
#'
#' @param x Model output of class hBayesDM
#' @param type Character value that specifies the plot type. Options are: "dist", "trace", or "simple". Defaults to "dist".
#' @param ncols Integer value specifying how many plots there should be per row. Defaults to the number of parameters.
#' @param fontSize Integer value specifying the size of the font used for plotting. Defaults to 10.
#' @param binSize Integer value specifying how wide the bars on the histogram should be. Defaults to 30.
#' @param ... Additional arguments to be passed on
#'
#' @importFrom bayesplot mcmc_trace mcmc_intervals
#'
#' @method plot hBayesDM
#' @export

plot.hBayesDM <- function(x        = NULL,
                          type     = "dist",
                          ncols    = NULL,
                          fontSize = NULL,
                          binSize  = NULL,
                          ...) {

  # cmdstanr stores VB results in CmdStanVB; MCMC in CmdStanMCMC.
  is_vb <- inherits(x$fit, "CmdStanVB")
  if (is_vb) {
    cat("\n************************************************************************\n")
    cat("Variational inference was used to approximate posterior distributions!!\n")
    cat("For final inferences, we strongly recommend using MCMC sampling.\n")
    cat("************************************************************************\n")
  }

  if (grepl(pattern = "lba", x = x$model)) {
    numPars <- 4
  } else {
    numPars <- dim(x$allIndPars)[2] - 1
  }

  parNames <- names(x$parVals)[1:numPars]

  if (type == "dist") {
    source(file = system.file("plotting", "plot_functions.R", package = "hBayesDM"),
           local = TRUE)
    eval(parse(text = paste0("plot_", x$model, "(obj = x",
                             ", fontSize = ", fontSize,
                             ", ncols = ", ncols,
                             ", binSize = ", binSize, ")")))
    invisible()
  } else if (type == "trace") {
    bayesplot::mcmc_trace(x$fit$draws(parNames),
                          facet_args = list(ncol = ncols), ...)
  } else if (type == "simple") {
    bayesplot::mcmc_intervals(x$fit$draws(parNames), ...)
  }
}
