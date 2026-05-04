# One-time setup per shell

uv lives in ~/.local/bin, so either add it to PATH permanently
or prefix:

```
export PATH="$HOME/.local/bin:$PATH"
```

(asdf's .tool-versions in the repo root pins Python 3.13.13 + R
4.4.1 automatically when you cd in.)

# Python

```
cd ~/Projects/hBayesDM/Python
```

## run the smoke test (matches what CI used to run)

```
uv run pytest tests/test_ra_prospect.py -x
```

## run the full test suite (all ~65 model smoke tests — slow,

fits each model briefly)

```
uv run pytest tests -x
```

## import + interactive sanity check

```
uv run python -c "
from hbayesdm.models import ra_prospect
out = ra_prospect(data='example', niter=10, nwarmup=5,
nchain=1, ncore=1)
print('fit type:', type(out.fit).__name__)
print(out.all_ind_pars)
"
```

## or drop into a REPL

```
uv run python
```

```
uv run auto-syncs .venv/ from pyproject.toml + uv.lock before
executing — you don't need to activate anything.
```

# R

cmdstanr needs to know where CmdStan lives. One-time per shell
(or stick it in ~/.Rprofile):

```
export CMDSTAN="$HOME/.cmdstan/cmdstan-2.38.0"
```

Then:

```
cd ~/Projects/hBayesDM/R
```

## install (or reinstall after edits)

```
R CMD INSTALL --no-test-load .
```

## import + sanity check

```
Rscript -e '
library(hBayesDM)
out <- ra_prospect(data="example", niter=10, nwarmup=5,
nchain=1, ncore=1)
cat("fit class:", class(out$fit)[1], "\n")
print(out$allIndPars)
'
```

## run a single test file

```
Rscript -e '
testthat::test_file("tests/testthat/test_ra_prospect.R")
'
```

## run the full test suite

```
Rscript -e '
library(hBayesDM)
testthat::test_dir("tests/testthat")
'
```

If you want devtools-style iteration (no reinstall between
edits), you'd need devtools to install cleanly first, which
currently fails on ragg (image system libs). Easiest fix:

```
brew install freetype libpng libtiff jpeg-turbo webp harfbuzz
fribidi
```

```
Rscript -e 'install.packages(c("ragg","pkgdown","devtools"),
repos="https://cloud.r-project.org/")'
```

Then devtools::load_all() and devtools::test() work without
reinstalling.

Quick "did anything break" check

# Python

```
cd ~/Projects/hBayesDM/Python && uv run pytest
tests/test_ra_prospect.py
```

# R (in another shell)

```
cd ~/Projects/hBayesDM/R && Rscript -e '
cmdstanr::set_cmdstan_path(Sys.getenv("CMDSTAN"))
testthat::test_file("tests/testthat/test_ra_prospect.R")
'
```
