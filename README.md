# CCmatic
Tool to synthesize CCAs that provably achieve given desired properties (e.g., high utilization, low delay), under all traces that a network model can exhibit (e.g., CCAC network model).

## Dependencies
python>=3, z3, numpy, pandas, all dependencies of ccac.
Uses repositories: pyz3_utils, cegis, and ccac.

## Run
```
python -m ccmatic.main_cca_lossess  # Syntheisze CCA for inf buffer case
python -m ccmatic.main_cca_loss  # Synthesize CCA that handles loss
```

## Play around
1. Change thresholds, e.g., `util_frac`, `delay_bound` in `main_cca_lossless.py`, or `util_frac`, `loss_rate`, `c.buf_min` in `main_cca_loss.py`.
2. Test your CCAs, e.g., (modify `tests/ccac/test_verifier.py` according to needs)
```
python -m tests.ccac.test_verifier
```
3. Create your own templates and environments by referencing `main_cca_lossess.py` and `main_cca_loss.py`.
