# CCmatic
Tool to synthesize CCAs that provably achieve given desired properties (e.g., high utilization, low delay), under all traces that a network model can exhibit (e.g., CCAC network model).

## Dependencies
python>=3, z3, numpy, pandas, all dependencies of ccac.
Uses repositories: pyz3_utils, cegis, and ccac.

## Run
Files `ccmatic/main_*` differ in environment/template. The desired properties are configurable.
```
python -m ccmatic.main_cca_lossess  # Infinite buffer. No loss/delay signals.
python -m ccmatic.main_multi_cca_lossless  # Infinite buffer with possibly multiple flows. Delay signal.
python -m ccmatic.main_cca_vb_loss  # Finite (possibly variable buffer). Loss signal.
python -m ccmatic.main_cca_vb_loss_qbound  # Finite (possibly variable buffer). Loss signal, delay signal.
python -m ccmatic.main_cca_vb_modeswitch  # Finite (possibly variable buffer). Loss/delay signal to switch between two modes, loss signal in one mode.
```

## Play around
1. Change thresholds, e.g., `util_frac`, `delay_bound` in `main_cca_lossless.py`, or `util_frac`, `loss_rate`, `c.buf_min` in `main_cca_loss.py`.
2. Test your CCAs, e.g., (modify `tests/ccac/test_verifier.py` according to needs)
```
python -m tests.ccac.test_verifier
```
3. Create your own templates and environments by referencing `main_cca_lossess.py` and `main_cca_loss.py`.
