# CCmatic
Tool to synthesize CCAs that provably achieve given desired properties (e.g., high utilization, low delay), under all traces that a network model can exhibit (e.g., CCAC network model).

## Dependencies
python>=3, z3, numpy, pandas, all dependencies of ccac.
Uses repositories: pyz3_utils, cegis, and ccac.

## Run
Files `ccmatic/main_*` differ in environment/template. The desired properties are configurable.
```
python -m ccmatic.main_cca_belief_template_modular --dynamic-buffer -T 10 --opt-ve-n --opt-pdt-n --opt-wce-n --opt-feasible-n --no-large-loss --verifier-type cbrdelay

# All below is deprecated
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

## Coding practices

### Where all are beliefs used:
1. Initial beliefs - beliefs are consistent or stale
2. Belief updates - common case and timeout
3. Templates
4. Specification - Beliefs improve, beliefs become consistent
5. Variables
6. Derived expressions
7. Proofs
8. definition vs. verifier variables.

Whenever updating beliefs, you might need to update some/all of above.

### Belief update
1. Since we don't use O(n^3) constraints to compute the belief update, a
   recomputed belief is only recomputed for the last step (its not recomputed
   for the entire trace), so we should not timeout if the belief changed
   (because that means that if we used recomputed belief using O(n^3)
   constraints then we would have had a different recomputed belief which is
   same as the non timeout belief.)
2. For min_c_lambda we actually recompute using O(n^3) constraints, so we can
   ignore the above.