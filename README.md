# Towards provably performance congestion control

This repository accompanies the NSDI 2024 paper: "Towards provably performant
congestion control", by Anup Agarwal, Venkat Arun, Devdeep Ray, Ruben Martins,
and Srinivasan Seshan.

# Scope

This tool (CCmatic) synthesizes congestion control algorithms (CCAs) that
achieve given desired performance properties (e.g., high utilization, low
delay, low loss), on networks described using network models like CCAC [SIGCOMM
2021]. For synthesis, CCmatic uses the program synthesis technique:
Counterexample guided inductive synthesis (CEGIS). We implement the verifier
and generator in CEGIS using an SMT solver (Z3 in this case). The synthesis is
done for an under-specified invariant, this repository also contains the code
to build tight proofs of CCA performance post synthesis.

# Dependencies
We use python3 and python packages including numpy, and pandas. Example
commands using the conda package manager
(https://docs.conda.io/projects/conda/en/latest/index.html).
```
conda create -yn ccmatic python=3
conda activate ccmatic
conda install numpy pandas pip
```

We use the Z3 SMT solver. Please install it and its python bindings from
https://github.com/Z3Prover/z3.
```
pip install z3-solver
```

This repository contains submodules, depending on which all submodules are
used, you may need to install the dependencies of the submodules as well:
1. cegis (python implementation of CEGIS)
2. pyz3_utils (convenience wrappers around Z3 python API)
3. CCAC (network model and SMT encoding of it)
4. plot_config (convenience wrappers on matplotlib for plotting and plot
   configuration)

Run following after cloning the repository to populate the submodules if you
forgot to clone with `--recursive` flag.: `git submodule update --init
--recursive`

Note, the dependency may be incomplete, please reach out if you face any issues
in running the code.

# Disclaimer

This is a research prototype, it may contain dead, undocumented, duplicate, or
suboptimal code and may not use the best coding practices (\# move fast break
things). Please reach out to me (anupa@andrew.cmu.edu), or create a GitHub
issue if you have any questions.

# Running

## Hello world (main files and commands)
The main entry point to the tool is
`ccmatic/main_cca_belief_template_modular.py`. Note, the other files
`ccmatic/main_*.py` are deprecated.

Note, run the commands in the root directory of the repository.

### Example command to synthesize CCAs
cc_qdel (described in row 2 (group 1) of Table 2 in the paper).
```
python -m ccmatic.main_cca_belief_template_modular --infinite-buffer --verifier-type ccac -T 5 --opt-ve-n --opt-pdt-n --opt-wce-n --opt-feasible-n
```

cc_probe_slow (described in row 11 (group 6) of Table 2 in the paper).
```
python -m ccmatic.main_cca_belief_template_modular --dynamic-buffer --verifier-type cbrdelay -T 7 --opt-ve-n --opt-pdt-n --opt-wce-n --opt-feasible-n
```

`ccmatic/solutions/solutions_belief_template_modular.py` contains some of the
solutions we produced. These are used for producing proofs. We just copy-paste
the solution from the output of the CEGIS loop to this file.

### Example command to produce and check proofs

For the solution cc_qdel (described in appendix F of the paper).
```
python -m ccmatic.main_cca_belief_template_modular --dynamic-buffer --large-buffer --verifier-type ccac -T 10 --opt-ve-n --opt-pdt-n --opt-wce-n --opt-feasible-n --solution probe_qdel --proofs --fix-minc --fix-maxc
```

For the solution cc_probe_slow (not described in the paper).
```
python -m ccmatic.main_cca_belief_template_modular --dynamic-buffer --verifier-type cbrdelay -T 10 --opt-ve-n --opt-pdt-n --opt-wce-n --opt-feasible-n --solution drain_bq_probe --proofs
```

Note, for T=10, these may take a while. If you just want to see how the tool
runs, you can try T=7. Not all lemmas will be satisfied for T=7.

Creating the proof has three steps:
1. Writing the lemmas
2. Searching for constants in the lemmas
3. Checking the lemmas

We have already done this for the two CCAs: cc_qdel and cc_probe_slow in the
paper (probe_qdel and drain_bq_probe in the code respectively). The proofs are
written in `ccmatic/verifier/proofs.py`. We already cached the constant terms
in the lemmas in this file. Running the above command by default, will just
search for the constants in the last lemma, that is search for the best
performance bounds in steady state.

If you want to use the verifier to check the proofs, you can set `check_lemmas
= True` for the corresponding network model in `ccmatic/verifier/proofs.py`.

If you want to re-do the search for the constant terms in the lemmas, you can
delete the corresponding line in the cache function. E.g., commenting out
```
self.recursive[self.steady__min_c.lo] = 69
self.recursive[self.steady__max_c.hi] = 301
```
in the probe_qdel proof, will re-do the search for these constants in
lemma21__beliefs_recursive (in the code). This corresponds to Lemma F.3 in the
paper. Note, lemmas are processed in a sequence, so checking or searching for
constants in a later lemma may require cached constants from earlier lemmas. If
these are absent the proof checking will show error message describing which
cached constants are missing.

### Loss vs. convergence time tradeoff

`tests/ccmatic/test_convergence_rate.py` produces the plot for the loss vs.
convergence time tradeoff (Figure 7 in the paper).

## Play around (changing inputs/specifications)
Flags used in `ccmatic/main_cca_belief_template_modular.py`

`--verifier-type` flag specifies the network model (e.g., ccac, cbrdelay, or
ideal link).

`-T` flag specifies the trace length to explore.

`--opt-ve-n --opt-pdt-n --opt-wce-n --opt-feasible-n` flags turn off some
optimizations we explored. These helped when we explored non-belief based CCA
templates (e.g., other `ccmatic/main_*.py` files). These do not help much with
belief-based templates, hence we keep them off. We used the
`scripts/run_optimizations_eval.sh` to evaluate the effect of these
optimizations. Please reach out if you are interested in learning more about
these optimizations.

`--infinite-buffer, --finite-buffer, --dynamic-buffer, --large-buffer,
--small-buffer` control the buffer size in the network. Dynamic buffer means
the verifier is free to choose an arbitrary (but constant over time) buffer
size.

`--proofs` Produce and check proofs for the solution specified using `--solution`.

`--fix-minc --fix-maxc` By default, in synthesis, we assume that the beliefs
are always consistent. Adding these flags enables code that handles stale
beliefs. This should be turned on when producing proofs for the CCAC network
model. Note, we have similar flags for CBR delay in
`ccmatic/verifier/cbr_delay.py`. This file needs to be manually updated to turn
these flgs to True when producing proofs for CBR delay (we have not yet exposed
these to the CLI API).
```
fix_stale__min_c_lambda: bool = False
fix_stale__bq_belief: bool = False
```

`--optimize` Binary search what are the best bounds on the performance metrics
in the synthesis invariant that are satisfied by the synthesized solution
specified using `--solution`. Binary search works because larger utilization
implies harder to meet (all performance metrics are monotonic).

`--solution SOLUTION` Name of the solution used for above flags.

`--cegis-with-solution` Used for debugging. Runs CEGIS with a hard-coded
solution specified using `--solution`.

`--manual-query` Used for debugging. Solves any custom Z3 formula setup using
the inputs to the CEGIS loop.
