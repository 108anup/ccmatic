#!/bin/bash

# set -xe

BUFFER="${BUFFER:---dynamic-buffer}"
# BUFFER=""
OUTDIR=3leaf-app$BUFFER #-nolargeloss$BUFFER

run() {
    args=$@
    echo $args
    cmd="timeout 7d python -m ccmatic.main_cca_belief_template -T 9 --run-log-dir logs/optimizations/$OUTDIR $args --app-limited"
    # cmd="timeout 7d python -m ccmatic.main_cca_lossless_ccmatic $args"
    # cmd="echo $args"
    tmux send-keys "$cmd" Enter
}

tmux rename-window $OUTDIR

tmux split-window -h
tmux split-window -v
tmux split-window -v
tmux select-pane -t 1
tmux split-window -v
tmux split-window -v
# tmux split-window -v
# tmux split-window -v
tmux select-layout tiled

tmux select-pane -t 1
# Vanilla CEGIS
run $BUFFER --opt-ve-n --opt-pdt-n --opt-feasible-n --opt-wce-n
tmux select-pane -t 2
# VE + pdt
run $BUFFER --opt-feasible-n --opt-wce-n

tmux select-pane -t 3
# WCE only
run $BUFFER --opt-ve-n --opt-pdt-n --opt-feasible-n
tmux select-pane -t 4
# VE + pdt + WCE
run $BUFFER --opt-feasible-n

tmux select-pane -t 5
# VE + pdt + WCE + ideal
run $BUFFER --opt-feasible-n --ideal
tmux select-pane -t 6
# ideal only
run $BUFFER --opt-ve-n --opt-pdt-n --opt-feasible-n --opt-wce-n --ideal

# set +xe
