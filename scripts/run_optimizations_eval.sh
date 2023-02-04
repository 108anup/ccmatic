#!/bin/bash

# set -xe

BUFFER="${BUFFER:---dynamic-buffer}"
BUFFER=""

run() {
    args=$@
    echo $args
    cmd="timeout 7d python -m ccmatic.main_cca_belief_template -T 9 $args"
    # cmd="timeout 7d python -m ccmatic.main_cca_lossless_ccmatic $args"
    # cmd="echo $args"
    tmux send-keys "$cmd" Enter
}

tmux rename-window lossless-opt-eval$BUFFER

tmux split-window -v
tmux split-window -v
tmux split-window -v
tmux split-window -v
tmux split-window -v
tmux split-window -v
tmux split-window -v
tmux select-layout tiled

tmux select-pane -t 1
# Z3 quantifier
run $BUFFER --opt-cegis-n --opt-ve-n --opt-pdt-n --opt-feasible-n --opt-wce-n
tmux select-pane -t 2
# Vanilla CEGIS
run $BUFFER --opt-ve-n --opt-pdt-n --opt-feasible-n --opt-wce-n

tmux select-pane -t 3
# VE only
run $BUFFER --opt-pdt-n --opt-feasible-n --opt-wce-n
tmux select-pane -t 4
# VE + pdt
run $BUFFER --opt-feasible-n --opt-wce-n

tmux select-pane -t 5
# VE + pdt + feasible
run $BUFFER --opt-wce-n
tmux select-pane -t 6
# VE + pdt + WCE
run $BUFFER --opt-feasible-n

tmux select-pane -t 7
# VE + pdt + WCE + ideal
run $BUFFER --opt-feasible-n --ideal
tmux select-pane -t 8
# VE + pdt + feasible + WCE + ideal
run $BUFFER --ideal

# set +xe
