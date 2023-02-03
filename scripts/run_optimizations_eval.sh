#!/bin/bash

# set -xe

BUFFER="${BUFFER:---dynamic-buffer}"
BUFFER=""

run() {
    args=$@
    echo $args
    # cmd="timeout 7d python -m ccmatic.main_cca_belief_template -T 9 $args"
    cmd="timeout 7d python -m ccmatic.main_cca_lossless_ccmatic $args"
    # cmd="echo $args"
    tmux send-keys "$cmd" Enter
}

tmux rename-window lossless-opt-eval$BUFFER

tmux split-window -h
tmux split-window -v
tmux split-window -v
tmux select-pane -t 1
tmux split-window -v
tmux split-window -v
tmux split-window -v
tmux split-window -v

tmux select-pane -t 1
run $BUFFER --opt-cegis-n --opt-ve-n --opt-pdt-n --opt-wce-n --opt-feasible-n
tmux select-pane -t 2
run $BUFFER --opt-ve-n --opt-pdt-n --opt-wce-n --opt-feasible-n

tmux select-pane -t 3
run $BUFFER --opt-pdt-n --opt-wce-n --opt-feasible-n
tmux select-pane -t 4
run $BUFFER --opt-wce-n --opt-feasible-n

tmux select-pane -t 5
run $BUFFER --opt-feasible-n
tmux select-pane -t 6
run $BUFFER

tmux select-pane -t 7
run $BUFFER --ideal
tmux select-pane -t 8
run $BUFFER --ideal --opt-feasible-n

tmux select-layout tiled

# set +xe
