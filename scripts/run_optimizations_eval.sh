#!/bin/bash

set -xe

BUFFER="${BUFFER:---dynamic-buffer}"

run() {
    args=$@
    echo $args
    cmd="python -m ccmatic.main_cca_belief_template -T 9 $args"
    # cmd="echo $args"
    tmux send-keys "$cmd" Enter
}

tmux rename-window opt-eval$BUFFER

tmux split-window -h
tmux split-window -v
tmux split-window -v
tmux select-pane -t 1
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

tmux select-layout tiled

set +xe