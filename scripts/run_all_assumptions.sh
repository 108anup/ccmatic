#!/bin/bash

set -xe

run() {
    dut=$1
    ref=$2
    util=$3
    path=./logs/assumptions_that_fix_cca_and_include_loss/$dut-ref_$ref-util$util.csv
    cmd="python -m ccmatic.main_cca_assumption_incal --solution-log-path $path --dut $dut --util $util --ref $ref"
    tmux send-keys "$cmd" Enter
}

tmux split-window -h
tmux split-window -v
tmux split-window -v
tmux select-pane -t 1
tmux split-window -v
tmux split-window -v

tmux select-pane -t 1
run copa paced 0.1
tmux select-pane -t 2
run copa paced 0.45

tmux select-pane -t 3
run copa bbr 0.1
tmux select-pane -t 4
run copa bbr 0.45

tmux select-pane -t 5
run bbr paced 0.1
tmux select-pane -t 6
run bbr paced 0.45

set +xe