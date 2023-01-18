#!/bin/bash

set -xe

run() {
    dut=$1
    util=$2
    path=./logs/assumptions_that_fix_cca_and_include_ideal/$dut-util$util.csv
    cmd="python -m ccmatic.main_cca_assumption_incal --solution-log-path $path --dut $dut --util $util --use-assumption-verifier"
    tmux send-keys "$cmd" Enter
}

tmux split-window -h
tmux split-window -v
tmux select-pane -t 1
tmux split-window -v

tmux select-pane -t 1
run copa 0.1
tmux select-pane -t 2
run copa 0.45

tmux select-pane -t 3
run bbr 0.1
tmux select-pane -t 4
run bbr 0.45

set +xe