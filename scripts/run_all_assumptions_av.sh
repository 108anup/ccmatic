#!/bin/bash

set -xe

# Call with MONO=--monotonic
MONO="${MONO:-}"
DIR=./logs/assumptions_that_fix_cca-ideal_verifier$MONO
mkdir -p $DIR

run() {
    dut=$1
    util=$2
    logpath=$DIR/$dut-util$util.csv
    cmd="python -m ccmatic.main_cca_assumption_incal --solution-log-path $logpath --dut $dut --util $util --use-assumption-verifier $MONO"
    tmux send-keys "$cmd" Enter
}

tmux rename-window av$MONO

tmux split-window -h
tmux split-window -v
tmux select-pane -t 1
tmux split-window -v

tmux select-layout tiled

tmux select-pane -t 1
run copa 0.1
tmux select-pane -t 2
run copa 0.45

tmux select-pane -t 3
run bbr 0.1
tmux select-pane -t 4
run bbr 0.45

set +xe