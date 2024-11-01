#!/bin/bash

set -xe

# Call with MONO=--monotonic
MONO="${MONO:-}"
DIR=./logs/assumptions_that_fix_cca-ideal_verifier$MONO
OUTDIR=./outputs/assumptions/assumptions_that_fix_cca-ideal_verifier$MONO
mkdir -p $OUTDIR

filter() {
    dut=$1
    util=$2
    logpath=$DIR/$dut-util$util.csv
    cmd="python -m ccmatic.main_cca_assumption_incal --solution-log-path $logpath --dut $dut --util $util --use-assumption-verifier $MONO --filter-assumptions -o $OUTDIR/$dut-util$util"
    tmux send-keys "$cmd" Enter
}

tmux rename-window analyse-av$MONO

tmux split-window -h
tmux split-window -v
tmux select-pane -t 1
tmux split-window -v

tmux select-layout tiled

tmux select-pane -t 1
filter copa 0.1
tmux select-pane -t 2
filter copa 0.45

tmux select-pane -t 3
filter bbr 0.1
tmux select-pane -t 4
filter bbr 0.45

set +xe