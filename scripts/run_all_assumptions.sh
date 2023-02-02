#!/bin/bash

set -xe

MONO="${MONO:---monotonic}"
DIR=./logs/assumptions_that_fix_cca$MONO
mkdir -p $DIR


run() {
    dut=$1
    ref=$2
    util=$3
    logpath=$DIR/$dut-ref_$ref-util$util.csv
    cmd="python -m ccmatic.main_cca_assumption_incal --solution-log-path $logpath --dut $dut --util $util --ref $ref $MONO"
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