#!/bin/bash

NGPU=1
SCRIPT=/afs/cs.stanford.edu/u/duyy/data/AudioLLM/demo/app.py

torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=1 \
    --nproc-per-node=$NGPU \
    $SCRIPT