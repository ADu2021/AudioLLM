
NGPU=$1
SCRIPT=/afs/cs.stanford.edu/u/duyy/data/AudioLLM/TTU/ttu_llama.py
# SCRIPT=/afs/cs.stanford.edu/u/duyy/data/AudioLLM/TTU/ttu_llama_overfit.py

torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=1 \
    --nproc-per-node=$NGPU \
    $SCRIPT