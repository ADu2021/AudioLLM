#!/bin/bash

# args
CKPT_FOLDER=/sailhome/duyy/data/checkpoints/TTU/ckpt-2024-09-30_09-45-16-llama3-UnitEmbed-CANINE_sum-Continue-Double-NGPU-2_BS-32_LR-1e-05-Warmup-0-EPOCH-6-EMBED-1280-FFN-2048-DR-0.1-NH-8-NL-6-independent_tgt_mask-independent_mem_mask

NGPU=1
SCRIPT=/afs/cs.stanford.edu/u/duyy/data/AudioLLM/TTU/ttu_llama_canine_inference.py

for ckpt_path in $(ls $CKPT_FOLDER); do
    if ! [[ $ckpt_path == epoch-*\.bin ]]; then
        continue
    fi
    # if ! [[ $ckpt_path == epoch-0*.bin ]] && ! [[ $ckpt_path == epoch-1*.bin ]]; then
    #     continue
    # fi
    model_path=${CKPT_FOLDER}/${ckpt_path}
    output_path=$CKPT_FOLDER
    echo $model_path

    torchrun \
        --rdzv-backend=c10d \
        --rdzv-endpoint=localhost:0 \
        --nnodes=1 \
        --nproc-per-node=$NGPU \
        $SCRIPT \
        --model_path $model_path \
        --output_path $output_path
    
done

