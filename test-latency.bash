#!/bin/bash

# passkey experiment
GPUS=0
WINDOW=16384
# CASCADES=4
CASCADES=1
SINKS=4
BATCH_SIZE=1

HEAD_REDUCTION=max
# MODEL=llama3.1-8b
METHODS=(vanilla sink sink sink sink sink sink sink sink sink)
CASCADE_STRIDES=(1 128 256 512 1024 2048 4096 8192 16384 1)

# MODEL=llama7b-chat
# METHODS=(h2o)
# CASCADE_STRIDES=(1)

MODEL=llama3.1-8b-instruct
# MODEL=llama7b-chat
# METHODS=(snapkv)
# CASCADE_STRIDES=(1)
# CASCADE_STRIDES=(114684)
# METHODS=(minference)
# METHODS=(sink)
#COMMENT=minference-cascade-rebuttal

for i in "${!METHODS[@]}";
do 
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=$GPUS python cascade/main/llama_eval.py \
        --model $MODEL \
        --job latency \
        --method ${METHODS[$i]} \
        --lora_r 0 \
        --window $WINDOW \
        --sinks $SINKS \
        --cascade_stride ${CASCADE_STRIDES[$i]} \
        --head_reduction $HEAD_REDUCTION \
        --cascades $CASCADES \
        --comment $COMMENT \
        --batch_size $BATCH_SIZE
        
        sleep 1
done
