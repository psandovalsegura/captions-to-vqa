#!/bin/bash

# for load_tiktoken_bpe:
export TIKTOKEN_CACHE_DIR="/scratch0/tiktoken-cache"

# For A4000, max_seq_len=512 and max_batch_size=4 are max values that fit in GPU memory
# For A5000: gpudef 1 2 default rtxa5000
max_seq_len=2048
max_batch_size=1
echo "Max sequence length: $max_seq_len"
echo "Max batch size: $max_batch_size"

# max_seq_len is set to 512 in Github repo
torchrun --nproc_per_node 1 example_chat_completion.py \
    --ckpt_dir /vulcanscratch/psando/llama-3/Meta-Llama-3-8B-Instruct/ \
    --tokenizer_path /vulcanscratch/psando/llama-3/Meta-Llama-3-8B-Instruct/tokenizer.model \
    --max_seq_len $max_seq_len --max_batch_size $max_batch_size
