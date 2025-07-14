#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Function to print usage
print_usage() {
    echo "Usage: $0 <mode> <cmd>"
    echo "  mode: prefill or decode"
    echo "  cmd:  dynamo or sglang"
    echo ""
    echo "Examples:"
    echo "  $0 prefill dynamo"
    echo "  $0 decode sglang"
    exit 1
}

# Check if correct number of arguments provided
if [ $# -ne 2 ]; then
    echo "Error: Expected 2 arguments, got $#"
    print_usage
fi

# Parse arguments
mode=$1
cmd=$2
LOG_DIR="./logs"

# Validate mode argument
if [ "$mode" != "prefill" ] && [ "$mode" != "decode" ]; then
    echo "Error: mode must be 'prefill' or 'decode', got '$mode'"
    print_usage
fi

# Validate cmd argument
if [ "$cmd" != "dynamo" ] && [ "$cmd" != "sglang" ]; then
    echo "Error: cmd must be 'dynamo' or 'sglang', got '$cmd'"
    print_usage
fi

echo "Mode: $mode"
echo "Command: $cmd"


# Check if required environment variables are set
if [ -z "$HOST_IP" ]; then
    echo "Error: HOST_IP environment variable is not set"
    exit 1
fi

if [ -z "$PORT" ]; then
    echo "Error: PORT environment variable is not set"
    exit 1
fi

if [ -z "$TOTAL_GPUS" ]; then
    echo "Error: TOTAL_GPUS environment variable is not set"
    exit 1
fi

if [ -z "$RANK" ]; then
    echo "Error: RANK environment variable is not set"
    exit 1
fi

if [ -z "$TOTAL_NODES" ]; then
    echo "Error: TOTAL_NODES environment variable is not set"
    exit 1
fi

# Construct command based on mode and cmd
if [ "$mode" = "prefill" ]; then
    if [ "$cmd" = "dynamo" ]; then
        # H100 dynamo prefill command
        VLLM_ALL2ALL_BACKEND="deepep_low_latency" \
        VLLM_USE_DEEP_GEMM=1 \
        VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1 \
        python3 examples/vllm_v1/components/main.py \
        --model deepseek-ai/DeepSeek-R1 \
        --data_parallel_size $TOTAL_GPUS \
        --data-parallel-rank $RANK \
        --enable-expert-parallel \
        --max-model-len 10240 \
        --data-parallel-address $HOST_IP \
        --data-parallel-rpc-port 13345 \
        --gpu-memory-utilization 0.95 \
        --enforce-eager \
        --is-prefill-worker \
        --kv-events-port 49700 2>&1 | tee $LOG_DIR/dsr1_dep_${dp_rank}.log &
    elif [ "$cmd" = "sglang" ]; then
        # H100 sglang prefill command
        echo "Error: sglang command not implemented here"
        exit 1
    fi
elif [ "$mode" = "decode" ]; then
    if [ "$cmd" = "dynamo" ]; then
        # H100 dynamo decode command
        VLLM_ALL2ALL_BACKEND="deepep_low_latency" \
        VLLM_USE_DEEP_GEMM=1 \
        VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1 \
        python3 examples/vllm_v1/components/main.py \
        --model deepseek-ai/DeepSeek-R1 \
        --data_parallel_size $TOTAL_GPUS \
        --data-parallel-rank $RANK \
        --enable-expert-parallel \
        --max-model-len 10240 \
        --data-parallel-address $HOST_IP \
        --data-parallel-rpc-port 13345 \
        --gpu-memory-utilization 0.95 \
        --enforce-eager \
        --kv-events-port 49700 2>&1 | tee $LOG_DIR/dsr1_dep_${dp_rank}.log &
    elif [ "$cmd" = "sglang" ]; then
        # H100 sglang decode command
        echo "Error: sglang command not implemented here"
        exit 1
    fi
fi


