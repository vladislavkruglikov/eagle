## Reproduce eagle 1 paper

Page will cover all steps neede to reproduce eagle 1 paper for model [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

## Docker

In order to use tool you need to build docker

```bash
docker build --tag eagle -f docker/Dockerfile .
```

Or use already build one

```bash
docker pull vladislavkruglikov/eagle:latest
```

## Model

Downloading model might require restricted access so you might need to export your token before

```bash
export HF_TOKEN=
huggingface-cli download meta-llama/Llama-2-7b-chat-hf \
    --local-dir ./resources/meta-llama2-7b-chat-hf
```

Make sure chate template guards assistant reply with {% generation %} and {% endgeneration %} as shown in [this example](../resources/example_chat_template_with_generation_keyword.json). This is needed for huggingface to generate loss mask correctly such that loss attends only on assistant replies

## Prepare dataset

In order to train models with our frameworks you need to supply dataset in particular format. It is jsonlines where each lines looks like this. Basically you have a list of messages and each message has 2 keys which are role and content

```json
{"id": 0, "messages": [{"role": "user", "content": "Give me 100 prompt parameters that I can specify that will influence your output, e.g. voice, tone, register, style, audience etc."}, {"role": "assistant", "content": "Sure, here are 100 prompt parameters that you can specify to influence my output:\n\n1. Voice (e.g., male or female)\n2. Tone (e.g., serious, sarcastic, humorous, etc.)"}, {"role": "user", "content": "Continue"}, {"role": "assistant", "content": "3. Timing (e.g., pacing, pauses, etc.)\n4. Emphasis (e.g., stress, intonation, etc.)"}]}
```

In order to reproduce eagle 1 paper we will be using pre built script that downloads sharegpt dataset and formats as reference proposes

```bash
docker run \
    -v ./resources:/mnt/resources \
    eagle \
    python3 eagle/prepare_sharegpt_dataset.py \
    --frac 1.0 \
    --output /mnt/resources/sharegpt.jsonl
```

## Train draft model

I will demonstrate how to train draft model for particular configuration but you can twick it under your needs. Training script will log data into clearml dashboards. If you want to disable it (which is reccomended for first try run script) run

```bash
export CLEARML_OFFLINE_MODE=1
```


Otherwise specify this environments

```bash
export CLEARML_WEB_HOST=
export CLEARML_API_HOST=
export CLEARML_FILES_HOST=
export CLEARML_API_ACCESS_KEY=
export CLEARML_API_SECRET_KEY=
export CLEARML_API_HOST_VERIFY_CERT=
```

Also it is up to you to configure

```bash
export OMP_NUM_THREADS=4 
export CUDA_VISIBLE_DEVICES=7 
```

I will demonstrate once again a very simple setup which will probably be the most popular. For that you need single gpu with roughly 80 HBM

```bash
docker run \
    --gpus all \
    -e OMP_NUM_THREADS=4 \
    -e CUDA_VISIBLE_DEVICES=7 \
    -e CLEARML_OFFLINE_MODE=1 \
    -v ./resources:/mnt/resources \
    eagle \
    torchrun \
        --nnodes=1 \
        --nproc_per_node=1 \
        eagle/train.py \
            --micro-batch-size 8 \
            --gradient-accumulation-steps 2 \
            --num-warmup-steps 1024 \
            --num-training-steps 131072 \
            --epochs 4 \
            --clearml-project eagle \
            --clearml-task 1gpus-8microbs-2accum-16globalbs \
            --verifier-model-path /mnt/resources/meta-llama2-7b-chat-hf \
            --dataset-path /mnt/resources/sharegpt.jsonl \
            --eagle-config-path /mnt/resources/eagle_config.json \
            --learning-rate 2e-4 \
            --maximum-model-length 2048 \
            --noise-low -0.1 \
            --noise-high 0.1 \
            --v-w 1.0 \
            --p-w 0.1 \
            --grad-clip 0.5 \
            --b1 0.9 \
            --b2 0.95 \
            --cpdir /mnt/resources/checkpoints \
            --save 4096 \
            --mixed-precision bf16 \
            --verifier-model-lm-head-dtype bfloat16 \
            --verifier-model-dtype bfloat16 \
            --eagle-dtype bfloat16 \
            --attn flash_attention_2
```

For thoose who have access to 2 GPUs

```bash
docker run \
    --gpus all \
    -e OMP_NUM_THREADS=4 \
    -e CUDA_VISIBLE_DEVICES=5,7 \
    -e CLEARML_OFFLINE_MODE=1 \
    -v ./resources:/mnt/resources \
    eagle \
    torchrun \
        --nnodes=1 \
        --nproc_per_node=2 \
        eagle/train.py \
            --micro-batch-size 4 \
            --gradient-accumulation-steps 2 \
            --num-warmup-steps 2048 \
            --num-training-steps 262144 \
            --epochs 4 \
            --clearml-project eagle \
            --clearml-task 2gpus-4microbs-2accum-16globalbs \
            --verifier-model-path /mnt/resources/meta-llama2-7b-chat-hf \
            --dataset-path /mnt/resources/sharegpt.jsonl \
            --eagle-config-path /mnt/resources/eagle_config.json \
            --learning-rate 2e-4 \
            --maximum-model-length 2048 \
            --noise-low -0.1 \
            --noise-high 0.1 \
            --v-w 1.0 \
            --p-w 0.1 \
            --grad-clip 0.5 \
            --b1 0.9 \
            --b2 0.95 \
            --cpdir /mnt/resources/checkpoints \
            --save 4096 \
            --mixed-precision bf16 \
            --verifier-model-lm-head-dtype bfloat16 \
            --verifier-model-dtype bfloat16 \
            --eagle-dtype bfloat16 \
            --attn flash_attention_2
```

For thoose who have access to 4 GPUs

```bash
docker run \
    --gpus all \
    -e OMP_NUM_THREADS=4 \
    -e CUDA_VISIBLE_DEVICES=1,3,5,7 \
    -e CLEARML_OFFLINE_MODE=1 \
    -v ./resources:/mnt/resources \
    eagle \
    torchrun \
        --nnodes=1 \
        --nproc_per_node=4 \
        eagle/train.py \
            --micro-batch-size 2 \
            --gradient-accumulation-steps 2 \
            --num-warmup-steps 4096 \
            --num-training-steps 524288 \
            --epochs 4 \
            --clearml-project eagle \
            --clearml-task 4gpus-2microbs-2accum-16globalbs \
            --verifier-model-path /mnt/resources/meta-llama2-7b-chat-hf \
            --dataset-path /mnt/resources/sharegpt.jsonl \
            --eagle-config-path /mnt/resources/eagle_config.json \
            --learning-rate 2e-4 \
            --maximum-model-length 2048 \
            --noise-low -0.1 \
            --noise-high 0.1 \
            --v-w 1.0 \
            --p-w 0.1 \
            --grad-clip 0.5 \
            --b1 0.9 \
            --b2 0.95 \
            --cpdir /mnt/resources/checkpoints \
            --save 4096 \
            --mixed-precision bf16 \
            --verifier-model-lm-head-dtype bfloat16 \
            --verifier-model-dtype bfloat16 \
            --eagle-dtype bfloat16 \
            --attn flash_attention_2
```
