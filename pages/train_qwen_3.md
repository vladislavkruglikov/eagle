## Train qwen3

```bash
huggingface-cli download Qwen/Qwen3-8B --local-dir ./resources/Qwen/Qwen3-8B
```

```bash
docker build --tag eagle -f docker/Dockerfile . && docker run \
    --gpus all \
    -e OMP_NUM_THREADS=4 \
    -e CUDA_VISIBLE_DEVICES=1,2,5,7 \
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
            --verifier-model-path /mnt/resources/qwen3-8b \
            --dataset-path /mnt/resources/sharegpt.jsonl \
            --eagle-config-path /mnt/resources/eagle_config_qwen3_8b.json \
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

```bash
docker build --tag eagle -f docker/Dockerfile . && docker run \
    --gpus all \
    -e CUDA_VISIBLE_DEVICES=1,2,5,7 \
    -e CLEARML_OFFLINE_MODE=1 \
    -v ./resources:/mnt/resources \
    eagle python3 ./eagle/train_tp.py \
            --micro-batch-size 2 \
            --gradient-accumulation-steps 2 \
            --num-warmup-steps 4096 \
            --num-training-steps 524288 \
            --epochs 4 \
            --clearml-project eagle \
            --clearml-task 4gpus-2microbs-2accum-16globalbs \
            --verifier-model-path /mnt/resources/qwen3-8b \
            --dataset-path /mnt/resources/sharegpt.jsonl \
            --eagle-config-path /mnt/resources/eagle_config_qwen3_8b.json \
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
