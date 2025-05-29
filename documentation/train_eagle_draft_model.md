## Train

Train model, mps is enabled by default on mac, to disable use --cpu

```bash
accelerate launch --mixed_precision bf16 eagle/train.py \
    --train-input ./tokenized_dataset \
    --test-input ./tokenized_dataset \
    --model /Users/vladislavkruglikov/Projects/download_and_research_eagle/llama2-7b-chat \
    --max-model-len 2048 \
    --epochs 100 \
    --lr 2e-4 \
    --save-freq-steps 10 \
    --cpdir ./checkpoints \
    --eagle-config ./resources/eagle_config.json \
    --micro-bs 1
```

Or docker

```bash
docker run \
    --gpus all \
    -v $(pwd)/resources:/mnt/resources \
    -v $(pwd)/checkpoints:/mnt/checkpoints \
    -v $(pwd)/tokenized_dataset:/mnt/tokenized_dataset \
    -e WANDB_MODE=offline \
    -v /mnt/eagle/models/meta-llama2-7b-chat-hf:/mnt/model \
    eagle \
    accelerate launch eagle/train.py \
    --train-input /mnt/tokenized_dataset \
    --test-input /mnt/tokenized_dataset \
    --model /mnt/model \
    --max-model-len 2048 \
    --epochs 100 \
    --lr 2e-4 \
    --save-freq-steps 10 \
    --cpdir /mnt/checkpoints \
    --eagle-config /mnt/resources/eagle_config.json \
    --micro-bs 1
```
