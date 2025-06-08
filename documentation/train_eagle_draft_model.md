## Train

Train model, mps is enabled by default on mac, to disable use --cpu

```bash
export CUDA_VISIBLE_DEVCES=0

export CLEARML_OFFLINE_MODE=1
export CLEARML_WEB_HOST=
export CLEARML_API_HOST=
export CLEARML_FILES_HOST=
export CLEARML_API_ACCESS_KEY=
export CLEARML_API_SECRET_KEY=
export CLEARML_API_HOST_VERIFY_CERT=

accelerate launch --num_processes 1 --mixed_precision bf16 eagle/train.py \
    --train-input ./tokenized_dataset \
    --test-input ./tokenized_dataset \
    --model /Users/vladislavkruglikov/Projects/download_and_research_eagle/llama2-7b-chat \
    --max-model-len 2048 \
    --steps 1 \
    --epochs 100 \
    --lr 2e-4 \
    --warmup-steps 4
    --evaluate 4 \
    --save 10 \
    --cpdir ./checkpoints \
    --eagle-config ./resources/eagle_config.json \
    --micro-bs 1 \
    --project eagle \
    --task example
```

To train from checkpoint add `--state ./path_to_checkpoint`

Or docker

```bash
docker run \
    --gpus all \
    -v $(pwd)/resources:/mnt/resources \
    -v $(pwd)/checkpoints:/mnt/checkpoints \
    -v $(pwd)/tokenized_dataset:/mnt/tokenized_dataset \
    -v $(pwd)/models/meta-llama2-7b-chat-hf:/mnt/model \
    -e WANDB_MODE=offline \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e CLEARML_OFFLINE_MODE=1 \
    eagle \
    accelerate launch --num_processes 1 --mixed_precision bf16 eagle/train.py \
    --train-input /mnt/tokenized_dataset \
    --test-input /mnt/tokenized_dataset \
    --model /mnt/model \
    --max-model-len 2048 \
    --steps 10 \
    --epochs 10 \
    --lr 2e-4 \
    --warmup-steps 4 \
    --evaluate 3 \
    --save 10 \
    --cpdir /mnt/checkpoints \
    --eagle-config /mnt/resources/eagle_config.json \
    --micro-bs 1 \
    --project eagle \
    --task example
```
