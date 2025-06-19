## Train

Train model

```bash
export CUDA_VISIBLE_DEVICES=0

export CLEARML_OFFLINE_MODE=1
export CLEARML_WEB_HOST=
export CLEARML_API_HOST=
export CLEARML_FILES_HOST=
export CLEARML_API_ACCESS_KEY=
export CLEARML_API_SECRET_KEY=
export CLEARML_API_HOST_VERIFY_CERT=

accelerate launch --num_processes 1 --mixed_precision bf16 eagle/train.py \
    --data ./presets/data.yaml \
    --model models/meta-llama2-7b-chat-hf \
    --max-model-len 2048 \
    --steps 80000 \
    --epochs 5 \
    --lr 2e-4 \
    --warmup-steps 10 \
    --evaluate 400000 \
    --save 10000 \
    --cpdir ./checkpoints \
    --eagle-config ./resources/eagle_config.json \
    --micro-bs 16 \
    --project eagle \
    --task example \
    --noise-low -0.1 \
    --noise-high 0.1
```

To train from checkpoint add `--state ./path_to_checkpoint`
