## Create eagle draft model

Run

```bash
export OMP_NUM_THREADS=4

export CUDA_VISIBLE_DEVICES=1,4,5,7

export CLEARML_WEB_HOST=
export CLEARML_API_HOST=
export CLEARML_FILES_HOST=
export CLEARML_OFFLINE_MODE=1
export CLEARML_API_ACCESS_KEY=
export CLEARML_API_SECRET_KEY=
export CLEARML_API_HOST_VERIFY_CERT=

torchrun --nnodes=1 --nproc_per_node=4 eagle/train.py \
    --model models/meta-llama2-7b-chat-hf \
    --max-model-len 2048 \
    --steps 80000 \
    --epochs 10 \
    --lr 2e-4 \
    --warmup-steps 2000 \
    --save 4096 \
    --cpdir ./checkpoints \
    --eagle-config ./resources/eagle_config.json \
    --micro-bs 1 \
    --project eagle \
    --task example \
    --noise-low -0.1 \
    --noise-high 0.1 \
    --gradient_accumulation_steps 4 \
    --dataset-path ./sharegpt.jsonl \
    --model-dtype bfloat16 \
    --eagle-dtype bfloat16 \
    --mixed-precision bf16 \
    --optimizer-beta-1 0.9 \
    --optimizer-beta-2 0.95 \
    --grad-clip 0.5 \
    --p-w 0.1 \
    --v-w 1.0
```
