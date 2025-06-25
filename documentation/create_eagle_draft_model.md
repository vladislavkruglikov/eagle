## Create eagle draft model

Training script will log data into clearml dashboards. If you want to disable it (which is reccomended for first try run script) run

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

In this page I will share scripts to run to train your eagle draft model

* Make sure loss does not change a lot with scale
* Select subset of data and optimize hyperparameters for small subset then scale

## 1 gpu accum 1 micro 8 global 8

```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    ./eagle/train.py \
        --micro-batch-size 8 \
        --gradient-accumulation-steps 1 \
        --num-warmup-steps 8 \
        --num-training-steps 32 \
        --epochs 8 \
        --clearml-project eagle \
        --clearml-task 1gpus-8microbs-1accum \
        --verifier-model-path /mnt/v.kruglikov/reproduce_old_eagle/models/llama2 \
        --dataset-path ./sharegpt.jsonl \
        --eagle-config-path ./resources/eagle_config.json \
        --learning-rate 1e-4 \
        --maximum-model-length 2048 \
        --noise-low -0.1 \
        --noise-high 0.1 \
        --v-w 1.0 \
        --p-w 0.1 \
        --grad-clip 0.5 \
        --b1 0.9 \
        --b2 0.95 \
        --cpdir ./checkpoints \
        --save 4096 \
        --mixed-precision bf16 \
        --verifier-model-lm-head-dtype bfloat16 \
        --verifier-model-dtype bfloat16 \
        --eagle-dtype bfloat16 \
        --attn flash_attention_2
```

## 1 gpu accum 2 micro 4 global 8

```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    ./eagle/train.py \
        --micro-batch-size 4 \
        --gradient-accumulation-steps 2 \
        --num-warmup-steps 8 \
        --num-training-steps 32 \
        --epochs 8 \
        --clearml-project eagle \
        --clearml-task 1gpus-4microbs-2accum \
        --verifier-model-path /mnt/v.kruglikov/reproduce_old_eagle/models/llama2 \
        --dataset-path ./sharegpt.jsonl \
        --eagle-config-path ./resources/eagle_config.json \
        --learning-rate 1e-4 \
        --maximum-model-length 2048 \
        --noise-low -0.1 \
        --noise-high 0.1 \
        --v-w 1.0 \
        --p-w 0.1 \
        --grad-clip 0.5 \
        --b1 0.9 \
        --b2 0.95 \
        --cpdir ./checkpoints \
        --save 4096 \
        --mixed-precision bf16 \
        --verifier-model-lm-head-dtype bfloat16 \
        --verifier-model-dtype bfloat16 \
        --eagle-dtype bfloat16 \
        --attn flash_attention_2
```

## 2 gpu accum 2 micro 2 global 8

```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    ./eagle/train.py \
        --micro-batch-size 2 \
        --gradient-accumulation-steps 2 \
        --num-warmup-steps 16 \
        --num-training-steps 64 \
        --epochs 8 \
        --clearml-project eagle \
        --clearml-task 2gpus-2microbs-2accum \
        --verifier-model-path /mnt/v.kruglikov/reproduce_old_eagle/models/llama2 \
        --dataset-path ./sharegpt.jsonl \
        --eagle-config-path ./resources/eagle_config.json \
        --learning-rate 1e-4 \
        --maximum-model-length 2048 \
        --noise-low -0.1 \
        --noise-high 0.1 \
        --v-w 1.0 \
        --p-w 0.1 \
        --grad-clip 0.5 \
        --b1 0.9 \
        --b2 0.95 \
        --cpdir ./checkpoints \
        --save 4096 \
        --mixed-precision bf16 \
        --verifier-model-lm-head-dtype bfloat16 \
        --verifier-model-dtype bfloat16 \
        --eagle-dtype bfloat16 \
        --attn flash_attention_2
```
