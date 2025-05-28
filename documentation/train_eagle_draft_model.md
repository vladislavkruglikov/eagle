## Train

Train model

```bash
accelerate launch eagle/train.py \
    --train-input ./tokenized_dataset \
    --test-input ./tokenized_dataset \
    --model /Users/vladislavkruglikov/Projects/download_and_research_eagle/llama2-7b-chat \
    --device mps \
    --max-model-len 2048 \
    --epochs 100 \
    --lr 2e-4 \
    --cpdir ./checkpoints \
    --save_freq 10 \
    --eagle-config ./resources/eagle_config.json
```
