## Train

Train model

```bash
accelerate launch --cpu eagle/train.py \
    --train-input ./tokenized_dataset \
    --test-input ./tokenized_dataset \
    --model /mnt/eagle/models/meta-llama2-7b-chat-hf \
    --device cuda \
    --max-model-len 2048 \
    --epochs 100 \
    --lr 2e-4 \
    --cpdir ./checkpoints \
    --save_freq 10 \
    --eagle-config ./resources/eagle_config.json
```

Or docker

```bash
docker run \
    --gpus all \
    -v $(pwd)/resources:/mnt/resources \
    -v $(pwd)/checkpoints:/mnt/checkpoints \
    -v $(pwd)/tokenized_dataset:/mnt/tokenized_dataset \
    -v /mnt/eagle/models/meta-llama2-7b-chat-hf:/mnt/model \
    eagle \
    python ./eagle/train.py \
    --train-input /mnt/tokenized_dataset \
    --test-input /mnt/tokenized_dataset \
    --model /mnt/model \
    --device cuda \
    --max-model-len 2048 \
    --epochs 100 \
    --lr 2e-4 \
    --cpdir /mnt/checkpoints \
    --save_freq 10 \
    --eagle-config /mnt/resources/eagle_config.json
```
