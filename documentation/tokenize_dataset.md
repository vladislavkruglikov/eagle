## Tokenize dataset

Command bellow tokenizes dataset and create all needed masks that will be retrieved during training and also generate hidden states using verifier model such that in training we can use thoose instead of generating them in online training loop

```bash
rm -rf ./tokenized_dataset
python ./eagle/prepare_dataset.py \
    --input ./resources/raw_example_dataset.jsonl \
    --model /Users/vladislavkruglikov/Projects/download_and_research_eagle/llama2-7b-chat \
    --tokenizer /Users/vladislavkruglikov/Projects/download_and_research_eagle/llama2-7b-chat \
    --device mps \
    --output ./tokenized_dataset \
    --frac 1.0
```

Or use docker

```bash
docker run \
    --gpus all \
    -v $(pwd)/resources:/mnt/resources \
    -v /mnt/eagle/models/meta-llama2-7b-chat-hf:/mnt/model \
    -v $(pwd)/tokenized_dataset:/mnt/tokenized_dataset \
    eagle \
    python ./eagle/prepare_dataset.py \
    --input /mnt/resources/raw_example_dataset.jsonl \
    --model /mnt/model \
    --tokenizer /mnt/model \
    --device cuda \
    --output /mnt/tokenized_dataset \
    --frac 1.0
```
