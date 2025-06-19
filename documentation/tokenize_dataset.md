## Tokenize dataset

In this example I describe how to run distributed tokenization process with remote storage. For example suppose we have 2 machines with GPUs which we can use in order to extract hidden states and form a dataset. We can split whole data between that 2 machines. Label them independently in parallel and then merge into single dataset

```bash
export AWS_ACCESS_KEY_ID=
export AWS_SECRET_ACCESS_KEY=
export AWS_ENDPOINT_URL=
```

Label first shard. Label 50 examples starting from index 0

```bash
python ./eagle/prepare_dataset.py \
    --input ./sharegpt.jsonl \
    --model models/meta-llama2-7b-chat-hf \
    --tokenizer models/meta-llama2-7b-chat-hf \
    --device cuda \
    --output s3://your_bucket/dataset/from_0_to_50 \
    --n 50 \
    --start 0
```

Label second shard. Label 50 examples starting from index 50

```bash
python ./eagle/prepare_dataset.py \
    --input ./sharegpt.jsonl \
    --model models/meta-llama2-7b-chat-hf \
    --tokenizer models/meta-llama2-7b-chat-hf \
    --device cuda \
    --output s3://your_bucket/dataset/from_50_to_100 \
    --n 50 \
    --start 50
```

Merge

```bash
python ./eagle/merge_prepared_datasets.py \
    --path s3://your_bucket/dataset
```

Now you can use this path as path to training dataset
