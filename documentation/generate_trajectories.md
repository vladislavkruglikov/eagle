## Generate assistant responses

It might be better to generate responses using verifier to do so run script

```bash
python ./eagle/generate_trajectories.py \
    --input ./resources/raw_example_dataset_ends_in_user.jsonl \
    --model /Users/vladislavkruglikov/Projects/download_and_research_eagle/llama2-7b-chat \
    --tokenizer /Users/vladislavkruglikov/Projects/download_and_research_eagle/llama2-7b-chat \
    --device mps \
    --output ./resources/raw_example_dataset_ends_in_user_with_trajectories.jsonl \
    --max-new-tokens 32 \
    --frac 1.0
```
