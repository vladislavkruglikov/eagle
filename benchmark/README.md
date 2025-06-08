## 📊 Benchmark

Create alpaca prompts

```bash
python3 ./benchmark/create_alpaca_prompts.py \
    --output ./benchmark/alpaca.jsonl \
    --n 128
```

Benchmark base model with batch size 1 meaning only 1 request runs at most at the same time

```bash
docker run \
    --gpus all \
    -e CUDA_VISIBLE_DEVICES=0 \
    -v ./models/meta-llama2-7b-chat-hf:/mnt/llama2-7b \
    -v ./eagle_model:/mnt/eagle \
    -v ./benchmark:/opt/benchmark \
    --ipc=host \
    --shm-size 32g \
    lmsysorg/sglang:v0.4.6.post5-cu124 \
    bash -c "cd /opt && export PYTHONPATH=$PYTHONPATH:. && python3 benchmark/benchmark.py \
        --model /mnt/llama2-7b \
        --prompts benchmark/alpaca.jsonl \
        --n 64 \
        --bs 1 \
        --output benchmark/report_alpaca_bs1_wo_eagle.json \
        --temperature 0
    "
```

Benchmark base model with draft model with batch size 1 meaning only 1 request runs at most at the same time

```bash
docker run \
    --gpus all \
    -e CUDA_VISIBLE_DEVICES=1 \
    -v ./models/meta-llama2-7b-chat-hf:/mnt/llama2-7b \
    -v ./eagle_model:/mnt/eagle \
    -v ./benchmark:/opt/benchmark \
    --ipc=host \
    --shm-size 32g \
    lmsysorg/sglang:v0.4.6.post5-cu124 \
    bash -c "cd /opt && export PYTHONPATH=$PYTHONPATH:. && python3 benchmark/benchmark.py \
        --model /mnt/llama2-7b \
        --prompts benchmark/alpaca.jsonl \
        --n 64 \
        --bs 1 \
        --output benchmark/report_alpaca_bs1_with_eagle_new_new.json \
        --eagle /mnt/eagle \
        --steps 4 \
        --k 1 \
        --draft 4 \
        --speculative-algorithm EAGLE \
        --temperature 0
    "
```
