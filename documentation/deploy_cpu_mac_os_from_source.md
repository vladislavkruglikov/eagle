## Deploy vllm eagle draft model on mac os

Build from source actual for mac os users

```bash
git clone https://github.com/vllm-project/vllm
cd vllm
python3 -m venv venv
source venv/bin/activate
pip install -r requirements/cpu.txt
pip install -e . 
```

and run

```bash
VLLM_CPU_KVCACHE_SPACE=4 VLLM_USE_V1=0 python3 -m vllm.entrypoints.openai.api_server \
    --model /Users/vladislavkruglikov/Projects/download_and_research_eagle/llama2-7b-chat \
    --max-model-len 1024 \
    --host 0.0.0.0 \
    --port 8000 \
    --speculative-config '{
        "method": "eagle", 
        "model": "/Users/vladislavkruglikov/Projects/my_eagle/checkpoints/epoch_100", 
        "num_speculative_tokens": 4
    }
```

With cpu model runned u might encounter  AttributeError: 'CPUModelRunner' object has no attribute 'model_runner'. u need to manualy remove this double reference from vllm
