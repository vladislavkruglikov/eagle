## Deploy eagle draft model in vllm gpu

Run vLLM with eagle checkpoint produced by train script above using docker

```bash
docker run \
  --rm \
  -e VLLM_CPU_KVCACHE_SPACE=4 \
  -e VLLM_USE_V1=0 \
  --name vllm-with-eagle \
  -v /Users/vladislavkruglikov/Projects/download_and_research_eagle/llama2-7b-chat:/mnt/models/llama2-7b-chat \
  -v $(pwd)/checkpoints:/mnt/checkpoints \
  vllm/vllm-openai:v0.9.0 \
  --model /mnt/models/llama2-7b-chat \
  --max-model-len 1024 \
  --speculative-config '{
    "method": "eagle", 
    "model": "/mnt/checkpoints/epoch_100", 
    "num_speculative_tokens": 4
    }'
```

Now you can send requests with prompt model was trained on to verify that acceptance rate is high. You can send this request multiple times such that vllm collect enought statistics and then you can check out metrics

```bash
time curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d @- << 'EOF'
{
  "model": "/Users/vladislavkruglikov/Projects/download_and_research_eagle/llama2-7b-chat",
  "prompt": [1, 518, 25580, 29962, 25538, 592, 29871, 29896, 29900, 29900, 9508, 4128, 393, 306, 508, 6084, 393, 674, 9949, 596, 1962, 29892, 321, 29889, 29887, 29889, 7314, 29892, 16225, 29892, 6036, 29892, 3114, 29892, 20026, 2992, 29889, 518, 29914, 25580, 29962, 18585, 29892, 1244, 526, 29871, 29896, 29900, 29900, 9508, 4128, 393, 366, 508, 6084, 304, 9949, 590, 1962, 29901, 13, 13, 29896, 29889, 4785, 625, 313, 29872, 29889, 29887, 1696, 14263, 470, 12944, 29897, 13, 29906, 29889, 323, 650, 313, 29872, 29889, 29887, 1696, 10676, 29892, 22887, 4384, 293, 29892, 3165, 20657, 29892, 2992, 1846, 13, 29941, 29889, 12577, 313, 29872, 29889, 29887, 1696, 11595, 29892, 1871, 284, 29892, 21567, 29892, 9678, 1288, 29892, 2992, 1846, 13, 29946, 29889, 22135, 313, 29872, 29889, 29887, 1696, 15474, 1230, 29892, 29037, 573, 29892, 429, 7036, 29892, 20408, 294, 573, 29892, 2992, 1846, 13, 29945, 29889, 319, 4749, 663, 313, 29872, 29889, 29887, 1696, 4344, 29892, 16157, 29879, 29892, 2902, 1372, 29892, 1661, 29899, 735, 546, 1372, 29892, 2992, 1846],
  "max_tokens": 32,
  "temperature": 0
}
EOF
```

You can view metric such as acceptance rate in logs or open http://localhost:8000/metrics and find spec decode metrics for acceptance rate
