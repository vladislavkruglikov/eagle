## Download model

Downloading model might require restricted access so you might need to export your token before

```bash
export HF_TOKEN=
huggingface-cli download meta-llama/Llama-2-7b-chat-hf \
    --local-dir ./models/meta-llama2-7b-chat-hf
```

Make sure chate template guards assistant reply with {% generation %} and {% endgeneration %} as shown in [this example](../resources/example_chat_template_with_generation_keyword.json). This is needed for huggingface to generate loss mask correctly such that loss attends only on assistant replies
