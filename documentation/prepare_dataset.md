## Prepare dataset

In order to train models with our frameworks you need to supply dataset in particular format. It is jsonlines where each lines looks like this. Basically you have a list of messages and each message has 2 keys which are role and content

```json
{"id": 0, "messages": [{"role": "user", "content": "Give me 100 prompt parameters that I can specify that will influence your output, e.g. voice, tone, register, style, audience etc."}, {"role": "assistant", "content": "Sure, here are 100 prompt parameters that you can specify to influence my output:\n\n1. Voice (e.g., male or female)\n2. Tone (e.g., serious, sarcastic, humorous, etc.)"}, {"role": "user", "content": "Continue"}, {"role": "assistant", "content": "3. Timing (e.g., pacing, pauses, etc.)\n4. Emphasis (e.g., stress, intonation, etc.)"}]}
```
