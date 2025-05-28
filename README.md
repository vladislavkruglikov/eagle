## 🦅 eagle

This repository allows you to train your eagle draft model with ease and prepare required artifacts fully compatible with vLLM to use out of the box. Below I show step by step how to **train your eagle draft model and deploy to vLLM in 3 minutes with around 60% acceptance rate** using simple pre defined dataset for visualization purposes. Then you can swap dataset and increase complexity of your draft model throw using more layers or something. I created this repository because every implementation of eagle was truly hard to run from documentation page and what is more important deploy to vLLM

Below I listed **steps** which you can go through in order **to train and deploy in vLLM your first eagle draft model**

* [Prepare environment](./documentation/prepare_environment.md)
* [Download model](./documentation/download_model.md)
* [Prepare raw chat dataset](./documentation/prepare_raw_chat_dataset.md)
* [Tokenize dataset](./documentation/tokenize_dataset.md)
* [Train eagle draft model](./documentation/train_eagle_draft_model.md)
* Deploy eagle in vLLM
  * [GPU](./documentation/deploy_eagle_gpu.md)
  * [CPU mac os from source](./documentation/deploy_cpu_mac_os_from_source.md)
