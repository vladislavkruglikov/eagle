## Prepare environment 

You can build a docker 

```bash
docker build --tag eagle -f docker/Dockerfile .
```

Or use already built docker

```bash
docker pull vladislavkruglikov/eagle:latest
```

But in this get started guide for simplicity I will go with virtual environment. Migration to docker is simple and easy

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```
