## Prepare environment 

Use pre built docker instead

```bash
docker pull vladislavkruglikov/eagle:latest
```

Or build docker yourself

```bash
docker build --tag eagle -f docker/Dockerfile .
```

Or create local environment and install current package

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```
