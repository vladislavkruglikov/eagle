"""
python3 ./eagle/convert_safetensors_to_bin.py \
    --model ./models/llama2 \
    --safetensors ./eagle.safetensors \
    --config ./eagle_config.json \
    --output ./eagle.bin
"""

import torch
import pathlib
import argparse
import transformers

from eagle.model import Model
from safetensors.torch import load_file


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert model checkpoint from safetensors file to .bin file"
    )
    parser.add_argument(
        "--model",
        type=pathlib.Path,
        required=True,
        help="Path to verificator large model"
    )
    parser.add_argument(
        "--safetensors",
        type=pathlib.Path,
        required=True,
        help="Path to single safetensors eagle checkpoint file"
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        required=True,
        help="Path to eagle config"
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="Path to output eagle checkpoint in bin format"
    )
    return parser.parse_args()


def convert_safetensors_to_bin() -> None:
    args = _parse_arguments()
    safetensors = args.safetensors
    model_path = args.model
    eagle_config_path = args.config
    output = args.output
    config = transformers.AutoConfig.from_pretrained(eagle_config_path)
    model = Model(config, load_emb=True, path=model_path)
    loaded_tensors = load_file(safetensors)
    model.load_state_dict(loaded_tensors)
    torch.save(model.state_dict(), output)


if __name__ == "__main__":
    convert_safetensors_to_bin()
