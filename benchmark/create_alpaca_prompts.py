import pathlib
import argparse
import datasets


def create_alpaca_prompts() -> None:
    arguments = _parse_arguments()
    output_path = arguments.output
    n = arguments.n
    frac = arguments.frac
    if n is not None and frac is not None:
        raise ValueError("One of --n or --frac must be set")
    
    print("Loading dataset")
    dataset = datasets.load_dataset("tatsu-lab/alpaca")["train"]
    print(f"Dataset has {len(dataset)} rows")
    dataset = dataset.shuffle(seed=0)
    indices = range(n) if n is not None else range(int(frac * len(dataset)))
    dataset = dataset.select(indices)
    print(f"Dataset after select has {len(dataset)} rows")
    dataset = dataset.map(
        _apply_template,
        batched=False,
        num_proc=1,
        remove_columns=dataset.column_names,
        desc="Applying template"
    )

    print("Saving to disk")
    dataset.to_json(output_path)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create alpaca prompts")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="Path to save prompts"
    )
    parser.add_argument(
        "--n",
        type=int,
        help="Number of sampels to take"
    )
    parser.add_argument(
        "--frac",
        type=float,
        help="Number of sampels to take from 0.0 to 1.0 percent"
    )
    return parser.parse_args()


def _apply_template(example: dict) -> dict:
    if example["input"] == "":
        return {"prompt": _ALPACA_TEMPLATE_WITHOUT_INPUT.format(instruction=example["instruction"])}
    else:
        return {"prompt": _ALPACA_TEMPLATE_WITH_INPUT.format(instruction=example["instruction"], input=example["input"])}


_ALPACA_TEMPLATE_WITH_INPUT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

_ALPACA_TEMPLATE_WITHOUT_INPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


if __name__ == "__main__":
    create_alpaca_prompts()
