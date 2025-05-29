import torch
import pathlib
import argparse
import datasets
import transformers


def _prepare_dataset() -> None:
    arguments = _parse_arguments()
    
    raw_path: pathlib.Path = arguments.input
    model_path: pathlib.Path = arguments.model
    tokenizer_path: pathlib.Path = arguments.tokenizer
    device: str = arguments.device
    output_path: pathlib.Path = arguments.output
    max_new_tokens = arguments.max_new_tokens
    n = arguments.n
    frac = arguments.frac
    if n is not None and frac is not None:
        raise ValueError("One of --n or --frac must be set")

    print("Loading tokenizer")
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)

    print("Loading raw dataset")
    dataset = (
        datasets
        .load_dataset("json", data_files={"train": [str(raw_path)]})
        ["train"]
        .shuffle(seed=0)
    )
    print(f"Dataset has {len(dataset)} rows")
    indices = range(n) if n is not None else range(int(frac * len(dataset)))
    dataset = dataset.select(indices)
    print(f"Dataset after select has {len(dataset)} rows")

    dataset = dataset.map(
        lambda example: _tokenize_dataset(example=example, tokenizer=tokenizer),
        batched=False,
        num_proc=1,
        desc="Tokenizing dataset"
    )

    print("Convert dataset to torch tensors")
    dataset.set_format(type="torch")

    print("Loading model")
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path,  device_map=device, torch_dtype="auto").eval()

    dataset = dataset.map(
        lambda example: _enrich_messages(example=example, tokenizer=tokenizer, model=model, device=device, max_new_tokens=max_new_tokens),
        batched=False,
        num_proc=1,
        desc="Generating assistant responses"
    )
    
    print("Saving to disk")
    dataset.select_columns(["id", "messages"]).to_json(output_path)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate trajectories"
    )
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        required=True,
        help="Path to JSON lines chat dataset as described in documentation wheres lines end with user response"
    )
    parser.add_argument(
        "--model",
        type=pathlib.Path,
        required=True,
        help="Path to verifier model"
    )
    parser.add_argument(
        "--tokenizer",
        type=pathlib.Path,
        required=True,
        help="Path to tokenizer, usually the same as the model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device that will be used by the large model to generate hidden states (e.g., 'cpu', 'cuda:0')"
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="Path to jsonlines file where the processed dataset will be stored"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max new tokens generated"
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


def _tokenize_dataset(example: dict, tokenizer: transformers.AutoTokenizer) -> dict[str, torch.LongTensor]:
    result = tokenizer.apply_chat_template(
        example["messages"], 
        tokenize=True, 
        add_generation_prompt=True, 
        return_dict=True
    )

    return {
        "input_ids": result["input_ids"],
        "messages": example["messages"], 
    }


@torch.no_grad()
def _enrich_messages(example: dict, tokenizer: transformers.AutoTokenizer, model: transformers.AutoModelForCausalLM, device: str, max_new_tokens: int) -> dict:
    outputs = model.generate(
        example["input_ids"].unsqueeze(0).to(device), 
        do_sample=False,
        max_new_tokens=max_new_tokens
    ) 
    prompt_tokens_count = example["input_ids"].shape[0]
    assistant_tokens = outputs[0][prompt_tokens_count:]
    assistant_text = tokenizer.decode(assistant_tokens)
    example["messages"].append({"role": "assistant", "content": assistant_text})
    return example


if __name__ == "__main__":
    _prepare_dataset()
