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
    output_dir: pathlib.Path = arguments.output

    print("Loading tokenizer")
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)

    print("Loading raw dataset")
    dataset = (
        datasets
        .load_dataset("json", data_files={"train": [str(raw_path)]})
        ["train"]
        .shuffle(seed=0)
    )

    print("Tokenizing dataset")
    dataset = dataset.map(
        lambda example: _tokenize_dataset(example=example, tokenizer=tokenizer),
        batched=False,
        num_proc=1,
        remove_columns=dataset.column_names
    )

    print("Convert dataset to torch tensors")
    dataset.set_format(type="torch")

    print("Loading model")
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path,  device_map=device, torch_dtype="auto").eval()
    
    if not output_dir.exists():
        output_dir.mkdir()
    
    print("Generating hidden states and saving checkpoints")
    for i, example in enumerate(dataset):
        enriched_example = _enrich_with_hidden_state(example=example, model=model, device=device)
        print(f"Saving {output_dir}/{i}.ckpt")
        torch.save(enriched_example, f'{output_dir}/{i}.ckpt')


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process a raw chat dataset using a verifier model and tokenizer."
    )
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        required=True,
        help="Path to JSON lines chat dataset as described in documentation"
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
        help="Directory where the processed dataset will be stored"
    )
    return parser.parse_args()


def _tokenize_dataset(example: dict, tokenizer: transformers.AutoTokenizer) -> dict[str, torch.LongTensor]:
    result = tokenizer.apply_chat_template(
        example["messages"], 
        tokenize=True, 
        add_generation_prompt=False, 
        return_dict=True,
        return_assistant_tokens_mask=True
    )

    return {
        "input_ids": result["input_ids"],
        "loss_mask": result["assistant_masks"]
    }  


@torch.no_grad()
def _enrich_with_hidden_state(example: dict[str, torch.LongTensor], model: transformers.AutoModelForCausalLM, device: str) -> dict:
    input_ids = example["input_ids"].to(device).unsqueeze(0)
    outs_big = model(input_ids, output_hidden_states=True)
    hidden_state_big = outs_big.hidden_states[-1][0]
    return example | {"hidden_state": hidden_state_big}


if __name__ == "__main__":
    _prepare_dataset()
