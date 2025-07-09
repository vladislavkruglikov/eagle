import torch
import sglang
import asyncio
import pathlib
import argparse
import datasets
import transformers


def _prepare_dataset() -> None:
    arguments = _parse_arguments()
    
    raw_path: pathlib.Path = arguments.input
    model_path: pathlib.Path = arguments.model
    tokenizer_path: pathlib.Path = arguments.tokenizer
    output_path: pathlib.Path = arguments.output
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
        num_proc=16,
        desc="Tokenizing dataset"
    )

    llm = sglang.Engine(
        model_path=str(model_path),
        tp_size=arguments.tp,
        pp_size=arguments.pp,
        dp_size=arguments.dp,
    )

    sampling_params = {
        "temperature": arguments.temperature,
        "max_new_tokens": arguments.max_new_tokens
    }

    dataset = dataset.map(
        lambda example: sync_wrapper(example, llm, sampling_params),
        batched=False,
        num_proc=16,
        desc="Running sglang inference ⚙️"
    )
    
    print("Saving to disk")
    dataset.select_columns(["messages"]).to_json(output_path)


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
        "--output",
        type=pathlib.Path,
        required=True,
        help="Path to jsonlines file where the processed dataset will be stored"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature"
    )
    parser.add_argument(
        "--n",
        type=int,
        help="Number of sampels to take"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Max new tokens"
    )
    parser.add_argument(
        "--frac",
        type=float,
        help="Number of sampels to take from 0.0 to 1.0 percent"
    )
    parser.add_argument(
        "--tp",
        type=int,
        help="tp size"
    )
    parser.add_argument(
        "--pp",
        type=int,
        help="tp size"
    )
    parser.add_argument(
        "--dp",
        type=int,
        help="dp size"
    )
    return parser.parse_args()


def _tokenize_dataset(example: dict, tokenizer: transformers.AutoTokenizer) -> dict[str, torch.LongTensor]:
    if example["messages"][-1]["role"] != "assistant":
        raise ValueError("Last message must be from an assistant")

    messages = example["messages"][:-1]  # keep all except last assitant reply

    result = tokenizer.apply_chat_template(
        messages,
        tokenize=True, 
        add_generation_prompt=True, 
        return_dict=True
    )

    return {
        "input_ids": result["input_ids"],
        "messages": messages
    }


async def get_assistant_reply(input_ids, llm, sampling_params):
    return (await llm.async_generate(input_ids=input_ids, sampling_params=sampling_params))["text"]


async def async_process(example, llm, sampling_params):
    processed_text = await get_assistant_reply(example["input_ids"], llm, sampling_params)
    messages = example["messages"]
    messages.append({"role": "assistant", "content": processed_text})
    return {"messages": messages}


def sync_wrapper(example, llm, sampling_params):
    return asyncio.get_event_loop().run_until_complete(async_process(example, llm, sampling_params))


if __name__ == "__main__":
    _prepare_dataset()
