import pathlib
import datasets
import argparse


def convert_sharegpt_dataset() -> None:
    arguments = _parse_arguments()
    output_path: pathlib.Path = arguments.output
    n = arguments.n
    frac = arguments.frac
    if n is not None and frac is not None:
        raise ValueError("One of --n or --frac must be set")

    print("Loading raw dataset")
    dataset = (
        datasets
        .load_dataset("json", data_files={
            "train": "https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V4.3_unfiltered_cleaned_split.json"
        })
        ["train"]
        .shuffle(seed=0)
    )

    print(f"Dataset has {len(dataset)} rows")
    indices = range(n) if n is not None else range(int(frac * len(dataset)))
    dataset = dataset.select(indices)
    print(f"Dataset after select has {len(dataset)} rows")

    dataset = dataset.map(
        _convert_sharegpt_dataset, 
        num_proc=1, 
        remove_columns=dataset.column_names, 
        desc="Converting dataset"
    )

    print("Saving to disk")
    dataset.to_json(output_path)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate trajectories"
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="Path to jsonlines file where the processed dataset will be stored"
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


def _convert_sharegpt_dataset(example: dict) -> dict:
    new_turns = []
    for turn in example["conversations"]:
        if turn["from"] == "gpt":
            role = "assistant"
        elif turn["from"] == "human":
            role = "user"
        else:
            raise ValueError("Unknown role")
        new_turn = {"role": role, "content": turn["value"]}
        new_turns.append(new_turn)
    return {"messages": new_turns}


if __name__ == "__main__":
    convert_sharegpt_dataset()
