import json
import sglang
import pathlib
import argparse
import datasets


def benchmark() -> None:
    # Parse arguments
    arguments = _parse_arguments()
    output_report_path = arguments.output
    prompts_path = arguments.prompts
    draft_tokens = arguments.draft
    top_k = arguments.k
    draft_steps = arguments.steps
    eagle_path = arguments.eagle
    model_path = arguments.model
    speculative_algorithm = arguments.speculative_algorithm
    n = arguments.n
    batch_size = arguments.bs
    frac = arguments.frac
    if n is not None and frac is not None:
        raise ValueError("One of --n or --frac must be set")
    
    # Load dataset
    print("Loading raw dataset")
    dataset = (
        datasets
        .load_dataset("json", data_files={"train": [str(prompts_path)]})
        ["train"]
        .shuffle(seed=0)
    )
    print(f"Dataset has {len(dataset)} rows")
    indices = range(n) if n is not None else range(int(frac * len(dataset)))
    dataset = dataset.select(indices)
    print(f"Dataset after select has {len(dataset)} rows")

    print("Creating sglang engine")
    
    # Prepare llm engine
    if speculative_algorithm is None:
        llm = sglang.Engine(
            model_path=str(model_path),
            max_running_requests=batch_size
        )
    else:
        llm = sglang.Engine(
            model_path=str(model_path),
            speculative_algorithm=speculative_algorithm,
            speculative_draft_model_path=str(eagle_path),
            speculative_num_steps=draft_steps,
            speculative_eagle_topk=top_k,
            speculative_num_draft_tokens=draft_tokens,
            max_running_requests=batch_size
        )
    
    # Send requests to llm engine
    sampling_params = {
        "temperature": 0,
    }

    total_verify_ct = 0
    total_latency = 0.0
    total_output_tokens = 0

    prompts = [example["prompt"] for example in dataset]
    outputs: list[dict] = llm.generate(prompts, sampling_params)
    llm.shutdown()

    # Collect metrics from llm engine
    for output in outputs:
        total_latency = max(total_latency, output["meta_info"]["e2e_latency"])
        total_output_tokens += output["meta_info"]["completion_tokens"]
        if speculative_algorithm is not None:
            total_verify_ct += output["meta_info"]["spec_verify_ct"]
    
    total_output_throughput = total_output_tokens / total_latency
    if speculative_algorithm is not None and total_verify_ct != 0:
        accept_length = total_output_tokens / total_verify_ct
    else:
        accept_length = None
    
    # Create report and save to disk
    report_dict = {
        "output_throughput": total_output_throughput,
        "total_output_tokens": total_output_tokens,
        "total_latency": total_latency
    }

    if speculative_algorithm is not None:
        report_dict["acceptance_length"] = accept_length

    print(report_dict)

    with output_report_path.open("w") as report_file:
        json.dump(report_dict, report_file, indent=4)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark")
    parser.add_argument(
        "--model",
        type=pathlib.Path,
        required=True,
        help="Path to jsonlines file where the processed dataset will be stored"
    )
    parser.add_argument(
        "--prompts",
        type=pathlib.Path,
        required=True,
        help="Path to jsonlines file with prompts"
    )
    parser.add_argument(
        "--eagle",
        type=pathlib.Path,
        required=False,
        help="Path to jsonlines file where the processed dataset will be stored"
    )
    parser.add_argument(
        "--speculative-algorithm",
        type=str,
        required=False,
        help="For example EAGLE or EAGLE 3"
    )
    parser.add_argument(
        "--bs",
        type=int,
        required=True,
        help="For example EAGLE or EAGLE 3"
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Number of sampels to take"
    )
    parser.add_argument(
        "--k",
        type=int,
        help="Number of top k"
    )
    parser.add_argument(
        "--draft",
        type=int,
        help="Number of draft tokens to generate"
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
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="Path to report"
    )
    return parser.parse_args()


if __name__ == "__main__":
    benchmark()
