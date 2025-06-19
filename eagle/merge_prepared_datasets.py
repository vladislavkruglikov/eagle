import argparse
import streaming


def merge_prepared_datasets() -> None:
    arguments = _parse_arguments()
    path = arguments.path
    streaming.base.util.merge_index(path, keep_local=False)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process a raw chat dataset using a verifier model and tokenizer."
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to remote dataset storafe"
    )
    return parser.parse_args()


if __name__ == "__main__":
    merge_prepared_datasets()
