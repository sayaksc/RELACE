"""Create text-only placeholder rows; tasks come from the environment at reset."""

import argparse
from pathlib import Path


def prepare(local_dir: str, train_data_size: int, val_data_size: int):
    import pandas as pd

    if train_data_size < 1 or val_data_size < 1:
        raise ValueError("Train and validation sizes must be positive")
    output = Path(local_dir).expanduser() / "text"
    output.mkdir(parents=True, exist_ok=True)
    for split, size in [("train", train_data_size), ("test", val_data_size)]:
        rows = [
            {
                "data_source": "text",
                "prompt": [{"role": "user", "content": ""}],
                "ability": "agent",
                "extra_info": {"split": split, "index": i},
            }
            for i in range(size)
        ]
        pd.DataFrame(rows).to_parquet(output / f"{split}.parquet", index=False)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["text"], default="text")
    parser.add_argument("--local_dir", default="data/relace")
    parser.add_argument("--train_data_size", type=int, default=16)
    parser.add_argument("--val_data_size", type=int, default=128)
    args = parser.parse_args()
    print(prepare(args.local_dir, args.train_data_size, args.val_data_size))
