"""Inspect basic statistics of a Bridge/RoboTwin-style LMDB dataset."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import lmdb


SUBDATABASES = ("index", "meta", "image", "depth")


def _open_env(path: Path) -> lmdb.Environment:
    return lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
    )


def _read_pickled_value(env: lmdb.Environment, key: str) -> Any:
    with env.begin() as txn:
        value = txn.get(key.encode("utf-8"))
    if value is None:
        return None
    return pickle.loads(value)


def _inspect_subdb(root: Path, name: str) -> dict[str, Any]:
    path = root / name
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
        }

    env = _open_env(path)
    try:
        stat = env.stat()
        info = env.info()
        result: dict[str, Any] = {
            "path": str(path),
            "exists": True,
            "entries": stat["entries"],
            "leaf_pages": stat["leaf_pages"],
            "branch_pages": stat["branch_pages"],
            "overflow_pages": stat["overflow_pages"],
            "map_size": info["map_size"],
            "last_pgno": info["last_pgno"],
        }
        if name == "index":
            dataset_len = _read_pickled_value(env, "__len__")
            if dataset_len is not None:
                result["dataset_len"] = dataset_len
        return result
    finally:
        env.close()


def inspect_dataset(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"LMDB dataset root does not exist: {root}")

    subdbs = {
        name: _inspect_subdb(root, name)
        for name in SUBDATABASES
    }

    dataset_len = subdbs["index"].get("dataset_len")
    return {
        "dataset_root": str(root),
        "dataset_len": dataset_len,
        "subdatabases": subdbs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_root",
        nargs="?",
        default="./bridge_lmdb_direct_smoke",
        help="Path to the LMDB dataset root containing index/meta/image/depth.",
    )
    parser.add_argument(
        "--only_dataset_len",
        action="store_true",
        help="only prints dataset length",
    )
    args = parser.parse_args()

    report = inspect_dataset(Path(args.dataset_root))
    if args.only_dataset_len:
        print(report["dataset_len"])
        return
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
