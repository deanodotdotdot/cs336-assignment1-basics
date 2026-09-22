"""Run the OpenWebText BPE training job on Modal.

Run from the repository root with:

    uv run modal run modal_train_bpe.py
"""

import gzip
import os
import pickle
import shutil
import tempfile
import time
import urllib.request
from pathlib import Path

import modal


app = modal.App("cs336-openwebtext-bpe")

image = (
    modal.Image.debian_slim()
    .pip_install("regex")
    .add_local_dir("cs336_basics", "/root/cs336_basics")
)

DATA_URL = (
    "https://huggingface.co/datasets/stanford-cs336/owt-sample/"
    "resolve/main/owt_train.txt.gz?download=true"
)


@app.function(
    image=image,
    cpu=16,
    memory=96 * 1024,
    gpu="L4",
    timeout=60 * 60,
)
def train_remote() -> bytes:
    # Imports from the repository are available in the Modal function image.
    from cs336_basics.train_bpe import train_bpe_tokenizer

    with tempfile.TemporaryDirectory() as tmp:
        archive_path = os.path.join(tmp, "owt_train.txt.gz")
        input_path = os.path.join(tmp, "owt_train.txt")

        print("Downloading OpenWebText sample...")
        urllib.request.urlretrieve(DATA_URL, archive_path)

        print("Decompressing OpenWebText sample...")
        with gzip.open(archive_path, "rb") as source, open(input_path, "wb") as target:
            shutil.copyfileobj(source, target, length=16 * 1024 * 1024)

        print("Training BPE tokenizer...")
        start = time.perf_counter()
        vocab, merges = train_bpe_tokenizer(
            input_path,
            vocab_size=32_000,
            special_tokens=["<|endoftext|>"],
        )
        elapsed = time.perf_counter() - start

        longest = max(vocab.values(), key=len)
        print(f"Training time: {elapsed / 60:.2f} minutes")
        print(f"Longest token: {longest!r} ({len(longest)} bytes)")

        # Return only the small artifacts, not the multi-GB corpus.
        return pickle.dumps(
            {
                "vocab": vocab,
                "merges": merges,
                "training_seconds": elapsed,
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )


@app.local_entrypoint()
def main():
    result = pickle.loads(train_remote.remote())

    Path("owt_vocab.pkl").write_bytes(pickle.dumps(result["vocab"]))
    Path("owt_merges.pkl").write_bytes(pickle.dumps(result["merges"]))

    print("Saved owt_vocab.pkl and owt_merges.pkl")
    print(f"Training time: {result['training_seconds'] / 60:.2f} minutes")
