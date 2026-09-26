"""Run the OpenWebText BPE training job on Modal.

Run from the repository root with:

    uv run modal run modal_train_bpe.py
"""

import gzip
import json
import os
import pickle
import shutil
import time
import urllib.request
from pathlib import Path

import modal


app = modal.App("cs336-openwebtext-bpe")
data_volume = modal.Volume.from_name("cs336-owt-data", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install("regex", "tokenizers")
    .add_local_dir("cs336_basics", "/root/cs336_basics")
)

DATA_URL = (
    "https://huggingface.co/datasets/stanford-cs336/owt-sample/"
    "resolve/main/owt_train.txt.gz?download=true"
)


def _byte_encoder() -> dict[int, str]:
    """Return the GPT-2 byte-to-unicode mapping used by ByteLevel BPE."""
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    unicode_values = byte_values[:]
    extra = 0
    for byte_value in range(256):
        if byte_value not in byte_values:
            byte_values.append(byte_value)
            unicode_values.append(256 + extra)
            extra += 1
    return dict(zip(byte_values, map(chr, unicode_values)))


def _documents(input_path: str):
    delimiter = b"<|endoftext|>"
    remainder = b""
    with open(input_path, "rb") as file:
        while chunk := file.read(16 * 1024 * 1024):
            pieces = (remainder + chunk).split(delimiter)
            remainder = pieces.pop()
            for piece in pieces:
                yield piece.decode("utf-8")
    if remainder:
        yield remainder.decode("utf-8")


@app.function(
    image=image,
    cpu=16,
    memory=96 * 1024,
    gpu="L4",
    timeout=60 * 60,
    volumes={"/data": data_volume},
)
def train_remote() -> bytes:
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers, trainers

    archive_path = "/data/owt_train.txt.gz"
    input_path = "/data/owt_train.txt"

    if not os.path.exists(archive_path):
        print("Downloading OpenWebText sample...")
        urllib.request.urlretrieve(DATA_URL, archive_path)
        data_volume.commit()
    else:
        print("Reusing cached OpenWebText archive...")

    if not os.path.exists(input_path):
        print("Decompressing OpenWebText sample...")
        with gzip.open(archive_path, "rb") as source, open(input_path, "wb") as target:
            shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
        data_volume.commit()
    else:
        print("Reusing cached decompressed OpenWebText sample...")

    print("Training Rust byte-level BPE tokenizer...")
    start = time.perf_counter()
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=True,
    )
    trainer = trainers.BpeTrainer(
        vocab_size=32_000,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(
        _documents(input_path),
        trainer=trainer,
        length=2_400_000,
    )

    model = json.loads(tokenizer.to_str())["model"]
    decoder = {encoded: byte for byte, encoded in _byte_encoder().items()}

    def decode_token(token: str) -> bytes:
        if token == "<|endoftext|>":
            return token.encode("utf-8")
        return bytes(decoder[character] for character in token)

    vocabulary = {i: bytes([i]) for i in range(256)}
    vocabulary[256] = b"<|endoftext|>"
    merges = []
    next_id = 257
    for left, right in model["merges"]:
        left_bytes = decode_token(left)
        right_bytes = decode_token(right)
        merges.append((left_bytes, right_bytes))
        vocabulary[next_id] = left_bytes + right_bytes
        next_id += 1

    if len(vocabulary) != 32_000:
        raise RuntimeError(f"Expected 32000 vocabulary entries, got {len(vocabulary)}")

    elapsed = time.perf_counter() - start

    longest = max(vocabulary.values(), key=len)
    print(f"Training time: {elapsed / 60:.2f} minutes")
    print(f"Longest token: {longest!r} ({len(longest)} bytes)")

    # Return only the small artifacts, not the multi-GB corpus.
    return pickle.dumps(
        {
                "vocab": vocabulary,
            "merges": merges,
            "training_seconds": elapsed,
        },
        protocol=pickle.HIGHEST_PROTOCOL,
    )


@app.local_entrypoint()
def main():
    result = pickle.loads(train_remote.remote())

    Path("vocab2.pkl").write_bytes(pickle.dumps(result["vocab"]))
    Path("merges2.pkl").write_bytes(pickle.dumps(result["merges"]))

    print("Saved vocab2.pkl and merges2.pkl")
    print(f"Training time: {result['training_seconds'] / 60:.2f} minutes")
