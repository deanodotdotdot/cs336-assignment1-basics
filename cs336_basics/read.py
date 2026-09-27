"""Encode both complete training corpora with ``uv run -m cs336_basics.read``."""

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import shutil
import tempfile
import time

import numpy as np
from tqdm import tqdm

from cs336_basics.tokenizer import Tokenizer


END_OF_TEXT = "<|endoftext|>"
_worker_tokenizer = None


def load_tokenizer_data(vocab_file, merges_file):
    with open(vocab_file, "rb") as vocab_handle, open(merges_file, "rb") as merges_handle:
        return pickle.load(vocab_handle), pickle.load(merges_handle)


def load_tokenizer(vocab_file, merges_file, eos_tokens):
    vocab, merges = load_tokenizer_data(vocab_file, merges_file)
    return Tokenizer(vocab, merges, eos_tokens)


# Helper function to load N stories/paragraphs (split by blank lines)
def load_n_stories(filepath, n):
    stories = []
    with open(os.path.expanduser(filepath), encoding="utf-8") as f:
        story_lines = []
        for line in f:
            if line.strip() == "":
                if story_lines:
                    stories.append("".join(story_lines))
                    story_lines = []
                    if len(stories) == n:
                        break
            else:
                story_lines.append(line)
        # handle end-of-file story
        if len(stories) < n and story_lines:
            stories.append("".join(story_lines))
    return stories[:n]


def compute_bytes_tokens_ratio(texts, tokenizer, decode=False):
    total_bytes = 0
    total_tokens = 0
    for t in texts:
        text_bytes = t.encode("utf-8")
        tokens = tokenizer.encode(t)
        total_bytes += len(text_bytes)
        total_tokens += len(tokens)
        if decode:
            for too in tokens:
                print(tokenizer.decode([too]))
    ratio = total_bytes / total_tokens if total_tokens != 0 else float("nan")
    return total_bytes, total_tokens, ratio


def _iter_document_chunks(input_path, chunk_bytes):
    """Keep every byte, cutting only after complete end-of-document tokens.

    Memory is proportional to chunk_bytes plus the longest document. Binary
    reads preserve CRLFs and markers/UTF-8 characters spanning read boundaries.
    """
    separator = END_OF_TEXT.encode("utf-8")
    buffer = bytearray()
    with open(input_path, "rb") as source:
        while data := source.read(chunk_bytes):
            search_start = max(0, len(buffer) - len(separator) + 1)
            buffer.extend(data)
            boundary = buffer.rfind(separator, search_start)
            if boundary != -1:
                boundary += len(separator)
                yield bytes(buffer[:boundary])
                del buffer[:boundary]
        if buffer:
            yield bytes(buffer)


def _init_worker(vocab, merges, special_tokens):
    global _worker_tokenizer
    _worker_tokenizer = Tokenizer(vocab, merges, special_tokens)


def _encode_chunk(chunk):
    token_ids = _worker_tokenizer.encode(chunk.decode("utf-8"))
    return np.asarray(token_ids, dtype=np.uint16)


def encode_and_save_token_ids_multiproc(
    input_path, output_path, vocab, merges, special_tokens, num_workers=None, chunk_bytes=1024 * 1024
):
    """Write the full corpus, in order, to a one-dimensional uint16 .npy file.

    At most two chunks per worker are in flight. Token IDs are spooled to disk
    before adding the NumPy header, so neither the corpus nor all token IDs must
    fit in RAM. Finalization temporarily needs twice the output's disk space.
    """
    if num_workers is None:
        num_workers = max(mp.cpu_count() - 1, 1)
    if num_workers < 1 or chunk_bytes < 1:
        raise ValueError("num_workers and chunk_bytes must be positive")
    if any(token_id < 0 or token_id > np.iinfo(np.uint16).max for token_id in vocab):
        raise ValueError("All vocabulary token IDs must fit in uint16 (0 through 65535)")
    if not special_tokens or END_OF_TEXT not in special_tokens:
        raise ValueError(f"{END_OF_TEXT} must be registered as a special token")
    if any(END_OF_TEXT in token and token != END_OF_TEXT for token in special_tokens):
        raise ValueError("The document delimiter must not be part of a longer special token")

    input_path = Path(input_path).expanduser()
    output_path = Path(output_path).expanduser()
    if output_path.suffix != ".npy":
        output_path = Path(f"{output_path}.npy")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("The output must be different from the input dataset")
    input_bytes = input_path.stat().st_size
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Encoding {input_path} -> {output_path} with {num_workers} workers", flush=True)
    started = time.perf_counter()
    total_tokens = 0

    # Keep temporary files beside the output so the completed file can be
    # atomically renamed, preserving any previous output if a worker fails.
    with tempfile.TemporaryDirectory(prefix=".token-encoding-", dir=output_path.parent) as temp_dir:
        raw_path = Path(temp_dir) / "tokens.uint16"
        with (
            raw_path.open("wb") as raw_file,
            tqdm(total=input_bytes, unit="B", unit_scale=True, desc=input_path.name) as progress,
        ):
            with ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=mp.get_context("spawn"),
                initializer=_init_worker,
                initargs=(vocab, merges, special_tokens),
            ) as executor:
                pending = deque()
                chunks = iter(_iter_document_chunks(input_path, chunk_bytes))
                for _ in range(2 * num_workers):
                    chunk = next(chunks, None)
                    if chunk is None:
                        break
                    pending.append((executor.submit(_encode_chunk, chunk), len(chunk)))

                while pending:
                    future, byte_count = pending.popleft()
                    tokens = future.result()
                    tokens.tofile(raw_file)
                    total_tokens += tokens.size
                    progress.update(byte_count)
                    # Release the result before submitting another chunk.
                    del tokens, future
                    chunk = next(chunks, None)
                    if chunk is not None:
                        pending.append((executor.submit(_encode_chunk, chunk), len(chunk)))

        print(f"Writing NumPy header and finalizing {total_tokens:,} token IDs", flush=True)
        completed_path = Path(temp_dir) / "tokens.npy"
        with completed_path.open("wb") as output_file, raw_path.open("rb") as raw_file:
            np.lib.format.write_array_header_1_0(
                output_file,
                {"descr": np.dtype(np.uint16).str, "fortran_order": False, "shape": (total_tokens,)},
            )
            shutil.copyfileobj(raw_file, output_file, length=8 * 1024 * 1024)
        os.replace(completed_path, output_path)

    elapsed = time.perf_counter() - started
    print(f"Saved {total_tokens:,} uint16 token IDs to {output_path} in {elapsed:.1f}s", flush=True)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=max(mp.cpu_count() - 1, 1), help="Worker processes")
    parser.add_argument("--chunk-bytes", type=int, default=1024 * 1024, help="Target input bytes per task")
    parser.add_argument("--dataset", choices=("both", "tinystories", "owt"), default="both")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    parser.add_argument("--sample-stories", type=int, default=0, help="Optionally benchmark this many paragraphs first")
    args = parser.parse_args()
    if args.workers < 1 or args.chunk_bytes < 1 or args.sample_stories < 0:
        parser.error("workers/chunk-bytes must be positive and sample-stories must be nonnegative")

    project_dir = Path(__file__).resolve().parent.parent
    datasets = (
        (
            "tinystories",
            "~/data/tinystories/TinyStoriesV2-GPT4-train.txt",
            "vocab.pkl",
            "merges.pkl",
            "tinystories_token_ids.npy",
        ),
        ("owt", "~/data/owt_train.txt", "vocab2.pkl", "merges2.pkl", "owt_train_token_ids.npy"),
    )
    for name, input_path, vocab_file, merges_file, output_file in datasets:
        if args.dataset not in ("both", name):
            continue
        vocab, merges = load_tokenizer_data(project_dir / vocab_file, project_dir / merges_file)
        if args.sample_stories:
            texts = load_n_stories(input_path, args.sample_stories)
            tokenizer = Tokenizer(vocab, merges, [END_OF_TEXT])
            started = time.perf_counter()
            byte_count, token_count, ratio = compute_bytes_tokens_ratio(texts, tokenizer)
            elapsed = time.perf_counter() - started
            print(
                f"{name}: {byte_count:,} bytes, {token_count:,} tokens, {ratio:.2f} bytes/token, {byte_count / elapsed:.2f} bytes/s"
            )
        encode_and_save_token_ids_multiproc(
            input_path,
            args.output_dir / output_file,
            vocab,
            merges,
            [END_OF_TEXT],
            num_workers=args.workers,
            chunk_bytes=args.chunk_bytes,
        )


if __name__ == "__main__":
    main()
