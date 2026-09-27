import numpy as np
import pytest

from cs336_basics.read import END_OF_TEXT, _iter_document_chunks, encode_and_save_token_ids_multiproc
from cs336_basics.tokenizer import Tokenizer


@pytest.fixture
def tokenizer_data():
    vocab = {i: bytes([i]) for i in range(256)}
    merges = [(b"a", b"b"), (b"\n", b"\n"), (b" ", b" ")]
    for left, right in merges:
        vocab[len(vocab)] = left + right
    vocab[65535] = END_OF_TEXT.encode("utf-8")
    return vocab, merges, [END_OF_TEXT]


@pytest.mark.parametrize("chunk_bytes", [1, 7, 13, 32, 4096])
def test_chunks_preserve_bytes_and_only_split_at_document_boundaries(tmp_path, chunk_bytes):
    contents = (f"héllo 🙃\r\n\n{END_OF_TEXT}" * 7 + "trailing text\n\n").encode("utf-8")
    source = tmp_path / "corpus.txt"
    source.write_bytes(contents)

    chunks = list(_iter_document_chunks(source, chunk_bytes))

    assert b"".join(chunks) == contents
    assert all(chunk.endswith(END_OF_TEXT.encode("utf-8")) for chunk in chunks[:-1])
    assert all(chunk.decode("utf-8") for chunk in chunks)


@pytest.mark.parametrize(
    "text",
    [
        f"  ab\n\nhéllo 🙃\r\n{END_OF_TEXT}{END_OF_TEXT}\n" * 20 + "\n  ab\n\nlast document",
        "ab\n\nno document delimiter 🙃\r\n",
        "",
    ],
    ids=["unicode-and-document-boundaries", "no-delimiter", "empty"],
)
def test_multiprocess_output_matches_serial_encoding(tmp_path, tokenizer_data, text):
    vocab, merges, special_tokens = tokenizer_data
    source = tmp_path / "corpus.txt"
    source.write_bytes(text.encode("utf-8"))
    tokenizer = Tokenizer(vocab, merges, special_tokens)

    output = encode_and_save_token_ids_multiproc(
        source, tmp_path / "tokens", vocab, merges, special_tokens, num_workers=2, chunk_bytes=7
    )
    token_ids = np.load(output, mmap_mode="r", allow_pickle=False)

    assert output == tmp_path / "tokens.npy"
    assert token_ids.dtype == np.uint16
    assert token_ids.ndim == 1
    np.testing.assert_array_equal(token_ids, tokenizer.encode(text))
    assert tokenizer.decode(token_ids) == text


@pytest.mark.parametrize("invalid_id", [-1, 65536])
def test_rejects_vocabulary_ids_that_overflow_uint16(tmp_path, tokenizer_data, invalid_id):
    vocab, merges, special_tokens = tokenizer_data
    vocab[invalid_id] = b"invalid"
    with pytest.raises(ValueError, match="uint16"):
        encode_and_save_token_ids_multiproc(tmp_path / "input", tmp_path / "output", vocab, merges, special_tokens)


def test_worker_failure_keeps_previous_output_and_cleans_temporary_files(tmp_path, tokenizer_data):
    vocab, merges, special_tokens = tokenizer_data
    source = tmp_path / "corpus.txt"
    source.write_bytes(b"ab<|endoftext|>\xff")
    output = tmp_path / "tokens.npy"
    np.save(output, np.array([42], dtype=np.uint16))
    original_output = output.read_bytes()

    with pytest.raises(UnicodeDecodeError):
        encode_and_save_token_ids_multiproc(source, output, vocab, merges, special_tokens, num_workers=2, chunk_bytes=7)

    assert output.read_bytes() == original_output
    assert not list(tmp_path.glob(".token-encoding-*"))
