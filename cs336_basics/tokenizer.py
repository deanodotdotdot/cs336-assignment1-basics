
from collections.abc import Iterable, Iterator
import pickle

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'rb') as file:
            vocab = pickle.load(file)
        with open(merges_filepath, 'rb') as file2:
            merges = pickle.load(file2)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        # Encode an input text into a sequence of token IDs.

        # pretokenize
        pass

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        # required for memory-efficient tokenization of large files that we cannot directly load into
        # memory.
        pass

    def decode(self, ids: list[int]) -> str:
        # Decode a sequence of token IDs into text.
        pass