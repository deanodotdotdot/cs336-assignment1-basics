from cs336_basics.train_bpe import PAT
from collections.abc import Iterable, Iterator
from collections import defaultdict
import pickle
import regex as re

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        # self.merges = merges
        # scored_merges is now keyed by pair (merge), value is its rank (index in merges list)

        self.vocab_rev = { bites: token_id for token_id, bites in vocab.items() }
        self.scored_merges: dict[tuple[bytes, bytes], int] = {merge: len(merges) - idx for idx, merge in enumerate(merges)}
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'rb') as file:
            vocab = pickle.load(file)
        with open(merges_filepath, 'rb') as file2:
            merges = pickle.load(file2)
        return cls(vocab, merges, special_tokens)


    def handle_merges(self, token_list: list[bytes]) -> list[int]:
        while True:
            # iterate through and find all pairs
            best_score = 0
            best_pair = None
            for i in range(0, len(token_list)-1):
                # ranking is index in merges list
                pair = (token_list[i], token_list[i+1])

                if pair in self.scored_merges:
                    score = self.scored_merges[pair]
                    if score > best_score:
                        best_score = score
                        best_pair = pair

            # print('best: ', best_pair)
            if best_pair is None:
                return [self.vocab_rev[x] for x in token_list]

            # merge the best pair
            i = 0
            while i < len(token_list) - 1:
                if (token_list[i], token_list[i+1]) == best_pair:
                    l = token_list.pop(i)
                    r = token_list.pop(i)
                    token_list.insert(i, l + r)
               
                i += 1


    def encode(self, text: str) -> list[int]:
        # Encode an input text into a sequence of token IDs.

        if self.special_tokens is not None and len(self.special_tokens) > 0:
            # Sort special_tokens longest-first to match longest tokens first
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            # Unroll the regex join so longest are matched first
            pattern = "|".join(re.escape(t) for t in sorted_special_tokens)
            # Use capturing parentheses to keep the pattern tokens in the splits
            split_regex = re.compile(f"({pattern})")
            special_splits = split_regex.split(text)
            # print(special_splits)
        else:
            special_splits = [text]

        # pretokenize
        out_list: list[int] = []
        for ss in special_splits:
            if self.special_tokens is not None and ss in self.special_tokens:
                encoded = ss.encode('utf-8')
                out_list.append(self.vocab_rev[encoded])
            else:
                text_iterator = re.finditer(PAT, ss)
                for t in text_iterator:
                    out_bytes = []
                    for char in t.group():
                        utf8_bytes = char.encode("utf-8") 
                        for b in utf8_bytes:
                            out_bytes.append(b.to_bytes())
                    out_list += self.handle_merges(out_bytes)
        return out_list

        

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        # required for memory-efficient tokenization of large files that we cannot directly load into
        # memory.
        # Dummy implementation: lazily yield token IDs (assuming self.encode exists)
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        # Decode a sequence of token IDs into text.
        out_bytes: list[bytes] = []
        for id in ids:
            out_bytes.append(self.vocab[id])
        # Join the list of bytes into a single bytes object, then decode to str
        return b"".join(out_bytes).decode("utf-8", errors='replace')
 