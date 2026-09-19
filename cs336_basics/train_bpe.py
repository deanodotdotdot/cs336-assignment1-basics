import regex as re
import collections
from cs336_basics.pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def process_chunk(filename, start_byte, size, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    pre_tokens: dict[tuple[bytes, ...], int] = {}

    with open(filename, 'rb') as f:
        f.seek(start_byte)
        # Read exactly the bytes allocated for this chunk
        chunk_data = f.read(size)
        pattern = "|".join(re.escape(t) for t in special_tokens)
        documents = re.split(pattern, chunk_data.decode("utf-8"))
        for doc in documents:
            text_iterator = re.finditer(PAT, doc)
            for t in text_iterator:
                token_list = []
                for char in t.group():
                    utf8_bytes = char.encode("utf-8") 
                    for b in utf8_bytes:
                        token_list.append(b.to_bytes())
                token_tuple = tuple(token_list)
                if token_tuple not in pre_tokens:
                    pre_tokens[token_tuple] = 1
                else:
                    pre_tokens[token_tuple] += 1
    return pre_tokens



def train_bpe_tokenizer(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    assert "" not in special_tokens
    assert len(special_tokens) > 0

    # vocabulary initialization
    # final size is  256 + number of merges + number of special tokens.
    id_counter = 0
    vocabulary: dict[int, bytes] = {}
    for _ in range(0,256):
        vocabulary[id_counter] = id_counter.to_bytes()
        id_counter += 1

    for special_token in special_tokens:
        vocabulary[id_counter] = special_token.encode("utf-8")
        id_counter += 1

    assert len(vocabulary) <= vocab_size


    # pre tokenization
    pre_tokens: dict[tuple[bytes, ...], int] = {}

    num_processes = 16
    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(file, num_processes, b"<|endoftext|>")

    # Prepare jobs for each chunk: (filename, start_byte, size, special_tokens)
    jobs = []
    for i in range(0, len(boundaries) - 1):
        start_byte = boundaries[i]
        size = boundaries[i + 1] - boundaries[i]
        jobs.append((input_path, start_byte, size, special_tokens))

    # Use multiprocessing to process chunks in parallel
    with Pool(num_processes) as p:
        results = p.starmap(process_chunk, jobs)

    # Merge partial results
    for partial_pt in results:
        for (k, v) in partial_pt.items():
            if k in pre_tokens:
                pre_tokens[k] += v
            else:
                pre_tokens[k] = v
        
    

    # compute byte level splits
    merges = []
    while len(vocabulary) < vocab_size:
        pairs = collections.defaultdict(int)
        for token, freq in pre_tokens.items():
            for i in range(len(token)-1):
                byte_tuple = (token[i],token[i+1])
                pairs[byte_tuple] += freq
        
        # find the highest frequency one (tiebreak on lexicographically greater)
        best: tuple[bytes,bytes] = max(pairs, key=lambda x: (pairs[x], x))
        best_bytes = b"".join(best)
        # add to vocab and merges
        
        vocabulary[id_counter] = best_bytes
        id_counter += 1
        merges.append(best)

        # update pre tokens tuples
        new_pt = {}
        for token, freq in pre_tokens.items():
            new_token_list = []
            # if the 2 byte seq is in there, make a new tuple token with it
            i = 0
            while i < len(token):
                if i+1 < len(token) and (token[i],token[i+1]) == best:
                    new_token_list.append(best_bytes)
                    i += 2
                else:
                    new_token_list.append(token[i])
                    i += 1
            
            new_pt[tuple(new_token_list)] = freq
        pre_tokens = new_pt
        

    assert len(vocabulary) == 256 + len(special_tokens) + len(merges)
    return (vocabulary, merges)



