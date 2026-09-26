import pickle
from cs336_basics.tokenizer import Tokenizer


with open('merges.pkl', 'rb') as file, open('vocab.pkl', 'rb') as file2:
    merges: list = pickle.load(file)
    vocab = pickle.load(file2)
    # print(max(vocab))
    # print(merges[0:5])

    # longest = max(vocab.values(), key=len, default='')
    # print(longest)

    tok = Tokenizer(vocab, merges, ["<|endoftext|>"])

    output = tok.encode("s")
    # output = tok.encode("well hello <|endoftext|> there!")
    print(output)

    for id in output:
        print(vocab[id])

    input = tok.decode(output)
    print(input)

