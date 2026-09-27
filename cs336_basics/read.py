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

    # with open('/Users/dean/dev/cs336-assignment1-basics/tests/fixtures/special_token_double_newlines_non_whitespace.txt', 'r') as file3:
    #     text = file3.read()
    #     output = tok.encode(text)
   
    #     # output = tok.encode("well hello <|endoftext|> there!")
    #     print(output)
    #     input = tok.decode(output)
    #     print(input)


    # output = tok.encode("s")
    output = tok.encode("well hello <|endoftext|> there!")
    print(output)

    for id in output:
        print(vocab[id])

    input = tok.decode(output)
    print(input)

