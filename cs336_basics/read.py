import pickle

with open('merges.pkl', 'rb') as file:
    merges: list = pickle.load(file)
    print(merges[0:5])

with open('vocab.pkl', 'rb') as file2:
    vocab = pickle.load(file2)
    print(max(vocab))
    longest = ''

    for v in vocab.values():
        if len(v) >= len(longest):
            longest = v
    print(longest)

