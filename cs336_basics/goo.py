from train_bpe import train_bpe_tokenizer
import pickle

if __name__ == "__main__":
    (vocab, merges) = train_bpe_tokenizer("/Users/dean/data/tinystories/TinyStoriesV2-GPT4-train.txt", 10_000, ["<|endoftext|>"])
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("merges.pkl", "wb") as f:
        pickle.dump(merges, f)

