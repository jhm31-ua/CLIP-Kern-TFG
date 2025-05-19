import pickle
import torch
import os

from util import Parameters

class BPEKernTokenizer:
    def __init__(self, filepath = Parameters.tokenizer_path, vocab_size = Parameters.vocab_size, load = False):
        self.filepath = filepath
        if (load):
            self.load(self.filepath)
        else:
            self.vocab_size = vocab_size
            self.merges = {}

    def load(self, filepath):
        with open(filepath, 'rb') as tokenizer_file:
            tokenizer = pickle.load(tokenizer_file)

        self.vocab_size = tokenizer['vocab_size']
        self.merges = tokenizer['merges']

    def save(self, filepath):
        tokenizer = {
            'vocab_size': self.vocab_size,
            'merges': self.merges
        }

        with open(filepath, 'wb') as tokenizer_file:
            pickle.dump(tokenizer, tokenizer_file)

    def get_stats(self, tokens):
        stats = {}

        for pair in zip(tokens, tokens[1:]):
            stats[pair] = stats.get(pair, 0) + 1

        return stats

    def merge(self, tokens, pair, index):
        merged_tokens = []

        i = 0
        while i < len(tokens):
            if (i < len(tokens) - 1) and (tokens[i] == pair[0]) and (tokens[i + 1] == pair[1]):
                merged_tokens.append(index)
                i += 2
            else:
                merged_tokens.append(tokens[i])
                i += 1

        return merged_tokens

    def train(self, merges, tokens, verbose, save):
        for i in range(merges):
            stats = self.get_stats(tokens)
            if not stats:
                print(f"Iteration {i} - Stopping early: no more pairs to merge.")
                break
            pair = max(stats, key = stats.get)
            index = 256 + i

            if verbose:
                print(f"Iteration {i} - New merged token: {pair}, identified by index {index}")

            tokens = self.merge(tokens, pair, index)
            self.merges[pair] = index

        if (save):
            self.save(self.filepath)

        return self.merges

    def train_text(self, text, verbose = False, save = True):
        assert self.vocab_size >= 256, "Vocab size must be greater or equal than 256 (UTF-8 encoding)"

        merges = self.vocab_size - 256
        tokens = list(text.encode('utf-8'))

        return self.train(merges, tokens, verbose, save)

    def train_folder(self, folder, verbose = False, save = True):
        assert self.vocab_size >= 256, "Vocab size must be greater or equal than 256 (UTF-8 encoding)"

        merges = self.vocab_size - 256
        tokens = list(text.encode('utf-8'))

        for file in os.listdir(folder):
            path = os.path.join(folder, file)

            if os.path.isfile(path) and file.endswith('.krn'):
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    i_tokens = list(text.encode('utf-8'))
                    tokens.extend(i_tokens)  
        
        return self.train(merges, tokens, verbose, save)

    def get_vocabulary(self):
        vocabulary = {index: bytes([index]) for index in range(256)}

        for (p0, p1), index in self.merges.items():
            vocabulary[index] = vocabulary[p0] + vocabulary[p1]

        return vocabulary

    def encode(self, text, max_seq_length):
        tokens = list(text.encode('utf-8'))

        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key = lambda x: self.merges.get(x, float("inf")))

            if pair not in self.merges:
                break

            index = self.merges[pair]
            tokens = self.merge(tokens, pair, index)

        tokens = tokens[:max_seq_length]
        mask = [1] * len(tokens)

        padding_length = max(0, max_seq_length - len(tokens))
        tokens.extend([0] * padding_length)
        mask.extend([0] * padding_length)

        tokens = torch.tensor(tokens, dtype=torch.long)
        mask = torch.tensor(mask, dtype=torch.long)

        return tokens, mask

    def decode(self, tokens, mask = None):
        if mask is not None:
            tokens = [token for token, value in zip(tokens, mask) if value == 1]

        vocabulary = self.get_vocabulary()

        byte_tokens = b"".join(vocabulary[index.item()] for index in tokens)
        text = byte_tokens.decode('utf-8', errors = "replace")

        return text

