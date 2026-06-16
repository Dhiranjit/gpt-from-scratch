from collections import Counter
from tqdm import tqdm

import regex as re


# Patten used by GPT4
GPT_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}{1,3}| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# ===========================================================================
# Helper
# ===========================================================================


def count_pairs(word_freqs):
    """Count adjacent-pair frequencies using pre-computed word frequencies"""
    pairs = Counter()
    for word, freq in word_freqs.items():
        # word is a tuple of intergers, e.g., (32, 116, 104, 101)
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pairs[pair] += freq # Directly add the frequency of the word
    return pairs


def merge(word_freqs, pair, new_id):
    """Replace all occurence of a `pair` with `new_id` across a word-frequency dict."""
    new_word_freqs = Counter()
    for word, freq in word_freqs.items():
        if len(word) < 2 or pair[0] not in word:
            new_word_freqs[word] += freq
            continue

        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
                new_word.append(new_id)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word_freqs[tuple(new_word)] += freq
    return new_word_freqs


# ===========================================================================
# Tokenizer
# ===========================================================================


class BPETokenizer:

    def __init__(self, special_tokens=None):
        self.pattern = re.compile(GPT_SPLIT_PATTERN)
        self.merges = {}
        self.special_tokens = special_tokens or {} # {"<|endoftext|>": vocab_size}
        self.inverse_special = {v: k for k, v in self.special_tokens.items()} 
        self.vocab = self._build_vocab()
    
    def _chunk(self, text):
        """Split text on the pattern, then UTF-8 encode each piece to byte-id list."""
        return [list(piece.encode("utf-8")) for piece in self.pattern.findall(text)]

    
    def _build_vocab(self):
        """Rebuild vocab from base bytes + merges"""
        vocab = {i: bytes([i]) for i in range(256)}
        for (p0, p1), new_id in self.merges.items():
            vocab[new_id] = vocab[p0] + vocab[p1]
        
        for tok_str, tok_id in self.special_tokens.items():
            vocab[tok_id] = tok_str.encode("utf-8")

        return vocab


    def _split_special(self, text):
        """Split text on special tokens, keeping the delimiters. Returns a list of pieces."""
        if not self.special_tokens:
            return [text]
        special_pattern = "(" + "|".join(re.escape(s) for s in self.special_tokens) + ")"
        return re.split(special_pattern, text)


    def train(self, text: str, vocab_size, verbose=False):
        assert vocab_size >= 256

        # Strip special tokens so their bytes never enter the merge statistics.
        ordinary = [p for p in self._split_special(text) if p not in self.special_tokens]

        # Get raw chunks
        raw_chunks = [c for piece in ordinary for c in self._chunk(piece)]
        original_len = sum(len(c) for c in raw_chunks)
        

        word_freqs = Counter(tuple(chunk) for chunk in raw_chunks)

        merges = {}
        iterator = tqdm(range(vocab_size - 256), desc="BPE Training", unit="merges", disable=not verbose)

        for i in iterator:
            counts = count_pairs(word_freqs)
            if not counts:
                break
            pair = max(counts, key=counts.get) # type: ignore
            new_id = 256 + i
            word_freqs = merge(word_freqs, pair, new_id)
            merges[pair] = new_id
        
        merged_len = sum(len(word) * freq for word, freq in word_freqs.items())
        
        print(f"Compression Ratio: {original_len / merged_len:.2f}X")

        self.merges = merges
        self.vocab = self._build_vocab()
    

    def encode(self, text):
        if not self.special_tokens:
            return self._encode_ordinary(text)

        ids = []
        for piece in self._split_special(text):
            if piece in self.special_tokens:
                ids.append(self.special_tokens[piece])   # atomic: one ID, no BPE
            else:
                ids.extend(self._encode_ordinary(piece)) # normal pipeline
        return ids

    def _encode_ordinary(self, text):
        """The old encode: regex-chunk + per-chunk BPE. No special tokens."""
        ids = []
        for chunk in self._chunk(text):
            ids.extend(self._encode_chunk(chunk))
        return ids


    def _encode_chunk(self, ids):
        """Greedily apply merges to a single byte-id sequence, lowest rank first."""
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            new_id = self.merges[pair]
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
        return ids

    def decode(self, ids):
        text = b"".join([self.vocab[idx] for idx in ids]).decode("utf-8", errors="replace")
        return text
    

    def show_merges(self):
        for (p0, p1), new_id in self.merges.items():
            s0  = self.vocab[p0].decode("utf-8", errors="replace")
            s1  = self.vocab[p1].decode("utf-8", errors="replace")
            out = self.vocab[new_id].decode("utf-8", errors="replace")
            print(f"{new_id}: {s0!r} + {s1!r} -> {out!r}")
    
    
    def save(self, path):
        with open(path, "w") as f:
            for pair in self.merges:
                f.write(f"{pair[0]} {pair[1]}\n")
    

    def load(self, path):
        merges = {}
        with open(path, "r") as f:
            for i, line, in enumerate(f):
                p0, p1 = map(int, line.split())
                merges[(p0, p1)] = 256 + i
        
        self.merges = merges
        self.vocab = self._build_vocab()
        return self


    

    
         
    

    

