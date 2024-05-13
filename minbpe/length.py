import regex as re
from collections import defaultdict
from .base import Tokenizer, get_stats, merge

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

def print_vocab_overview(vocab, n=None):
    """Print each element in vocab sorted by (length of key * frequency), up to n elements"""
    sorted_vocab = sorted(vocab.items(), key=lambda x: (len(x[0]) - 1) * x[1], reverse=True)

    for i, (key, frequency) in enumerate(sorted_vocab):
        if n is not None and i >= n:
            break
        key_str = ''.join(map(chr, key))  # Convert list of integers back to string
        length = len(key)
        length_freq_product = (length - 1) * frequency
        print(f"{key_str}: {length_freq_product}, {frequency}, {length}")

class LengthTokenizer(Tokenizer):

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'': 100257}
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}


    def _generate_sublists(self, lst):
        """Generate all contiguous sublists of a list"""
        length = len(lst)
        for start in range(length):
            for end in range(start + 1, length + 1):
                yield tuple(lst[start:end])

    def prune(self, vocab):
        """Prune the vocabulary by removing proper sublists with the same frequency as their parent list"""
        to_remove = set()

        for key, value in vocab.items():
            for sublist in self._generate_sublists(key):
                if sublist != key and sublist in vocab and vocab[sublist] == value:
                    to_remove.add(sublist)

        for sublist in to_remove:
            del vocab[sublist]

    def optimal_tokenize(self, s, vocab):
        n = len(s)
        dp = [[float('inf')] * n for _ in range(n)]
        backtrack = [[-1] * n for _ in range(n)]

        # Check for single words in the vocabulary
        for i in range(n):
            for j in range(i, n):
                if s[i:j+1] in vocab:
                    dp[i][j] = 1

        # Compute the dp values
        for length in range(2, n + 1):  # length of the substring
            for i in range(n - length + 1):
                j = i + length - 1
                if dp[i][j] == 1:  # If the whole substring is a word, skip further processing
                    continue
                for k in range(i, j):
                    if dp[i][j] > dp[i][k] + dp[k+1][j]:
                        dp[i][j] = dp[i][k] + dp[k+1][j]
                        backtrack[i][j] = k

        # Reconstruct the tokens from the dp table
        def reconstruct_tokens(i, j):
            if i > j:
                return []
            if backtrack[i][j] == -1:
                return [s[i:j+1]]
            k = backtrack[i][j]
            return reconstruct_tokens(i, k) + reconstruct_tokens(k + 1, j)

        return reconstruct_tokens(0, n - 1)


    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256

        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # input text preprocessing
        ids = [tuple(ch.encode("utf-8")) for ch in text_chunks]

        # create a vocabulary dictionary
        vocab = defaultdict(int)

        # add each element of ids and all its sublists as tokens to vocab
        for chunk in ids:
            vocab[chunk] = 0
            for sublist in self._generate_sublists(chunk):
                vocab[sublist] = 0

        one_tokens = {(idx,) for idx in range(256)} # int -> bytes
        while len(vocab) > vocab_size:
            for chunk in ids:
                for token in self.optimal_tokenize(chunk, vocab):
                    vocab[token] += 1

            # Sort the vocab by frequency * length, and then by length
            # Todo: add chars to vocab, then the rest of tokens. sort just the part after chars.
            vocab = sorted(vocab.items(), key=lambda item: (item[1]*len(item[0]), -len(item[0])))
            # vocab = [(key, value) for key, value in vocab if key not in one_tokens]
            # remove half of rare tokens (or the remaining 10), that are longer than 1 char.
            dropn = len(vocab) - vocab_size
            dropn = dropn if dropn <= 10 else dropn // 2
            drop_count, i = 0, 0
            while True:
                if vocab[i][0] not in one_tokens and drop_count < dropn:
                    del vocab[i]
                    drop_count += 1
                    if drop_count == dropn:
                        break
                else:
                    i += 1

            #Set the values to 0
            vocab = {key:0 for key, value in vocab}
        self.vocab = vocab



        # print the vocab for debugging purposes
        # if verbose:
        #     # print(f"Initial vocabulary: {dict(vocab)}")
        #     print_vocab_overview(vocab, 100)

        # self.prune(vocab)
        # print_vocab_overview(vocab, 100)

        return vocab

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_ids = self.optimal_tokenize(chunk_bytes, self.vocab)
            ids.extend(chunk_ids)
        return ids



# Example usage:
# tokenizer = LengthTokenizer()
# text = "Hello, world! Hello!"
# vocab = tokenizer.train(text, vocab_size=300, verbose=True)