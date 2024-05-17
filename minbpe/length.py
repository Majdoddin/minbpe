import regex as re
from collections import defaultdict
from .base import Tokenizer, get_stats, merge
from .regex import RegexTokenizer, GPT2_SPLIT_PATTERN, GPT4_SPLIT_PATTERN

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class ContiguousSupersetTrie:
    def __init__(self):
        self.trie = TrieNode()
        self.superset_map = {}

    def insert(self, bstring):
        # Insert bstring into the trie
        node = self.trie
        for byte in bstring:
            if byte not in node.children:
                node.children[byte] = TrieNode()
            node = node.children[byte]
        node.is_end = True

        # Generate all proper contiguous substrings
        n = len(bstring)
        for i in range(n):
            for j in range(i + 1, n + 1):
                if j - i < n:  # Ensuring it's a proper substring
                    substring = bstring[i:j]
                    if substring not in self.superset_map:
                        self.superset_map[substring] = set()
                    self.superset_map[substring].add(bstring)

    def delete(self, bstring):
        # Remove bstring from the trie (tricky: might leave dangling nodes)
        def _delete(node, bstring, depth):
            if not node:
                return False
            if depth == len(bstring):
                if node.is_end:
                    node.is_end = False
                return len(node.children) == 0
            byte = bstring[depth]
            if byte in node.children and _delete(node.children[byte], bstring, depth + 1):
                del node.children[byte]
                return not node.is_end and len(node.children) == 0
            return False

        _delete(self.trie, bstring, 0)

        # Remove bstring from the superset map entries
        n = len(bstring)
        for i in range(n):
            for j in range(i + 1, n + 1):
                if j - i < n:  # Ensuring it's a proper substring
                    substring = bstring[i:j]
                    if substring in self.superset_map:
                        self.superset_map[substring].discard(bstring)
                        if not self.superset_map[substring]:
                            del self.superset_map[substring]

    def has_superset(self, bstring):
        return bstring in self.superset_map and len(self.superset_map[bstring]) > 0

class LengthTokenizer(RegexTokenizer):

    def _encode_chunk(self, text_bytes, vocab=None):
        #called in training?
        voc = vocab if vocab else self.vocab_rev

        n = len(text_bytes)
        dp = [[float('inf')] * n for _ in range(n)]
        backtrack = [[-1] * n for _ in range(n)]

        # Check for single words in the vocabulary
        for i in range(n):
            for j in range(i, n):
                if text_bytes[i:j+1] in voc:
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
                return [text_bytes[i:j+1]]
            k = backtrack[i][j]
            return reconstruct_tokens(i, k) + reconstruct_tokens(k + 1, j)

        if vocab:
            return reconstruct_tokens(0, n - 1)
        else: #called in training, return the ids
            return [voc[token] for token in reconstruct_tokens(0, n - 1)]

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # input text preprocessing
        ids = [ch.encode("utf-8") for ch in text_chunks]

        # create a vocabulary dictionary, the key is the token bytes, the value is the number of times it appears the text-chunks tokenizatoin
        #vocab = defaultdict(int)
        #first add all single bytes
        vocab = {bytes([idx]):0  for idx in range(256)} # int -> bytes

        #add each chunk and all its sublists to vocab
        for chunk in ids:
            for start in range(len(chunk)):
                for end in range(start + 1, len(chunk) + 1):
                    vocab[chunk[start:end]] = 0

        cs_trie = ContiguousSupersetTrie()
        for t in vocab:
            cs_trie.insert(t)

        while len(vocab) > vocab_size:
            for chunk in ids:
                for token in self._encode_chunk(chunk, vocab):
                    assert token in vocab
                    vocab[token] += 1

            # take the tokens longer than 1 byte, and sort them by frequency * length, and then by length
            # 1-byte tokens are put at the end
            s = sorted(list(vocab.items()), key=lambda item: (item[1]*len(item[0]) if len(item[0]) > 1 else float("inf"), -len(item[0])))
            # drop half of rare tokens (or the remaining 10)
            dropn = len(vocab) - vocab_size
            dropn = dropn if dropn <= 10 else dropn // 2
            #rebuild vocab from all 1 bytes, and the not-dropped tokens
            # vocab = defaultdict(int)
            # vocab.update({key: 0 for key, value in s[dropn:]})
            #remove dropn of tokens, from those that have no supertoken in vocan
            for i, t in enumerate(s):
                if cs_trie.has_superset(t[0]) or len(t[0]) == 1:
                    continue
                del vocab[t[0]]
                cs_trie.delete(t[0])
                dropn -= 1
                if dropn == 0:
                    break





        #vocab of id:token, and vocab_rev of token:id
        self.vocab = {i:key for i, (key, value) in enumerate(vocab.items())}
        self.vocab_rev = {key:i for i, (key, value) in enumerate(vocab.items())}
