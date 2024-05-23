import regex as re
from collections import defaultdict
from .base import Tokenizer, get_stats, merge
from .regex import RegexTokenizer, GPT2_SPLIT_PATTERN, GPT4_SPLIT_PATTERN
from functools import reduce

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

    def get_supersets(self, bstring):
        # Return the set of supersets if they exist, otherwise return an empty set
        return self.superset_map.get(bstring, set())

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
        alltoks = defaultdict(int)
        alltoks.update({bytes([idx]):0  for idx in range(256)})

        #add each chunk and all its sublists to vocab
        for i, chunkid in enumerate(ids):
            for start in range(len(chunkid)):
                for end in range(start + 1, len(chunkid) + 1):
                    alltoks[chunkid[start:end]] += 1

        #pruning any token that has same freq as one of its supertoken
        toremove = set()
        for token in alltoks:
            for start in range(len(token)):
                for end in range(start + 1, len(token) + 1):
                    if token[start:end] != token and alltoks[token[start:end]] == alltoks[token]:
                        toremove.add(token[start:end])
        #pruning tokens that have small frequency
        for token in alltoks:
            if alltoks[token] <= 25:
                toremove.add(token)

        for token in toremove:
            if len(token) > 1:
                del alltoks[token]

        #auxillary data structures for efficiency. because some chunks can be identical, we identify a chunk by its idx index
        supchunks = defaultdict(set)
        subtkns = defaultdict(set)
        #add each chunk and all its sublists to vocab
        for i, chunkid in enumerate(ids):
            for start in range(len(chunkid)):
                for end in range(start + 1, len(chunkid) + 1):
                    if chunkid[start:end] not in alltoks:
                        continue
                    supchunks[chunkid[start:end]].add(i)
                    subtkns[i].add(chunkid[start:end])
        #we do not need the frequencies anymore
        alltoks = set(alltoks.keys())
        #initialize vocab bei all single byte tokens
        vocab = set(bytes([idx])  for idx in range(256)) # int -> bytes

        change, added = None, None
        #utils holds sum of utilitis of a token, over all chuncks with that token
        utility, utils = defaultdict(int), defaultdict(int)
        #add the token with most reduction in size of tokenization
        while len(vocab) < vocab_size:
            #find those tokens that share a chunk with change (from prev iteration)
            affected_chunks = supchunks[change] if change else set(range(len(ids)))
            affected_tokens = (reduce(set.union, [subtkns[chunk] for chunk in affected_chunks]) if change else alltoks) - set(bytes([idx])  for idx in range(256))#- vocab
            #update length of minimal tokenizations of each chunk, according to the current vocab
            for chunkid in affected_chunks:
                utils[(None, chunkid)] = len(self._encode_chunk(ids[chunkid], vocab))
            for token in (affected_tokens & vocab):
                vocab.remove(token)
                for chunkid in (supchunks[token] & affected_chunks):
                    utility[token] -= utils[(token, chunkid)]
                    utils[(token, chunkid)] = (len(self._encode_chunk(ids[chunkid], vocab)) - utils[(None, chunkid)])
                    utility[token] += utils[(token, chunkid)]
                vocab.add(token)

            for token in (affected_tokens - vocab):
                #sum the improvements on minimal tokenizations, if token is added to vocab. improvements can happen only in affected_chunks
                vocab.add(token)
                for chunkid in (supchunks[token] & affected_chunks):
                    utility[token] -= utils[(token, chunkid)]
                    utils[(token, chunkid)] = (utils[(None, chunkid)] - len(self._encode_chunk(ids[chunkid], vocab)))
                    utility[token] += utils[(token, chunkid)]
                vocab.remove(token)


            #TODO: remove a token only if its utility has reduced since it was inserted.
            worst = None
            for token in vocab - set(bytes([idx])  for idx in range(256)):
                if utility[token] < utility[added] and not (supchunks[added] & supchunks[token]):
                    if worst and utility[token] > utility[worst]:
                        continue
                    worst = token
            if worst:
                vocab.remove(worst)
                change = worst
                print(f"removed {change} utility:{utility[change]}")
            else:
                added = max((token for token in utility if token not in vocab), key=utility.get)
                change = added
                vocab.add(change)
                print(f"added {change} utility:{utility[change]}")

        #vocab of id:token, and vocab_rev of token:id
        self.vocab = {i:token for i, token in enumerate(vocab)}
        self.vocab_rev = {token:i for i, token in enumerate(vocab)}


