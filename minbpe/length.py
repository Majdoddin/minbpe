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
        #in training vocab != None
        voc = vocab if vocab else self.vocab_rev

        n = len(text_bytes)
        dp = [float("inf") for _ in range(n)]
        backtrack = [[-1] * n for _ in range(n)]

        #At ith iteration, dp[j] is the size of optimal tokenization of text_bytes[:j+1], for 0 < j < i-1, and
        #backtrack[j] is the index of the first byte of the last token of an optimal tokenization.
        for i in range(0, n):
            if text_bytes[:i+1] in voc:
                dp[i] = 1
                backtrack[i] = 0
                continue
            for j in range(1, i+1):
                if text_bytes[j:i+1] in voc:
                    assert dp[j-1] >= 1
                    if dp[i] > dp[j-1] + 1:
                        dp[i] = dp[j-1] + 1
                        backtrack[i] = j

        # Reconstruct the tokens from the backtrack table
        def reconstruct_tokens(i):
            if backtrack[i] == 0:
                return [text_bytes[:i+1]]
            k = backtrack[i]
            return reconstruct_tokens(k-1) + [text_bytes[k:i+1]]

        if vocab: #called in training, return the size of optimal tokenization
           return dp[n-1]
        else: #return an optimal tokenization
            return [voc[token] for token in reconstruct_tokens(n - 1)]

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # input text preprocessing
        ids = [ch.encode("utf-8") for ch in text_chunks]

        #The key is a token (as bstring), the value is the number of times the token appears in the  tokenizatoin of text-chunks
        alltoks = defaultdict(int)
        alltoks.update({bytes([idx]):0  for idx in range(256)})

        #add each chunk and all its sublists to alltoks
        for i, chunkid in enumerate(ids):
            for start in range(len(chunkid)):
                for end in range(start + 1, len(chunkid) + 1):
                    alltoks[chunkid[start:end]] += 1

        #pruning any token that has same freq as one of its supertokens
        toremove = set()
        for token in alltoks:
            for start in range(len(token)):
                for end in range(start + 1, len(token) + 1):
                    if token[start:end] != token and alltoks[token[start:end]] == alltoks[token]:
                        toremove.add(token[start:end])

        #pruning tokens that have small frequency
        #The constant is emperical, b'over' occured just 13 times in tokenizaton of talor swirt wiki article
        for token in alltoks:
            if alltoks[token] <= 3 * max(1, len(text) / 180000):
                toremove.add(token)

        for token in toremove:
            if len(token) > 1:
                del alltoks[token]

        #Auxillary data structures for efficiency.
        #supchunks maps each token to the set of all the text chunks that contain the token.
        #subtkns maps each token to the set of all of its contagious subtokens
        #Because the can be identical chunks, we identify a chunk by its idx index
        supchunks = defaultdict(set)
        subtkns = defaultdict(set)
        for i, chunkid in enumerate(ids):
            for start in range(len(chunkid)):
                for end in range(start + 1, len(chunkid) + 1):
                    if chunkid[start:end] not in alltoks:
                        continue
                    supchunks[chunkid[start:end]].add(i)
                    subtkns[i].add(chunkid[start:end])

        #we do not need the values any more
        alltoks = set(alltoks.keys())
        #initialize vocab bei all single byte tokens. These will not be removed.
        vocab = set(bytes([idx])  for idx in range(256)) # int -> bytes

        mut, added = None, None
        #utility maps each token to its effect on the size of an optimal tokenization of all chunks:
        # if the token is not in vocab: How much the size would increase if the token is added to the current vocab
        # if the token is in vocab: How much the size would decrease if the token is removed from the current vocab

        #utility maps each token, chunk pair to effect of token on the size of an optimal tokenization of the chunks:
        # if token is None: the size of an optimal tokenization of the chunk. Otherwise
        #  if the token is not in vocab: How much the size would increase if the token is added to the current vocab
        #  if the token is in vocab: How much the size would decrease if the token is removed from the current vocab
        utility, utils  = defaultdict(int), defaultdict(int)
        #Greedy algorithm: iteratively, add the token with most utility to vocab, or removing the token with ...
        while len(vocab) < vocab_size:
            #find all chunks ....
            affected_chunks = supchunks[mut] if mut else set(range(len(ids)))
            #find those tokens that share a chunk with the mutation (from prev iteration). If first iteration, alltoks. exclude 1-byte tokens
            affected_tokens = (reduce(set.union, [subtkns[chunk] for chunk in affected_chunks]) if mut else alltoks) - set(bytes([idx])  for idx in range(256))#- vocab
            #update utitlity and utils according to the last mutation
            #change of utility of each token after the last mutation, is the sum of change of its utility for each chunk
            #enough to consider those chunks that contain the token of the last mutation (or all chunks if it's the first iteration), and all the tokens contained in those chunks.
            for chunkid in affected_chunks:
                utils[(None, chunkid)] = self._encode_chunk(ids[chunkid], vocab)
            for token in (affected_tokens & vocab):
                vocab.remove(token)
                for chunkid in (supchunks[token] & affected_chunks):
                    utility[token] -= utils[(token, chunkid)]
                    utils[(token, chunkid)] = self._encode_chunk(ids[chunkid], vocab) - utils[(None, chunkid)]
                    utility[token] += utils[(token, chunkid)]
                vocab.add(token)

            for token in (affected_tokens - vocab):
                #sum the improvements on minimal tokenizations, if token is added to vocab. improvements can happen only in affected_chunks
                vocab.add(token)
                for chunkid in (supchunks[token] & affected_chunks):
                    utility[token] -= utils[(token, chunkid)]
                    utils[(token, chunkid)] = (utils[(None, chunkid)] - (self._encode_chunk(ids[chunkid], vocab)))
                    utility[token] += utils[(token, chunkid)]
                vocab.remove(token)

            #utility of a token can change by mutations.
            #From the tokens in vocab that have less utility than the last added token,
            #and do not share a chunk with the lass added token (because then utility of last addition depends of the token, but it should not be affected to remain a valid removal reference point)
            #remove the one with least utility
            worst = None
            for token in vocab - set(bytes([idx])  for idx in range(256)):
                if (utility[token] < utility[added]) and not (supchunks[added] & supchunks[token]):
                    if worst and utility[token] > utility[worst]:
                        continue
                    worst = token
            if worst:
                vocab.remove(worst)
                mut = worst
                if verbose:
                    print(f"removed {mut} utility:{utility[mut]}")
            else:
                #Add the token to vocab, that has most utility.
                added = max((token for token in utility if token not in vocab), key=utility.get)
                mut = added
                vocab.add(mut)
                if verbose:
                    print(f"added {mut} utility:{utility[mut]}")

        #vocab of id:token, and vocab_rev of token:id
        self.vocab = {i:token for i, token in enumerate(vocab)}
        self.vocab_rev = {token:i for i, token in enumerate(vocab)}


