import regex as re
from collections import defaultdict
from .regex import RegexTokenizer
from functools import reduce

class LengthTokenizer(RegexTokenizer):
    def _encode_chunk(self, text_bytes, vocab=None):
        # in training vocab != None
        voc = vocab if vocab else self.vocab_rev

        n = len(text_bytes)
        dp = [float("inf") for _ in range(n)]
        backtrack = [[-1] * n for _ in range(n)]

        # at i'th iteration, dp[j] is the size of optimal tokenization of text_bytes[:j+1], for 0 < j < i-1, and
        # backtrack[j] is the index of the first byte of the last token of an optimal tokenization.
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

        # reconstruct the tokens from the backtrack table
        def reconstruct_tokens(i):
            if backtrack[i] == 0:
                return [text_bytes[:i+1]]
            k = backtrack[i]
            return reconstruct_tokens(k-1) + [text_bytes[k:i+1]]

        if vocab: # called in training, return the size of optimal tokenization
           return dp[n-1]
        else: # return an optimal tokenization
            return [voc[token] for token in reconstruct_tokens(n - 1)]

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # input text preprocessing
        ids = [ch.encode("utf-8") for ch in text_chunks]

        # the key is a token (as bstring), the value is the number of times the token appears in the  tokenizatoin of text-chunks
        alltoks = defaultdict(int)
        alltoks.update({bytes([idx]):0  for idx in range(256)})

        # add each chunk and all its sublists to alltoks
        for i, chunk in enumerate(ids):
            for start in range(len(chunk)):
                for end in range(start + 1, len(chunk) + 1):
                    alltoks[chunk[start:end]] += 1

        # pruning any token that has same freq as one of its supertokens
        toremove = set()
        for token in alltoks:
            for start in range(len(token)):
                for end in range(start + 1, len(token) + 1):
                    if token[start:end] != token and alltoks[token[start:end]] == alltoks[token]:
                        toremove.add(token[start:end])

        # pruning tokens that have small frequency
        # The constant is emperical, b'over' occured just 13 times in tokenizaton of talor swirt wiki article
        for token in alltoks:
            if alltoks[token] <= 3 * max(1, len(text) / 180000):
                toremove.add(token)

        for token in toremove:
            if len(token) > 1:
                del alltoks[token]

        # auxillary data structures for efficiency.
        # supchunks maps each token to the set of all the text chunks that contain the token.
        # subtkns maps each token to the set of all of its contagious subtokens
        # because the can be identical chunks, we identify a chunk by its idx index
        supchunks = defaultdict(set)
        subtkns = defaultdict(set)
        supchunks[b''] = set(range(len(ids)))
        for i, chunk in enumerate(ids):
            for start in range(len(chunk)):
                for end in range(start + 1, len(chunk) + 1):
                    if chunk[start:end] not in alltoks:
                        continue
                    supchunks[chunk[start:end]].add(i)
                    subtkns[i].add(chunk[start:end])

        # we do not need the values any more
        alltoks = set(alltoks.keys())
        # initialize vocab bei all single byte tokens. These will not be removed.
        vocab = set(bytes([idx])  for idx in range(256)) # int -> bytes

        added, removed = b'', None
        # utility maps each token to its effect on the size of an optimal tokenization of all chunks:
        #  if the token is not in vocab: How much the size would increase if the token is added to the current vocab
        #  if the token is in vocab: How much the size would decrease if the token is removed from the current vocab

        # utility maps each token, chunk pair to effect of token on the size of an optimal tokenization of the chunks:
        #  if token is None: the size of an optimal tokenization of the chunk. Otherwise
        #   if the token is not in vocab: How much the size would increase if the token is added to the current vocab
        #   if the token is in vocab: How much the size would decrease if the token is removed from the current vocab
        utility, utils  = defaultdict(int), defaultdict(int)
        # Greedy algorithm: iteratively, add the token with most utility to vocab, or removing the token with ...
        while len(vocab) < vocab_size:
            # find those tokens that share a chunk with the mutation (from prev iteration). If first iteration, alltoks. exclude 1-byte tokens
            affected_tokens = (reduce(set.union, [subtkns[chunk] for chunk in supchunks[removed or added]]) if removed or added else alltoks) - set(bytes([idx])  for idx in range(256))
            # update utitlity and utils according to the last mutation
            # change of utility of each token after the last mutation, is the sum of change of its utility for each chunk
            # enough to consider those chunks that contain the token of the last mutation (or all chunks if it's the first iteration), and all the tokens contained in those chunks.
            for chunk in supchunks[removed or added]:
                utils[(None, chunk)] = self._encode_chunk(ids[chunk], vocab)
            for token in (affected_tokens & vocab):
                vocab.remove(token)
                for chunk in (supchunks[token] & supchunks[removed or added]):
                    utility[token] -= utils[(token, chunk)]
                    utils[(token, chunk)] = self._encode_chunk(ids[chunk], vocab) - utils[(None, chunk)]
                    utility[token] += utils[(token, chunk)]
                vocab.add(token)

            for token in (affected_tokens - vocab):
                # sum the improvements on minimal tokenizations, if token is added to vocab. improvements can happen only in affected_chunks
                vocab.add(token)
                for chunk in (supchunks[token] & supchunks[removed or added]):
                    utility[token] -= utils[(token, chunk)]
                    utils[(token, chunk)] = (utils[(None, chunk)] - (self._encode_chunk(ids[chunk], vocab)))
                    utility[token] += utils[(token, chunk)]
                vocab.remove(token)

            # utility of a token can change by mutations.
            # from the tokens in vocab that have less utility than the last added token,
            # and do not share a chunk with the lass added token (because then utility of last addition depends of the token, but it should not be affected to remain a valid removal reference point)
            # remove the one with least utility
            removed = None
            for token in vocab - set(bytes([idx])  for idx in range(256)):
                if (utility[token] < utility[added]) and not (supchunks[added] & supchunks[token]):
                    if removed and utility[token] > utility[removed]:
                        continue
                    removed = token
            if removed:
                vocab.remove(removed)
                if verbose:
                    print(f"removed {removed} utility:{utility[removed]}")
                continue

            # add the token to vocab, which has the most utility.
            added = max((token for token in utility if token not in vocab), key=utility.get)
            vocab.add(added)
            if verbose:
                print(f"added {added} utility:{utility[added]}")

        # vocab of id:token, and vocab_rev of token:id
        self.vocab = {i:token for i, token in enumerate(vocab)}
        self.vocab_rev = {token:i for i, token in enumerate(vocab)}
