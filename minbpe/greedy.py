import regex as re
from collections import defaultdict
from .regex import RegexTokenizer
from functools import reduce

class GreedyTokenizer(RegexTokenizer):
    def _encode_chunk(self, text_bytes, vocab=None):
        # returns a guaranteed optimal (least number of tokens) tokenizaton of text_bytes, given the vocab
        # uses dynamic programming

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

        if vocab: # called in training: return the length of optimal tokenization
           return dp[n-1]
        else: # called in encoding: return the tokens of an optimal tokenization
            return [voc[token] for token in reconstruct_tokens(n - 1)]

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # input text preprocessing
        chs = [ch.encode("utf-8") for ch in text_chunks]

        #keep just one instance of identical chunks, keep their count in idsw
        tmp = {}
        for byte_str in chs:
            tmp[byte_str] = tmp.get(byte_str, 0) + 1

        chs = tuple(tmp.keys())
        chcs = tuple(tmp.values())

        # the key is a token (as bstring), the value is the number of times the token appears in the  tokenizatoin of text-chunks
        alltoks = defaultdict(int)
        alltoks.update({bytes([idx]):0  for idx in range(256)})

        # add each chunk and all its sublists to alltoks. count the appearences of each token.
        for i, ch in enumerate(chs):
            for start in range(len(ch)):
                for end in range(start + 1, len(ch) + 1):
                    alltoks[ch[start:end]] += chcs[i]

        # pruning any token that has same frequency as one of its supertokens
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
        supchunks = defaultdict(set)
        subtkns = defaultdict(set)
        supchunks[b''] = set(chs)
        for ch in chs:
            for start in range(len(ch)):
                for end in range(start + 1, len(ch) + 1):
                    if tkn:=ch[start:end] not in alltoks:
                        continue
                    supchunks[tkn].add(ch)
                    subtkns[ch].add(tkn)

        # initialize vocab bei all single byte tokens. These will not be removed.
        vocab = set(bytes([idx])  for idx in range(256)) # int -> bytes

        added, removed = b'', None
        # utility maps each token to its effect on the length of an optimal tokenization of all chunks:
        #  if the token is not in vocab: How much the length would decrease if the token is added to the current vocab
        #  if the token is in vocab: How much the length would increase if the token is removed from the current vocab

        # utils maps each (token, chunk) pair to effect of the token on the length of an optimal tokenization of the chunks:
        #  if token is None: the length of an optimal tokenization of the chunk. Otherwise
        #   if the token is not in vocab: How much the length would decrease if the token is added to the current vocab
        #   if the token is in vocab: How much the length would increase if the token is removed from the current vocab
        utility, utils  = defaultdict(int), defaultdict(int)
        # Greedy algorithm: iteratively, add the token with most utility to vocab, or removing the token with ...
        while len(vocab) < vocab_size:
            # update utitlity and utils according to the last add/remove
            # change of utility of each token after the last add/remove, is the sum of change of its utility for each chunk
            # enough to consider those chunks that contain the token of the last add/remove (or all chunks if it's the first iteration), and all the tokens contained in those chunks.
            affected_tokens = (reduce(set.union, [subtkns[chunk] for chunk in supchunks[removed or added]]) if removed or added else alltoks) - set(bytes([idx])  for idx in range(256))
            # update utils of these chunks
            for ch in supchunks[removed or added]:
                utils[(None, ch)] = self._encode_chunk(ch, vocab)
            # tokens in vocab
            for token in (affected_tokens & vocab):
                vocab.remove(token)
                for ch in (supchunks[token] & supchunks[removed or added]):
                    #minus the outdated value
                    utility[token] -= utils[(token, ch)] * chcs[ch]
                    utils[(token, ch)] = self._encode_chunk(ch, vocab) - utils[(None, ch)]
                    #plus the updated value
                    utility[token] += utils[(token, ch)] * chcs[ch]
                vocab.add(token)

            # tokens not in vocab
            for token in (affected_tokens - vocab):
                # sum the improvements on minimal tokenizations, if token is added to vocab. improvements can happen only in affected_chunks
                vocab.add(token)
                for ch in (supchunks[token] & supchunks[removed or added]):
                    # minus the outdated value
                    utility[token] -= utils[(token, ch)] * chcs[ch]
                    utils[(token, ch)] = (utils[(None, ch)] - (self._encode_chunk(chs[ch], vocab)))
                    # plus the updated value
                    utility[token] += utils[(token, ch)] * chcs[ch]
                vocab.remove(token)

            # utility of a token can change by addition/removals.
            # from the tokens in vocab that now have less utility than the last added token,
            # and do not share a chunk with the last added token (because then utility of the last addition many depend on the to-be-removed token, but it should not be affected by a removal, to remain a valid reference point for removals)
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
