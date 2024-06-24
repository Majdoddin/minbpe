import regex as re
from collections import defaultdict
from .regex import RegexTokenizer
from functools import reduce
import ast

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
        for ch in chs:
            tmp[ch] = tmp.get(ch, 0) + 1
        chs = tmp
        # chs = tuple(tmp.keys())
        # chcs = tuple(tmp.values())

        # the key is a token (as bstring), the value is the number of times the token appears in the  tokenizatoin of text-chunks
        alltoks = defaultdict(int)
        onebytes = set(bytes([idx])  for idx in range(256))
        alltoks.update({tkn:0  for tkn in onebytes})

        # add each chunk and all its sublists to alltoks. count the appearences of each token.
        for ch, count in chs.items():
            for start in range(len(ch)):
                for end in range(start + 1, len(ch) + 1):
                    alltoks[ch[start:end]] += count

        # pruning any token that has same frequency as one of its supertokens
        toremove = set()
        for tkn in alltoks:
            for start in range(len(tkn)):
                for end in range(start + 1, len(tkn) + 1):
                    if (subtkn:=tkn[start:end]) != tkn and alltoks[subtkn] == alltoks[tkn]:
                        toremove.add(subtkn)

        # pruning tokens that have small frequency
        # The constant is emperical, b'over' occured just 13 times in tokenizaton of Taylor Swift's wiki article
        # TODO: take care of rare texts. for example if the text is "aaaaaaa bbbb bbbb bbbb ....", then aaaaaaa should not be removed.
        if len(text) >= 180000:
            for tkn in alltoks:
                if alltoks[tkn] <= 3 * len(text) / 180000:
                    toremove.add(tkn)

        for tkn in toremove:
            if len(tkn) > 1:
                del alltoks[tkn]

        # auxillary data structures for efficiency.
        # supchunks maps each token (of length > 1) to the set of all the text chunks that contain the token.
        # subtkns maps each chunk to the set of all of its contagious subtokens (of length > 1)
        supchunkss = defaultdict(set)
        subtkns = defaultdict(set)
        supchunkss[b''] = set(ch for ch in chs if len(ch) > 1)
        for ch in chs:
            for start in range(len(ch)):
                for end in range(start + 2, len(ch) + 1):
                    if (tkn:=ch[start:end]) not in alltoks:
                        continue
                    supchunkss[tkn].add(ch)
                    subtkns[ch].add(tkn)

        # initialize vocab bei all single byte tokens. These will not be removed.
        vocab = set(onebytes)

        added, removed = b'', None
        # `margimpact`, marginal impact, maps each token to its effect on the length of an optimal tokenization of all chunks:
        #  if the token is not in vocab: How much the length would decrease if the token is added to the current vocab
        #  if the token is in vocab: How much the length would increase if the token is removed from the current vocab

        # `michk` , marginal impact on chunk, maps each (token, chunk) pair to effect of the token on the length of an optimal tokenization of the chunk:
        #  if token is None: the length of an optimal tokenization of the chunk. Otherwise
        #   if the token is not in vocab: How much the length would decrease if the token is added to the current vocab
        #   if the token is in vocab: How much the length would increase if the token is removed from the current vocab

        # mirm, marginal impact on removal, maps each token to a set. Everytime the token is removed from vocab, its marginal impact is added to the set.
        # if its marginal impact is already in the set, then it is not removed. This is to avoid endless loops of  addition/removals
        margimpact, michk, mirm  = defaultdict(int), defaultdict(int), defaultdict(set)
        # Greedy algorithm: iteratively, add the token with the highest marginal impact to vocab, or removing the token with ...
        while len(vocab) < vocab_size:
            # update margimpact and michk according to the last vocab add/remove
            # change of marginal impact of each token after the last add/remove, is the sum of change of its marginal impact for each chunk
            # enough to consider those chunks that contain the token of the last add/remove (or all chunks if it's the first iteration), and all the tokens contained in those chunks.
            affected_tkns = (reduce(set.union, [subtkns[chunk] for chunk in supchunkss[removed or added]]) if removed or added else alltoks.keys()) - onebytes
            # update michk of the affected chunks
            for ch in supchunkss[removed or added]:
                michk[(None, ch)] = self._encode_chunk(ch, vocab)
            # tokens in vocab
            for tkn in (affected_tkns & vocab):
                vocab.remove(tkn)
                for ch in (supchunkss[tkn] & supchunkss[removed or added]):
                    #minus the outdated value
                    margimpact[tkn] -= michk[(tkn, ch)] * chs[ch]
                    michk[(tkn, ch)] = self._encode_chunk(ch, vocab) - michk[(None, ch)]
                    #plus the updated value
                    margimpact[tkn] += michk[(tkn, ch)] * chs[ch]
                vocab.add(tkn)

            # tokens not in vocab
            for tkn in (affected_tkns - vocab):
                # sum the improvements on minimal tokenizations, if token is added to vocab. improvements can happen only in affected_chnks
                vocab.add(tkn)
                for ch in (supchunkss[tkn] & supchunkss[removed or added]):
                    # minus the outdated value
                    margimpact[tkn] -= michk[(tkn, ch)] * chs[ch]
                    michk[(tkn, ch)] = (michk[(None, ch)] - (self._encode_chunk(ch, vocab)))
                    # plus the updated value
                    margimpact[tkn] += michk[(tkn, ch)] * chs[ch]
                vocab.remove(tkn)

            # marginal impact of a token can change by addition/removals.
            # from the tokens in vocab that now have less marginal impact than the last added token,
            # and do not share a chunk with the last added token (because then marginal impact of the last addition many depend on the to-be-removed token, but it should not be affected by a removal, to remain a valid reference point for removals)
            # and are not already removed having same marginal impact (to avoid endless loops of  addition/removals)
            # remove the one with the least marginal impact
            removed = None
            for tkn in vocab - onebytes:
                if (margimpact[tkn] < margimpact[added]) and (margimpact[tkn] not in mirm[tkn]) and not (supchunkss[added] & supchunkss[tkn]):
                    if removed and margimpact[tkn] > margimpact[removed]:
                        continue
                    removed = tkn
            if removed:
                mirm[removed].add(margimpact[removed])
                vocab.remove(removed)

                if verbose:
                    print(f"removed {removed}  marginal impact:{margimpact[removed]}")
                continue

            # add the token to vocab, which has the most marginal impact.
            added = max((token for token in margimpact if token not in vocab), key=margimpact.get)
            vocab.add(added)
            if verbose:
                print(f"added {added}  marginal impact:{margimpact[added]}")

        # vocab of id:token, and vocab_rev of token:id
        self.vocab = sorted(vocab, key=lambda x: (len(x), x)) #{i:token for i, token in enumerate(vocab)}
        self.vocab = {i:token for i, token in enumerate(self.vocab)}
        self.vocab_rev = {token:i for i, token in self.vocab.items()}

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write the version, pattern and tokens, that's all that's needed
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # the vocab dict
            for idx, token_bytes in self.vocab.items():
                f.write(f"[{idx}] {token_bytes}\n")
        # write the vocab: for the human to look at
        vocab_file = file_prefix + ".vocab"
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token_bytes in self.vocab.items():
                f.write(f"[{idx}] {token_bytes}\n")

    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        vocab = {}
        special_tokens = {}
        with open(model_file, 'r', encoding="ascii") as f:
            # read the version
            version = f.readline().strip()
            assert version == "minbpe v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the vocab
            while (line := f.readline().strip()):
                idx, token = line.split(maxsplit=1)
                idx = int(idx.strip('[]'))
                token = ast.literal_eval(token)
                vocab[idx] = token
        self.vocab = vocab
        self.vocab_rev = {token:i for i, token in vocab.items()}
