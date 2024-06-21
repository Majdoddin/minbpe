import regex as re
from collections import defaultdict
from .regex import RegexTokenizer
from ortools.sat.python import cp_model
import ast

"""
Models the training of a tokenizer as an Integer Linear Programming problem,
and uses a SAT-solver to find an optimal solution to it. It is mathematically guaranteed that the tokenization of the training text with the resulting
vocabulary has the minimum number of tokens, among all the vocabs of the same size.

Also contains the load() and save() functions.
"""

class ILPTokenizer(RegexTokenizer):
    def _encode_chunk(self, text_bytes, vocab=None):
        # returns a mathematically guaranteed optimal (least number of tokens) tokenizaton of text_bytes, given the vocab
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

        return [voc[token] for token in reconstruct_tokens(n - 1)]

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # input text preprocessing
        ids = [ch.encode("utf-8") for ch in text_chunks]

        tmp = defaultdict(int)
        for byte_str in ids:
            tmp[byte_str] += 1
        ids = tmp

        # the key is a token (as bstring), the value is the number of times the token appears in the  tokenizatoin of text-chunks
        alltoks = defaultdict(int)
        alltoks.update({bytes([idx]):0  for idx in range(256)})

        # add each chunk and all its sublists to alltoks
        for chunk in ids:
            for start in range(len(chunk)):
                for end in range(start + 1, len(chunk) + 1):
                    alltoks[chunk[start:end]] += ids[chunk]

        # pruning any token that has same freq as one of its supertokens
        toremove = set()
        for token in alltoks:
            for start in range(len(token)):
                for end in range(start + 1, len(token) + 1):
                    if token[start:end] != token and alltoks[token[start:end]] == alltoks[token]:
                        toremove.add(token[start:end])

        # pruning tokens that have small frequency
        # The constant is emperical, b'over' occured just 13 times in tokenizaton of talor swirt wiki article
        if len(text) >= 180000:
            for tkn in alltoks:
                if alltoks[tkn] <= 3 * len(text) / 180000:
                    toremove.add(tkn)

        for token in toremove:
            if len(token) > 1:
                del alltoks[token]
        # we do not need the values any more
        alltoks = set(alltoks.keys())

        # precompute positions where each token can appear in each chunk
        # get rid of substr
        P = defaultdict(set)
        for chunk in ids:
            for start in range(len(chunk)):
                for end in range(start + 1, len(chunk) + 1):
                    if chunk[start:end] in alltoks:
                        P[(chunk, start)].add(chunk[start:end])

        model = cp_model.CpModel()

        x, y = {}, {}

        # Define variables
        for tok in alltoks:
            if len(tok) > 1:
                x[tok]= model.new_bool_var(f"{tok}")
        for (chunk, start), tokens in P.items():
            for token in tokens:
                y[chunk, token, start] = model.new_bool_var(f"{chunk}_{token}_{start}")

        # Constraint: Exactly (vocab_size - 256) additional tokens must be selected
        model.add(sum(x[token] for token in alltoks if len(token) > 1) == (vocab_size - 256))

        # Constraint: only selected tokens may be used in tokenization of chunks
        for (chunk, start), tokens in P.items():
            for token in tokens:
                if len(token) > 1:
                    model.add(y[chunk, token, start] <= x[token])

        # Constraint: Complete coverage and non-overlapping
        for chunk in ids:
            for pos in range(len(chunk)):
                model.add_exactly_one(y[chunk, token, start]
                                for start in range(pos + 1)
                                for token in P[(chunk, start)]
                                if start + len(token) > pos)

        # Objective function: Minimize the total number of pairs (token, position) used
        objective = [cp_model.LinearExpr.term(y[chunk, token, start], ids[chunk])
                        for chunk, token, start in y]
        model.minimize(cp_model.LinearExpr.sum(objective))

        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = True
        solver.parameters.symmetry_level = 3
        solver.parameters.num_search_workers = 7
        status = solver.solve(model)

        if status == cp_model.OPTIMAL:
            self.vocab = sorted([token for token in alltoks if len(token) == 1 or solver.value(x[token]) == 1], key=lambda x: (len(x), x))
            self.vocab = {i:token for i, token in enumerate(self.vocab)}
            self.vocab_rev = {token:i for i, token in self.vocab.items()}
            print(f"Length of tokenization: {sum(solver.value(y[chunk, token, start]) * ids[chunk] for chunk, token, start in y)}")

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

