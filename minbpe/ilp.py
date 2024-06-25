import regex as re
from collections import defaultdict
from .base import SaveLoad
from .regex import RegexTokenizer
from ortools.sat.python import cp_model
import ast

"""
Models the training of a tokenizer as an Integer Linear Programming problem,
and uses a SAT-solver to find an optimal solution to it. It is mathematically guaranteed that the tokenization of the training text with the resulting
vocabulary has the minimum number of tokens, among all the vocabs of the same size.

Also contains the load() and save() functions.
"""

class ILPTokenizer(RegexTokenizer, SaveLoad):
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

