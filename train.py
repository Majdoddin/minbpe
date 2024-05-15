"""
Train our Tokenizers on some data, just to see them in action.
The whole thing runs in ~25 seconds on my laptop.
"""

import os
import time
from minbpe import BasicTokenizer, RegexTokenizer, LengthTokenizer

# open some text and train a vocab of 512 tokens
text = open("tests/taylorswift.txt", "r", encoding="utf-8").read()

# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

t0 = time.time()
for TokenizerClass, name in zip([LengthTokenizer, RegexTokenizer, BasicTokenizer], [ "length", "regex", "basic"]):

    # construct the Tokenizer object and kick off verbose training
    tokenizer = TokenizerClass()
    tokenizer.train(text, 512, verbose=True)

    tkn_ids = tokenizer.encode_ordinary(text)
    text2 = tokenizer.decode(tkn_ids)
    print(f"{name} {len(tkn_ids)} tokens")
    # writes two files in the models directory: name.model, and name.vocab
    # prefix = os.path.join("models", name)
    # tokenizer.save(prefix)
t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds")