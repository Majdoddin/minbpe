"""
Train our Tokenizers on some data, just to see them in action.
The whole thing runs in ~25 seconds on my laptop.
"""

import os
import time
from minbpe import BasicTokenizer, RegexTokenizer, LengthTokenizer

# open some text and train a vocab of 512 tokens
# text = open("tests/taylorswift.txt", "r", encoding="utf-8").read()
text = open("tests/wikitext_103_train.txt", "r", encoding="utf-8").read()

# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)


for TokenizerClass, name in zip([LengthTokenizer, RegexTokenizer, BasicTokenizer], [ "length", "regex", "basic"]):
    # construct the Tokenizer object and kick off verbose training
    tokenizer = TokenizerClass()

    t0 = time.time()
    tokenizer.train(text[:2**20], 512, verbose=True)#512
    t1 = time.time()
    print(f"Training {name} took {t1 - t0:.2f} seconds")

    t0 = time.time()
    tkn_ids = tokenizer.encode_ordinary(text[2**20:2*2**20])
    t1 = time.time()
    print(f"Encoding with {name} took {t1 - t0:.2f} seconds")

    t0 = time.time()
    text2 = tokenizer.decode(tkn_ids)
    t1 = time.time()
    print(f"Decoding with {name} took {t1 - t0:.2f} seconds")

    print(f"{name} {len(tkn_ids)} tokens")
    # writes two files in the models directory: name.model, and name.vocab
    # prefix = os.path.join("models", name)
    # tokenizer.save(prefix)


