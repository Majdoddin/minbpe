"""
Train our Tokenizers on some data, just to see them in action.
The whole thing runs in ~25 seconds on my laptop.
"""

import os
import time
from minbpe import RegexTokenizer, GreedyTokenizer, ILPTokenizer

# open some text and train a vocab of 512 tokens
train_text = test_text = open("../tests/taylorswift.txt", "r", encoding="utf-8").read()
# train_text = test_text = train_text[:5000]

# text = open("../tests/linux-kernel.txt", "r", encoding="utf-8").read()
# train_text = text[:2**20]
# test_text = text[2**20:2 * 2**20]

# text = open("../tests/wikitext.txt", "r", encoding="utf-8").read()
# train_text = text[:2**20]
# test_text = text[2**20:2 * 2**20]

# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

for TokenizerClass, name in zip([RegexTokenizer, GreedyTokenizer, ILPTokenizer], ["regex", "greedy", "ilp"]):

    # construct the Tokenizer object and kick off verbose training
    tokenizer = TokenizerClass()
    t0 = time.time()
    tokenizer.train(train_text, 512, verbose=True)
    t1 = time.time()

    print(f"Training took {t1 - t0:.2f} seconds")
    # writes two files in the models directory: name.model, and name.vocab
    prefix = os.path.join("models", name)
    tokenizer.save(prefix)

    t0 = time.time()
    tr_tkn_ids = tokenizer.encode_ordinary(train_text)
    t1 = time.time()
    print(f"{name}: Encoding of test text took {t1 - t0:.2f} seconds")
    print(f"{name}: Encoding of train text has {len(tr_tkn_ids)} tokens")

    t0 = time.time()
    tst_tkn_ids = tokenizer.encode_ordinary(test_text)
    t1 = time.time()
    print(f"{name}: Encoding of test text took {t1 - t0:.2f} seconds")
    print(f"{name}: Encoding of test text has {len(tst_tkn_ids)} tokens")



