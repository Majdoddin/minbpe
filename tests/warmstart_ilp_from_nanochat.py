"""
Warmstart ILP tokenizer from nanochat's trained vocab and create optimized RustBPE tokenizer.

Prerequisites:
  - Run: python scripts/tok_train.py (to train and save nanochat tokenizer)

This script:
  1. Loads nanochat's saved tokenizer vocab
  2. Warmstarts ILP with that vocab and optimizes it to minimize tokenization length
  3. Creates a RustBPE tokenizer with the optimized vocab and saves it
"""
import sys
import os
import time
import argparse

# Add paths
sys.path.insert(0, '/root/minbpe')
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from minbpe import ILPTokenizer
from nanochat.tokenizer import RustBPETokenizer, SPLIT_PATTERN
from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched

# -----------------------------------------------------------------------------
# Parse command line arguments

parser = argparse.ArgumentParser(description='Warmstart ILP from nanochat vocab')
parser.add_argument('--max_chars', type=int, default=4_000_000_000, help='Maximum characters to optimize on (default: 10B)')
parser.add_argument('--doc_cap', type=int, default=10_000, help='Maximum characters per document (default: 10,000)')
args = parser.parse_args()
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")

# -----------------------------------------------------------------------------
# Load nanochat's trained tokenizer

print("\nLoading nanochat's trained tokenizer...")
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")

if not os.path.exists(tokenizer_dir):
    print(f"Error: Tokenizer not found at {tokenizer_dir}")
    print("Please run: python scripts/tok_train.py first")
    sys.exit(1)

nanochat_tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)
vocab_size = nanochat_tokenizer.get_vocab_size()
print(f"Loaded tokenizer with vocab_size={vocab_size}")

# Extract vocab
mergeable_ranks = nanochat_tokenizer.enc._mergeable_ranks  # dict[bytes, int]
print(f"Extracted {len(mergeable_ranks)} mergeable tokens")

# -----------------------------------------------------------------------------
# Collect training text

print("\nCollecting training text from FineWeb-Edu...")
training_docs = []
nchars = 0

for batch in parquets_iter_batched(split="train"):
    for doc in batch:
        doc_text = doc
        if len(doc_text) > args.doc_cap:
            doc_text = doc_text[:args.doc_cap]
        training_docs.append(doc_text)
        nchars += len(doc_text)
        if nchars > args.max_chars:
            break
    if nchars > args.max_chars:
        break

training_text = "\n".join(training_docs)
print(f"Collected {len(training_text):,} characters ({len(training_text)/1024/1024:.2f} MB)")
print(f"Number of documents: {len(training_docs):,}")

# -----------------------------------------------------------------------------
# Test nanochat tokenizer

print("\nTesting nanochat tokenizer...")
tokens_nanochat = nanochat_tokenizer.encode(training_text)
print(f"Tokenization length: {len(tokens_nanochat):,}")
print(f"Compression ratio: {len(training_text) / len(tokens_nanochat):.2f} bytes/token")

# -----------------------------------------------------------------------------
# Create ILP tokenizer with warmstart vocab

print("\nInitializing ILP tokenizer with nanochat's vocab...")

ilp_tokenizer = ILPTokenizer(pattern=SPLIT_PATTERN)

# Set vocab for warmstart
ilp_tokenizer.vocab = {i: token for token, i in mergeable_ranks.items()}
ilp_tokenizer.vocab_rev = mergeable_ranks
ilp_tokenizer.special_tokens = {}

print(f"ILP tokenizer initialized with {len(ilp_tokenizer.vocab)} tokens")

# -----------------------------------------------------------------------------
# Train ILP with warmstart

print("\nTraining ILP tokenizer with warmstart=True...")
print("OR-Tools SAT solver will optimize the vocab to minimize tokenization length")
print("This may take several minutes...\n")

t0 = time.time()
ilp_tokenizer.train(training_text, vocab_size=len(mergeable_ranks), warmstart=True, verbose=True)
t1 = time.time()

print(f"\nILP optimization time: {t1 - t0:.2f}s ({(t1-t0)/60:.2f} minutes)")

# Save ILP native format right after training
os.makedirs("out", exist_ok=True)
ilp_tokenizer.save("out/ilp_optimized_tokenizer")
print(f"Saved ILP format to: out/ilp_optimized_tokenizer.model")

# -----------------------------------------------------------------------------
# Compare results

print("\n" + "="*70)
print("RESULTS COMPARISON")
print("="*70)

tokens_ilp = ilp_tokenizer.encode_ordinary(training_text)
improvement = len(tokens_nanochat) - len(tokens_ilp)
improvement_pct = 100 * improvement / len(tokens_nanochat)

print(f"\nNanochat tokenization length: {len(tokens_nanochat):,}")
print(f"ILP tokenization length:      {len(tokens_ilp):,}")
print(f"Improvement:                  {improvement:,} tokens ({improvement_pct:.2f}%)")

# Vocab changes
nanochat_vocab_set = set(mergeable_ranks.keys())
ilp_vocab_set = set(ilp_tokenizer.vocab.values())

tokens_kept = len(nanochat_vocab_set & ilp_vocab_set)
tokens_removed = len(nanochat_vocab_set - ilp_vocab_set)
tokens_added = len(ilp_vocab_set - nanochat_vocab_set)

print(f"\nVocab changes:")
print(f"  Tokens kept:    {tokens_kept:,} ({100*tokens_kept/len(mergeable_ranks):.1f}%)")
print(f"  Tokens removed: {tokens_removed:,}")
print(f"  Tokens added:   {tokens_added:,}")

# -----------------------------------------------------------------------------
# Create RustBPE tokenizer with optimized vocab

print("\nCreating RustBPE tokenizer with ILP optimized vocab...")
import pickle
import tiktoken

# Create optimized mergeable_ranks from ILP vocab
optimized_mergeable_ranks = ilp_tokenizer.vocab_rev  # dict[bytes, int]

# Get special tokens from original nanochat tokenizer
special_tokens = nanochat_tokenizer.enc._special_tokens  # dict[str, int]

# Create new tiktoken Encoding with optimized vocab
optimized_enc = tiktoken.Encoding(
    name="ilp_optimized",
    pat_str=SPLIT_PATTERN,
    mergeable_ranks=optimized_mergeable_ranks,
    special_tokens=special_tokens,
)

# Save to tokenizer directory for RustBPE
output_dir = os.path.join(base_dir, "tokenizer_ilp_optimized")
os.makedirs(output_dir, exist_ok=True)
pickle_path = os.path.join(output_dir, "tokenizer.pkl")

with open(pickle_path, "wb") as f:
    pickle.dump(optimized_enc, f)

print(f"Saved optimized RustBPE tokenizer to: {output_dir}")
print(f"  - Total vocab size: {optimized_enc.n_vocab}")
print(f"  - Mergeable tokens: {len(optimized_mergeable_ranks)}")
print(f"  - Special tokens: {len(special_tokens)}")

print("\n✓ Optimization complete!")
print(f"\nTo use the optimized tokenizer in nanochat:")
print(f"  tokenizer = RustBPETokenizer.from_directory('{output_dir}')")
