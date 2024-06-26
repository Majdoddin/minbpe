# Greedy Tokenizer
Minimal, clean code for a novel (byte-level) greedy algorithm for LLM tokenization. It beats the commonly used Byte Pair Encoding (BPE) algorithm by 2% in compression (5% for code), and runs 4 times faster.
The algorithm is "byte-level" because it runs on UTF-8 encoded strings.

Note that every improvement in compression (number of tokens in the tokenization of a text) is important, as it directly translates to  LLM's throughput.

AFAIK most research on LLM tokenization is focused on boosting the speed, so this work is somehow unique in (also) improving the compression.

### The Greedy tokenizer is almost optimal

Selecting the optimal tokenization vocabulary for a given training text, can be modelled as an Integer Linear Programming (ILP) problem, where the vocab tokens should give a minimal (in number of tokens) tokenization of the (chunks of the) training text.

To measure the the compression efficiency of the Greedy Tokenizer, I have implemented an ILP Tokenizer using the CA-SAT solver of Google's or-tools. Surprisingly, it turns out that the Greedy Tokenizer is within 1% of the optimal solution.

The ILP Tokenizer can be used as an independent tokenizer, although it takes longer to train (within 1-2% of optimal solution on 1MB training text takes ~15 min. on 48 CPU cores). If you have already a vocabulary, you can warm-start the ILP Tokenizer with your vocab, to see how optimal it is, and to optimize it further.

### What is the problem with BPE?
The purpose of tokenization is to compress the text, so the LLM can be fed with larger texts. But how good is BPE in compressing?

To take a token in the vocab, not just its frequency, but also its length is important. Because its contribution to the compression is freq * length.

BPE does subsequent merges of the most frequent pairs in the vocab. Now, suppose token "nevertheless" has a high (freq * length) in the training text, and BPE has done the merges th->X1, X1e->X2 and le->X3, but it does not merge X2X3, because "thele" is not frequent in the training text. So the following merges cannot result in token "nevertheless", just because some part of it is infrequent ☹️


### How does the Greedy Tokenizer work?
After splitting the text using a regular expression into chunks, all the chunk substrings are taken as tokens. After an initial pruning of infrequent tokens, the vocab is initialized with all single byte tokens. Then iteratively, a token is added to the vocab, if it shortens the tokenization of the training text (with the current vocab) the most. And tokens are removed from vocab, if they do not contribute enough any more.

This needs special data structures to run efficiently, and extra care should be given to avoid endless loops of additions/removals.

### Keep just the unique chunks
In long texts, the set of unique chunks is about 10% of the total number of chunks. Since identical chunks receive the same tokenization, we can boost speed by keeping just one instance of each identical chunk while noting its frequency. This does not affect the final vocabulary. This way, I have achieved a 10x speed-up in the runtime of both the BPE and Greedy Tokenizers.

### Optimal chunk tokenization
Given the fixed vocabulary after training, a chunk of text can be tokenized in many ways. BPE apply merges to the chunk, similar to training (albeit only those in the vocab), to tokenize. But this is suboptimal. I replaced it with a dynamic programming algorithm, with guaranteed minimum number of tokens for a chunk. Without affecting the runtime, it improves the compression of BPE up to 0.3%.

The same algorithm is used by the Greedy Tokenizer.

## Comparison

The table below shows the compression rate of the three tokenizers, on training text and by tokenizing a test text. We define the compression rate as

(number of tokens in the tokenization of the text) / (length of text in bytes).

We see that the Greedy Tokenizer does 1.1% better compression than BPE on English Wikipedia. It can do even better on specialized text, like code. It does 3.3% better compression on Linux source code.
Also notice how Optimal Chunk Tokenization improves the compression of BPE.

|          | Taylor Swift's article (185KB)| Wikitext-103 (1MB) | Linux source code (1MB)|
|----------|--------------|---------------------|------------------------|
| ILP      |     | tr: 47.3%, lower-bound: 46.8%  |       |
| Greedy   | tr: 44.8%   | tr: 47.2%, tst: 48.4%   | tr: 56.6%, tst: 61.9%|
| BPE      | tr: 47.1%   | tr: 48.3%, tst: 49.0%   | tr: 59.9%, tst: 63.6% |
| BPE (optimized)     | tr: 46.9% | tr:48.2%, tst: 48.7% | tr: 59.6%, tst: 63.4% |

The runtime comparison of training and encoding is shown below. We see that the Greedy Tokenizer is faster than BPE. It has slower training than boosted BPE, but is as fast in Encoding. They are run on a CPU with 4 hard cores (8 soft). ILp is run on 48 cores.

|          | Taylor Swift's article (185KB)| Wikitext-103 (1MB) | Linux source code (1MB)|
|----------|--------------|---------------------|------------------------|
| ILP      | _  | < 30m   | _ |
| Greedy   | tr: 10.32s, enc:0.10 | tr: 52.81s, tst: 0.38s  | tr: 13.37s, tst: 0.30s |
| BPE (optimized)| tr: 3.20s, tst: 0.11s | tr: 11.69s, tst: 0.35s | tr: 8:04s, tst:0.30s |
| BPE (original)| tr: 18.55s, tst: 0.31s    | tr: 81:80s, tst: 1:64s   | tr: 100.91s, tst: 1.25s |

---
This code is a fork of Karpathy's [minbpe](https://github.com/karpathy/minbpe). See that repo for more background and an excellent lecture.

Both ILP and Greedy Tokenizers can perform the 3 primary functions of a Tokenizer: 1) train the tokenizer vocabulary, 2) encode from text to tokens, 3) decode from tokens to text.
The files of the repo are as follows:

1. [minbpe/greedy.py](minbpe/greedy.py): Implements the `GreedyTokenizer` class, which is a subclass of `RegexTokenizer`.
2. [minbpe/ilp.py](minbpe/ilp.py): Implements the `ILPTokenizer` class, which is a subclass of `RegexTokenizer`.
3. [minbpe/regex.py](minbpe/regex.py): improves the `RegexTokenizer`, with Optimal Chunk Tokenization and keeping just the unique chunks.

Finally, the script [train.py](train.py) trains the three tokenizers and saves the vocab to disk for visualization. The training text can be selected to be [tests/taylorswift.txt](tests/taylorswift.txt) (this is the Wikipedia entry for her), or [tests/wikitext_103.txt](tests/wikitext_103.txt) (a collection of Wikipedia articles), or [tests/linux-kernel.txt](tests/linux-kernel.txt) (Linux kernel source files).

All of the files above are very short and thoroughly commented, and also contain a usage example on the bottom of the file.

## Quick start
As the simplest example, we can test the Greedy Tokenizer as follows:

```python
from minbpe import GreedyTokenizer
tokenizer = GreedyTokenizer()
text = "wabcdefx and yabcdefz and go"
tokenizer.train(text, 256 + 2) # 256 are the byte tokens, then select 2 tokens
tokens = tokenizer.encode_ordinary(text)
print(tokens)
# [119, 257, 120, 256, 32, 121, 257, 122, 256, 32, 103, 111]
print(tokenizer.decode(tokens))
# wabcdefx and yabcdefz and go
tokenizer.save("toy")
# writes two files: toy.model (for loading) and toy.vocab (for viewing)
```

The tricky thing to note is that the GreedyTokenizer, like BPE, always allocates the 256 individual bytes as tokens, and then adds new tokens from there (the vocab gets sorted at the end of training).


## Training
Unlike [tiktoken](https://github.com/openai/tiktoken), this code allows you to train your own tokenizer.

Following along with what OpenAI did for their text tokenizer, it's a good idea to adopt their approach of using regex pattern to split the text by categories. The GPT-4 pattern is a default with the `GreedyTokenizer`, `ILPTokenizer` and `RegexTokenizer`, so you'd simply do something like:

```python
from minbpe import GreedyTokenizer
tokenizer = GreedyTokenizer()
tokenizer.train(very_long_training_string, vocab_size=32768)
tokenizer._encode_chunk("hello world") # string -> tokens
tokenizer.decode([1000, 2000, 3000]) # tokens -> string
tokenizer.save("tok32k") # writes tok32k.model and tok32k.vocab
tokenizer.load("tok32k.model") # loads the model back from disk
```

Where, of course, you'd want to change around the vocabulary size depending on the size of your dataset.

**Special tokens**. Finally, you might wish to add special tokens to your tokenizer. Register these using the `register_special_tokens` function. For example if you train with vocab_size of 32768, then the first 256 tokens are raw byte tokens, the next 32768-256 are merge tokens, and after those you can add the special tokens. The last "real" merge token will have id of 32767 (vocab_size - 1), so your first special token should come right after that, with an id of exactly 32768. So:

```python
from minbpe import GreedyTokenizer
tokenizer = RegexTokenizer()
tokenizer.train(very_long_training_string, vocab_size=32768)
tokenizer.register_special_tokens({"<|endoftext|>": 32768})
tokenizer.encode("<|endoftext|>hello world", allowed_special="all")
```

You can of course add more tokens after that as well, as you like. Finally, I'd like to stress that I tried hard to keep the code itself clean, readable and hackable. You should not have feel scared to read the code and understand how it works. The tests are also a nice place to look for more usage examples.

## License
MIT
