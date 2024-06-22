# minbpe

Minimal, clean code for a new (byte-level) greedy algorithm for LLM tokenization. It beats the commonly used Byte Pair Encoding (BPE) algorithm by 2% for text (5% for code) compression, and has 4 times faster tokenization speed.
The algorithm is "byte-level" because it runs on UTF-8 encoded strings.

Note that every impromvent in compression (number of tokens in the tokenization of a text) is important, as it directly translates to  LLM's throughput.

Finding the optimal tokenization vocabulary for a given training text, can be modelled as an Integer Linear Programming problem, where the selected tokens should give a minimal (in number of tokens) tokenization of the (chunks) of the training text. To test the efficiency of the greedy tokenizer, I have implemented an ILP tokenizer using the CA-SAT solver of Google's or-tools. Surprisingly, it turnes out that the Greedy algorithm is within 1% of the optimal solution.

It seems to be a hard problem. In this implementation, we use the greedy method, where in each step a token is added to the vocab if it maximizes the gain, or is removed if it gain is less than the last added token (gain of tokens change as the vocab evolvs).

**How does Greedy Tokenizer work?**
After splitting the text using a regular expression into chunks, all the chunk substrings are taken as tokens. After an initial pruning of infrequent tokens, the vocab is initialized with single byte tokens. Then iteratively, a token is added to the vocab, if it shortens the tokenizaion of the training text (with the current vocab) the most. And tokens are removed from vocab, if they do not contribute enough any more.

This needs special datastructures to run efficiently, and extra care should be given to avoid endless loops of additions/removals.

**Comparision**
|          | Taylor Swift's article| Wikitext-103 (1MB) | Linux source code (1MB)|
|----------|--------------|---------------------|------------------------|
| Greedy   | Data 1   | tr: 47.2%, enc: 48.4%   | tr: 56.6%, enc: 61.9|
| ILP      | Data 3   | tr: 47.3% lower-bound: 46.8%  |    nothing    |
| BPE      | Data 5   | tr: 48.3%, enc: 49.0%   | tr: 59.9, enc: 63.6     |
compression achieved

Objective value:                506413.00000000
Lower bound:                    490806.500

491305,495770

regex 628048 tokens on train text
loaded regex 667392 tokens



|          | Taylor Swift's article (185KB)| Wikitext-103 (1MB) | Linux source code (1MB)|
|----------|--------------|---------------------|------------------------|
| Greedy   | Data 1   | tr: 55.05s, enc: 1.33s  | tr: 13.98s, enc:1.48 |
| ILP      | Data 3   | < 30m   ||
| BPE (original)      | tr:18.55s, enc: 0.31    | tr: 81:80s, enc: 1:64s   | tr:100.91s, enc:1.25 |
| BPE (unique chunks) | Data 5   | tr: 10.61,  enc:0.40  | tr: 7.45s,  enc:0.27s |
training and encoding runtimes

**What is the problem with BPE?**
To take a token in the vocab, not just its frequently, but also its length is important. Because you can get a better text compression with longer tokens.

Byte Pair Encoding (BPE) does subsequent merges of the most frequent pairs in the vocab. Now, suppose "nevertheless" is a frequent word in the training text. The merges th->X1, X1e->X2 and le->X3 are done, but X2X3 is not selected to merge, because "thele" is not frequent in the training text. So the following merges cannot result in token 'nevertheless', just becasue some part of it is infrequent ☹️








This code is a fork of Karpathy's [minbpe](https://github.com/karpathy/minbpe). See that repo for more background and an excellent lecture.

I have added two Tokenizers, both of which can perform the 3 primary functions of a Tokenizer: 1) train the tokenizer vocabulary, 2) encode from text to tokens, 3) decode from tokens to text.
Moreover, I have improved the implementation of BPE, where only distinct chunks of code are kept in a list, while keeping track of their frequency. This results in a 5x faster code.
The files of the repo are as follows:

1. [minbpe/greedy.py](minbpe/greedy.py): Implements the `GreedyTokenizer` class, which is a subclass of `RegexTokenizer`.
2. [minbpe/ilp.py](minbpe/ilp.py): Implements the `ILPTokenizer` class, which is a subclass of `RegexTokenizer`.
3. [minbpe/regex.py](minbpe/regex.py): Implements the `RegexTokenizer` that improved implementation of the BPE algorithm.


Finally, the script [train.py](train.py) trains the three tokenizers and saves the vocab to disk for visualization. The training text can be selected to be [tests/taylorswift.txt](tests/taylorswift.txt) (this is the Wikipedia entry for her), or wikitext-103 (a collectoin of Wikipedia articles), or linux kernel source.

All of the files above are very short and thoroughly commented, and also contain a usage example on the bottom of the file.

## quick start

As the simplest example, we can reproduce the [Wikipedia article on BPE](https://en.wikipedia.org/wiki/Byte_pair_encoding) as follows:

```python
tokenizer = SSTokenizer()
text = "aaa bda aabac cabdaa"
tokenizer.train(text, 256 + 2) # 256 are the byte tokens, then select 2 tokens
tokens = tokenizer.encode_ordinary(text)
print(tokens)
print(tokenizer.decode(tokens))
# aaa bda aabac cabdaa
tokenizer.save("toy")
# writes two files: toy.model (for loading) and toy.vocab (for viewing)
```

Running SST on the input string "aaa bda aabac cabdaa" for 2 tokens results in tokens "aabac" and  "aabac". The tricky thing to note is that SST, like minbpe always allocates the 256 individual bytes as tokens, and then find other tokens needed. So for us a=97, b=98, c=99, d=100 (their [ASCII](https://www.asciitable.com) values). So we start with the 256 bytes, and add 2 tokens to get to the result above.

## training

Unlike tiktoken, this code allows you to train your own tokenizer.

Following along with what OpenAI did for their text tokenizer, it's a good idea to adopt their approach of using regex pattern to split the text by categories. The GPT-4 pattern is a default with the `SSTokenizer` and `RegexTokenizer`, so you'd simply do something like:

```python
from minbpe import RegexTokenizer
tokenizer = SSTokenizer()
tokenizer.train(very_long_training_string, vocab_size=32768)
tokenizer._encode_chunk("hello world") # string -> tokens
tokenizer.decode([1000, 2000, 3000]) # tokens -> string
tokenizer.save("tok32k") # writes tok32k.model and tok32k.vocab
tokenizer.load("tok32k.model") # loads the model back from disk
```

Where, of course, you'd want to change around the vocabulary size depending on the size of your dataset.

**Special tokens**. Finally, you might wish to add special tokens to your tokenizer. Register these using the `register_special_tokens` function. For example if you train with vocab_size of 32768, then the first 256 tokens are raw byte tokens, the next 32768-256 are merge tokens, and after those you can add the special tokens. The last "real" merge token will have id of 32767 (vocab_size - 1), so your first special token should come right after that, with an id of exactly 32768. So:

```python
from minbpe import RegexTokenizer
tokenizer = RegexTokenizer()
tokenizer.train(very_long_training_string, vocab_size=32768)
tokenizer.register_special_tokens({"<|endoftext|>": 32768})
tokenizer.encode("<|endoftext|>hello world", allowed_special="all")
```

You can of course add more tokens after that as well, as you like. Finally, I'd like to stress that I tried hard to keep the code itself clean, readable and hackable. You should not have feel scared to read the code and understand how it works. The tests are also a nice place to look for more usage examples. That reminds me:

## tests

We use the pytest library for tests. All of them are located in the `tests/` directory. First `pip install pytest` if you haven't already, then:

```bash
$ pytest -v .
```

to run the tests. (-v is verbose, slightly prettier).

## todos

- write a more optimized Python version that could run over large files and big vocabs
- write an even more optimized C or Rust version (think through)
- rename GPT4Tokenizer to GPTTokenizer and support GPT-2/GPT-3/GPT-3.5 as well?
- write a LlamaTokenizer similar to GPT4Tokenizer (i.e. attempt sentencepiece equivalent)

## License
MIT
