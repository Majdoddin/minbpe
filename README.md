# minbpe

Code for a (byte-level) Subset Selection Tokenization (SST) algorithm for LLM tokenization. It beats the commonly used Byte Pair Encoding (BPE) algorithm by 2% in compression, and 4 times faster tokenization speed.
The algorithm is "byte-level" because it runs on UTF-8 encoded strings.

Note that every impromvent in compression (number of tokens in the tokenization of a text) is important, as it directly translates to  LLM's throughput.

Basically, finding the optimal tokenization vocabulary can be seen as subset selection problem, where the selected tokens should give a minimal (in size) tokenization of the (chunks) of the training text. It seems to be a hard problem. In this implementation, we use the greedy method, where in each step a token is added to the vocab if it maximizes the gain, or is removed if it gain is less than the last added token (gain of tokens change as the vocab evolvs).

**What is the problem with BPE?**
for a token to be in vocab it is not just important appear frequently in training text, but its length is also important. So the word "Pneumonoultramicroscopicsilicovolcanoconiosis" can be a better choice than "Eke" even the former is 40 times less frequent.

Byte Pair Encoding (BPE) tries to solve the sebset selection problem by doing subsequent merges to frequent pairs of the vocab.
Now, suppose the merges li->X1 co->X2 are done but X1X2 cannot be merged, because "lico" is not frequent. So the following merges cannot result in the long word, just becasue some part of it is infrequent ☹️


This code is based on Karpathy's [minbpe](https://github.com/karpathy/minbpe). See that repo for more background and an excellent lecture.

There are two Tokenizers in this repository, both of which can perform the 3 primary functions of a Tokenizer: 1) train the tokenizer vocabulary, 2) encode from text to tokens, 3) decode from tokens to text. The files of the repo are as follows:

1. [minbpe/base.py](minbpe/base.py): Implements the `Tokenizer` class, which is the base class. It contains the `train`, `encode`, and `decode` stubs, save/load functionality, and there are also a few common utility functions. This class is not meant to be used directly, but rather to be inherited from.
<!-- 2. [minbpe/basic.py](minbpe/basic.py): Implements the `BasicTokenizer`, the simplest implementation of the BPE algorithm that runs directly on text.  -->
2. [minbpe/regex.py](minbpe/regex.py): Implements the `RegexTokenizer` that implements the BPE algorithm, but splits the input text by a regex pattern, which is a preprocessing stage that splits up the input text by categories (think: letters, numbers, punctuation) before tokenization. This ensures that no merges will happen across category boundaries. This was introduced in the GPT-2 paper and continues to be in use as of GPT-4.
<!-- 4. [minbpe/gpt4.py](minbpe/gpt4.py): Implements the `GPT4Tokenizer`. This class is a light wrapper around the `RegexTokenizer` (2, above) that exactly reproduces the tokenization of GPT-4 in the [tiktoken](https://github.com/openai/tiktoken) library. The wrapping handles some details around recovering the exact merges in the tokenizer, and the handling of some unfortunate (and likely historical?) 1-byte token permutations. -->
3. [minbpe/sst.py] Implements the `SSTTokenizer` class, that implements the SST algortihm, and also splits the input text by a regex pattern.

Finally, the script [train.py](train.py) trains the two tokenizers on the input text [tests/taylorswift.txt](tests/taylorswift.txt) (this is the Wikipedia entry for her), or on wikitext-103 (a collectoin of Wikipedia articles) and saves the vocab to disk for visualization.

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
