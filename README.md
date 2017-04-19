# FastSent
###### A project for Peter's class

---

#### Credits

- [fastText](https://github.com/facebookresearch/fastText) by facebook
- the [keras tutorial](https://github.com/fchollet/keras/blob/master/examples/imdb_fasttext.py) on the subject

#### Architecture

- Assign a unique integer to the 20000 most common words and bigrams
- Take the dot product of the one-hot representation to create a N dimensions embedding
- Put a softmax on top to get the so-called rating

#### Usage

Better see demo.py

### TODO

Grid search to find optimal parameters. Available for hacking:

- epochs
- lr (learning rate)
- n\_dense (to dense or not)
- embedding\_dims
