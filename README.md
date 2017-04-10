# Sentiment Analysis
###### A project for Peter's class

---

## TODO

- Explore preprocessing possibilities such as stopwords, leave
punctuation in or whatever.
I think this can be done on sentiment/utilities.py:generate\_dataset with
a custom argument on the Tokenizer

- Make a nice interface to load saved models, preprocess input and get ratings

## Help on install

    pip install -r requirements.txt

OR

    pip install --user -r requirements.txt

**Important!!!** Keras can work with tensorflow OR theano. I'm using theano.
To do the same, look [here](https://keras.io/backend/)

## What's new?

### Intro

I've decided on a 1-d (not-so) deep convolutional network.

We have a number of filters with fixed weights that shift through the input,
like the convolutional filters shift through the image. Here, though, it is
in 1-dimension. Take a look at this image, I hope it makes sense:

![1D Convolution](https://www.researchgate.net/profile/Alistair_Mcewan2/publication/255564269/figure/fig1/AS:297866045214735@1448028207647/Figure-3-Convolution-of-a-radius-1-1D-filter-and-an-8-element-input-array-with-one.png)


### This model

**Input (N samples x N words) or here (1000 x 1000 x 1)**
Let's say we have 1000 samples that we pad to 1000 words each.
The words are hashed into integer values which we feed to the network

**Embedding (1000 x 400)**
We embed the integers into a 400-dimensional space

**First convolutions (1000 x N features[0]), (1000 x N features[0])**
We apply a 2-gram and a 5-gram convolution over the input

**Second convolutions (500 x N features[1]), (500 x N features[1])**
We max pool each previous level's layer and run a 3-window convolution.

**...**

After we've reached the desired depth we merge and Average pool **each level**,
concatenate them and feed them to a dense network

**Merge**
(1000 x N features), (1000 x N features) -> (1000 x 2\*N features)

**Average Pool**
(1000 x 2\*N features) -> (,2\*N features)

**Concatenate**
(,2\*N features[0] + 2\*N features[1] etc...)

**Dense**
(N allfeatures, N dense)

**Output**
The model can predit class, rating or both. I will train to see what works
best.

## A simple usage scenario!!!


```python
>>> from sentiment import (download_data, load_data, generate_dataset,
		           make_model, training_session)
>>> download_data()
>>> data = load_data()
# Get train_set, val_set, test_set and the tokenizer
>>> datasets, tkn = generate_dataset(data, mode='rating')
# We generate two models, one for regression and one for classification
# 10 - 20 for the conv layer and 100 for the fully connected
>>> model_reg, model_clf = make_model(input_shape=datasets[0][0].shape,
				      conv_size=[10, 20],
				      n_dense=100,
				      mode='single')
>>> test_err, model = training_session(model_reg, datasets, 'single')
>>> print("We achieved a %.3g mean absolute error on a held-out set" % test_err)
```
