""" Utilities for the sentiment analysis project
Useful ones:
    download_data: Checks and downloads data...
    generate_dataset: Generate train/test sets from data
"""
import os
import urllib.request
import traceback
import re
import zipfile

from gensim.models import Word2Vec
from keras.preprocessing import sequence, text
import numpy as np
import pandas as pd


URL = "http://2.110.57.134/LangProc2/scaledata_TRAIN.zip"
REVIEWERS = ["Dennis+Schwartz", "James+Berardinelli", "Steve+Rhodes"]


def download_data(override=False, path="./data", url=URL):
    """ Downloads and extracts a zip file from the web.
    Arguments:
        override (bool): Override the data even if folders exist.
            Defaults to True
        path (str): Path to put the data. Defaults to './data'
        url (str): URL to retrieve. Defaults to Movie Review Dataset
            on Peter's server
    Returns:
        exit code (int): 0 if success, 1 if any problem
    """
    checks = (os.path.exists(path + '/scaledata'),
              os.path.exists(path + '/scale_whole_review'),
              not override)

    if all(checks):
        print("Data seem to be available, to force redownloading "
              "Do you want to download agian? y/N")
        action = getch.getch().lower()
        if action == 'n':
            return 0
    try:
        print("Downloading zip file, please be patient...", end='', flush=True)
        filehandle, _ = urllib.request.urlretrieve(url)
        print("\rExtracting zip file...", end='', flush=True)
        zip_object = zipfile.ZipFile(filehandle, 'r')
        zip_object.extractall(path)
        print("\nDone!")
        return 0
    except Exception:
        print("An error occured. DEBUG INFO:")
        print(traceback.format_exc())
        return 1


def load_data(path="./data"):
    """ Loads text data and review scores to a pandas dataframe.
    Arguments:
        path (str), default: ./data
    Returns:
        pd.DataFrame(columns=["rating", "text"])
    """
    data = pd.DataFrame({'ratings': [], 'text': []})
    for i, reviewer in enumerate(REVIEWERS):
        ids = np.loadtxt(path + '/scaledata/' + reviewer + '/id.' + reviewer)
        files = ["%s/scale_whole_review/%s/txt.parag/%0.0f.txt" %
                 (path, reviewer, id)
                 for id in ids]
        ratings = np.loadtxt(path + '/scaledata/' + reviewer +
                             '/rating.' + reviewer)
        labels = np.loadtxt(path + '/scaledata/' + reviewer +
                             '/label.3class.' + reviewer)
        reviewers = [i for j in range(len(ratings))]
        text_data = []
        for text_file in files:
            with open(text_file, encoding='latin-1') as fhandle:
                text_data.append(fhandle.read())

        data = data.append(pd.DataFrame({'reviewers': reviewers,
                                         'classes': labels,
                                         'ratings': ratings,
                                         'text': text_data}),
                           ignore_index=True)
    return data


def tokenize(sent):
    return re.findall(r"[\w]+|[^\s\w]", sent.lower())


def generate_word2vec(sentences, fname="./data/w2vmodel", **kwargs):
    """ Generate word2vec model using gensim and save it to disk
    Arguments:
        sentences (list of strings): Sentences to train the model
        fname (str)
        ALL ADDITIONAL KWARGS WILL BE PASSED TO WORD2VEC
    Returns:
        vocabulary of the model (dict-like)
    """
    if os.path.exists(fname):
        action = input("There is a model for word2vect.\n"
                       "Do you want to generate a new one? (n or any key)\n")
        if action.lower() == 'n':
            return load_word2vec()

    stripped = [tokenize(x) for x in sentences]
    model = Word2Vec(stripped, **kwargs)
    model.save(fname)
    return model.wv


def load_word2vec(fname="./data/w2vmodel"):
    """ Load a word2vec dictionary.
    Arguments:
        fname(str)
    Returns:
        vocabulary (dict-like)
    """
    model = Word2Vec.load(fname)
    return model.wv


def text2vec(text_data, model=None, lim=0):
    """ Tokenization for doc
    Arguments:
        text_data (list of sentences)
        model (word2vec model): trained model's vocabulary
        lim (int): drop sentences with less than this number of tokens
    Returns:
        vectors transformed from the doc (list)
    """
    if model is None:
        model = load_word2vec()
    vecs = [np.array([model[w] for w in tokenize(x) if w in model.vocab])
            for x in text_data]
    return [v for v in vecs if v.shape[0] >= lim]


"""
mabe we need , I am not sure
"""
def generate_dataset(data, pad=600, holdout=.15, validation=.15, seed=42,
                     mode='both'):
    """ SORRY FOR THE MESS!!!
    Generate a train-validation-test dataset.
    Arguments:
        data: (pd.DataFrame): data as given by the load_data function
        pad (int): Specific pad with zeros /truncate value.
        houldout (float): Fration of data for holdout set
        validation (float): Fration of data for validation set
        seed (int): Random seed argument for reliable random splitting
        mode (st): y values to return, one of 'rating', 'class', 'both'
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, tokenizer
    """
    np.random.seed(seed)

    # Remove the last 5% of the review
    data.text = data.text.apply(lambda x: x[:-len(x)//20])

    tkn = text.Tokenizer(6000)
    tkn.fit_on_texts(data.text.values)
    tokens = np.array(tkn.texts_to_sequences(data.text.values))

    vals = data.ratings.values
    labels = data.classes.values
    reviewers = data.reviewers.values

    # Get length of reviews in tokens
    lens = np.array([len(x) for x in tokens])

    # Remove reviews, ratings, labels, and reviewers for reviews
    # 2 standard deviations smaller than the mean
    filt = [lens >= (lens.mean() - 2*lens.std())]
    tokens = tokens[filt]
    vals = vals[filt]
    labels = labels[filt]
    reviewers = reviewers[filt]

    if mode == 'class':
        y_all = labels
    elif mode == 'rating':
        y_all = vals
    else:
        if mode != 'both':
            print("Unrecognized option ", mode, " for 'mode' argument, "
                  "returning both")
        y_all = np.c_[vals, labels]

    all_idc = np.arange(vals.shape[0])
    idc_holdout = []
    idc_validation = []
    idc_train = []
    for rating in np.unique(vals)[1:]:
        idc_temp = all_idc[vals == rating]
        N_holdout = int(holdout * len(idc_temp))
        idc_holdout_temp = np.random.permutation(np.arange(idc_temp.size))
        idc_holdout.extend(idc_temp[idc_holdout_temp][:N_holdout])
        idc_temp = np.delete(idc_temp, idc_holdout_temp[:N_holdout])

        N_validation = int(validation * len(idc_temp))
        idc_validation_temp = np.random.permutation(np.arange(idc_temp.size))
        idc_validation.extend(idc_temp[idc_validation_temp][:N_validation])
        idc_temp = np.delete(idc_temp, idc_validation_temp[:N_validation])

        idc_train.extend(idc_temp)

    tokens = sequence.pad_sequences(tokens, pad)
    X_train, y_train = tokens[idc_train], y_all[idc_train]
    X_test, y_test = tokens[idc_holdout], y_all[idc_holdout]
    X_val, y_val = tokens[idc_validation], y_all[idc_validation]

    return ((X_train, y_train), (X_val, y_val), (X_test, y_test)), tkn


def dataset_split(vals, holdout=.2, validation=.2, seed=42):
    """ Generate a train-validation-test dataset.
    """
    np.random.seed(seed)
    all_idc = np.arange(vals.shape[0])
    idc_holdout = []
    idc_validation = []
    idc_train = []
    for rating in np.unique(vals)[1:]:
        idc_temp = all_idc[vals == rating]
        N_holdout = int(holdout * len(idc_temp))
        idc_holdout_temp = np.random.permutation(np.arange(idc_temp.size))
        idc_holdout.extend(idc_temp[idc_holdout_temp][:N_holdout])
        idc_temp = np.delete(idc_temp, idc_holdout_temp[:N_holdout])

        N_validation = int(validation * len(idc_temp))
        idc_validation_temp = np.random.permutation(np.arange(idc_temp.size))
        idc_validation.extend(idc_temp[idc_validation_temp][:N_validation])
        idc_temp = np.delete(idc_temp, idc_validation_temp[:N_validation])

        idc_train.extend(idc_temp)

    return idc_train, idc_holdout, idc_validation
