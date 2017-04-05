""" Utilities for the sentiment analysis project
Includes:
    check_download_data: Checks and downloads data...
"""
import os
import urllib.request
import traceback
import re
import zipfile

from gensim.models import Word2Vec
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
              "use 'override=True'")
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
    for reviewer in REVIEWERS:
        ids = np.loadtxt(path + '/scaledata/' + reviewer + '/id.' + reviewer)
        files = ["%s/scale_whole_review/%s/txt.parag/%0.0f.txt" %
                 (path, reviewer, id)
                 for id in ids]
        ratings = np.loadtxt(path + '/scaledata/' + reviewer +
                             '/rating.' + reviewer)
        text_data = []
        for text_file in files:
            with open(text_file, encoding='latin-1') as fhandle:
                text_data.append(fhandle.read())

        data = data.append(pd.DataFrame({'ratings': ratings,
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


def text2vec(text_data, model=None):
    """ Tokenization for doc
    Arguments:
        text_data (list of sentences)
    Returns:
        vectors transformed from the doc (list)
    """
    if model is None:
        model = load_word2vec()
    return [np.array([model[w] for w in tokenize(x) if w in model.vocab])
            for x in text_data]
