""" Utilities for the sentiment analysis project """
import glob
import os
from itertools import islice
from collections import Counter
import pickle
import re
import string
import traceback
import urllib.request
import zipfile

import keras.layers
from keras.preprocessing import sequence
from keras.models import load_model
import numpy as np
import pandas as pd

from .model import AveragePooling


keras.layers.AveragePooling = AveragePooling


URL = "http://2.110.57.134/LangProc2/scaledata_TRAIN.zip"
REVIEWERS = ["Dennis+Schwartz", "James+Berardinelli", "Steve+Rhodes"]
STOPWORDS = ['dennis schwartz', 'james berardinelli', 'steve rhodes',
             'movie reviews', 'all rights reserved', 'us availability',
             'running length' 'classification pg', 'reviewed on', 'ozus world',
             'review written on', 'reviewed written on',
             'reviewed by'
             'a must see film', 'excellent show look for it',
             'average movie kind of enjoyable',
             'poor show dont waste your money',
             'totally and painfully unbearable picture',
             'opinions expressed are mine and not meant to reflect my employers',
             'lowest rating', 'perfection', 'good memorable film',
             'average hits and misses', 'subpar on many levels',
             'unquestionably awful',
             'one of the top few films of this or any year one of the worst films of this or any year totally unbearable',
             'want free reviews and weekly movie and video recommendations via email just send me a letter with the word subscribe in the subject line',
             'rating', 'mpaa', 'pg',
             ]


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
        return load_data(path)

    try:
        print("Downloading zip file, please be patient...", end='', flush=True)
        filehandle, _ = urllib.request.urlretrieve(url)
        print("\rExtracting zip file...", end='', flush=True)
        zip_object = zipfile.ZipFile(filehandle, 'r')
        zip_object.extractall(path)
        print("\nDone!")
        return load_data(path)
    except Exception:
        print("An error occured. DEBUG INFO:")
        print(traceback.format_exc())


def load_data(path="./data", reviewers=REVIEWERS):
    """ Loads text data and review scores to a pandas dataframe.
    Arguments:
        path (str), default: ./data
    Returns:
        pd.DataFrame(columns=["rating", "text", "reviewers", "classes"])
    """
    data = pd.DataFrame({'ratings': [], 'text': []})
    for i, reviewer in enumerate(reviewers):
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

        # Reviews by dennis schartz guy have a first sentence with
        # the actors, this leads to overfitting
        if reviewer == "Dennis+Schwartz":
            text_data = ['\n'.join(x.split('\n')[1:]) for x in text_data]

        data = data.append(pd.DataFrame({'reviewers': reviewers,
                                         'classes': labels,
                                         'ratings': ratings,
                                         'text': text_data}),
                           ignore_index=True)
    # HACKS FOR BAD SAMPLES
    data = data.drop([449, 91, 2793])

    return data


def load_imdb():
    """ Actually assumes you have downloaded and extracted the imdb dataset
    from http://ai.stanford.edu/~amaas/data/sentiment/ """
    path = 'data/aclImdb/train/'
    data = pd.DataFrame({'ratings': [], 'text': []})
    text_data = []
    ratings = []
    for cat in ('pos', 'neg'):
        for fname in glob.glob('%s/%s/*.txt' % (path, cat)):
            with open(fname, encoding='latin-1') as fhandle:
                text_data.append(fhandle.read())
                ratings.append(1.0 if cat == 'pos' else 0.0)
    data = data.append(pd.DataFrame({'ratings': ratings, 'text': text_data}))
    data = data.reindex(np.random.permutation(data.index))
    return data


def dataset_split(vals, holdout=.2, validation=.2, seed=42):
    """ Generate a train-validation-test dataset, weighted for rating
    values rounded to 1 decimal digit.
    Arguments:
        vals (list): values to split with respect to
        holdout (float): portion of holdout set
        validation (float): portion of validation set
        seed (int): seed number for reproducability
    Returns:
        idc_train, idc_holdout, idc_validation (np.array)
        Indexes for each dataset
    Example:
        >>> idc_train, idc_holdout, idc_validation = dataset_split(y)
        >>> X_train, y_train = X[idc_train], y[idc_train]
        >>> X_test, y_test = X[idc_validation], y[idc_validation]
    """
    np.random.seed(seed)
    all_idc = np.arange(vals.shape[0])
    idc_holdout = []
    idc_validation = []
    idc_train = []
    for rating in np.unique(np.round(vals, 1)):
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


def clean_text(text, stopwords=STOPWORDS):
    """ Clean and lowercase a text:
        - Whitespace
        - Punctuation
        - HTML
        - Numbers
        - Non-ASCII characters
        - Stopwords/phrases
    Arguments:
        text (str)
    Returns:
        text (str)
    """
    regexes = ['[%s0-9]' % re.escape(string.punctuation),
               r'<.?>', r'[^\x00-\x7f]']
    for regex in regexes:
        comp = re.compile(regex)
        text = comp.sub('', text)
    text = text.lower().strip('\n')
    text = re.split(r"©|====|----|\*\*\*\*", text)[0]
    pattern = re.compile(r'\b(' + r'|'.join(STOPWORDS) + r')\b\s*')
    text = pattern.sub('', text)
    return text


def generate_dicts(texts, fname='dict.pkl', word_lim=20000, bigram_lim=20000,
                   override=False):
    """ Generate a word and bigram dictionary from a collection of texts.
    Arguments:
        text (list(str)): texts to analyze
        word_lim (int): keep N most common words
        bigram_lim (int): keep N most common bigrams
        fname (str): filename to write
        override (bool): whether to overwrite existing dictionaries
    Returns:
        None (writes to file)
    """
    checks = (os.path.isfile(fname),
              not override)
    if all(checks):
        print("Dictionaries seem to be available, to force redownloading "
              "use 'override=True'")
        return None

    clean = [clean_text(t) for t in texts]
    tokens = [[w for w in t.split() if w not in STOPWORDS] for t in clean]
    word_count = Counter([w for x in tokens for w in x])
    words = [x[0] for x in word_count.most_common(word_lim)]
    word_dict = {k: v+1 for (v, k) in enumerate(words)}

    bigram_count = Counter()
    n_all = len(clean)
    for i, text in enumerate(clean):
        print("\rGenerating bigrams, please be patient! %6d/%6d" % (i+1,
                                                                    n_all),
              end='')
        bigram_count += Counter(zip(text.split(),
                                    islice(text.split(), 1, None)))

    bigrams = [x[0] for x in bigram_count.most_common(bigram_lim)]
    bigram_dict = {k: v+word_lim+1 for (v, k) in enumerate(bigrams)}

    with open('%s' % fname, 'wb') as fhandle:
        pickle.dump([word_dict, bigram_dict], fhandle)
        print("\nModel saved to '%s'." % fname)


def texts_to_sequences(texts, fname='dict.pkl', pad=1000, bigrams=True):
    """ Convert a collection to texts to zero-padded integer arrays
    using preexisting dictionaries.
    Arguments:
        texts (list(str)): texts to convert
        fname (str): filename of dictionaries file, as generated
            by 'generate_dicts' function
        pad (int): limit all sentences to this length
    Returns:
        X (np.array) (n_samples x pad): Processed samples
    """
    if not os.path.isfile(fname):
        print("'%s' does not exist, specify another dict or use"
              "'generate_dicts' function first." % fname)
        return None

    with open(fname, 'rb') as fhandle:
        word_dict, bigram_dict = pickle.load(fhandle)
    out = []

    for text in texts:
        temp = []
        tokens = clean_text(text).split()
        for i in range(len(tokens)-1):
            temp.append(word_dict.get(tokens[i], 0))
            if bigrams:
                if (tokens[i], tokens[i+1]) in bigram_dict.keys():
                    temp.append(bigram_dict[(tokens[i], tokens[i+1])])
        temp.append(word_dict.get(tokens[-1], 0))
        out.append(temp)
    out = sequence.pad_sequences(out, pad)
    return out


def predict_from_texts(texts, model_fname='last_model.h5',
                       pad=1000, bigrams=True):
    """ Get a sentiment prediction [0.0, 1.0] from texts.
    Arguments:
        texts (array-like)
        model_fname: Filename of saved keras model
        pad (int): Pad value, as model was trained on
        bigrams (bool): Wether to use bigrams, as model was trained on
    Returns:
        y_pred (np.array(float)), shaped (len(texts),)
    """
    clf = load_model(model_fname)
    sequences = texts_to_sequences(texts, pad=pad, bigrams=bigrams)
    y_pred = clf.predict(sequences)
    return y_pred


def interactive_session(model_fname='last_model.h5', pad=1000, bigrams=True):
    """ Starts an interactive sentiment analysis session.
    Arguments:
        model_fname: Filename of saved keras model
        pad (int): Pad value, as model was trained on
        bigrams (bool): Wether to use bigrams, as model was trained on
    Returns:
        None
    Type 'q' to quit
    """
    clf = load_model(model_fname)
    clf = load_model(model_fname)
    last_input = ''
    print("Insert the sentence and then enter (just a 'q' to quit)")
    while last_input.lower() != 'q':
        last_input = input()
        if last_input != '':
            sequences = texts_to_sequences([last_input],
                                           pad=pad, bigrams=bigrams)
            rating = clf.predict(sequences)[0]
            print(" --> %.2f" % rating)
    del keras.layers
    print("Bye!")
