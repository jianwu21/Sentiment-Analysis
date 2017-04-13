""" Utilities for the sentiment analysis project """
import os
import urllib.request
import traceback
import re
import zipfile

from keras.preprocessing import sequence
import numpy as np
import pandas as pd


URL = "http://2.110.57.134/LangProc2/scaledata_TRAIN.zip"
REVIEWERS = ["Dennis+Schwartz", "James+Berardinelli", "Steve+Rhodes"]
FILTS = '#$%&*+-/<=>@[\\]^_`{|}~\t\n'
PREFIXES = r"(Mr|St|Mrs|Ms|Dr|Inc|Ltd|Jr|Sr|Co)[.]"
MAPS = [0, 1, 1, 1, 1, 1, 2, 3, 4, 4, 4, 5, 5, 6, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 18, 18, 18, 19, 19, 19, 19,
        19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
        19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45]


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
    data = data.drop([449, 91, 2793])
    return data


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


def clean_text(text):
    """ Clean reviews text """
    text = ' '.join([l for l in text.split('\n')if len(l) > 70])
    text = re.sub(PREFIXES, "\\1", text)
    text = re.split(r"©|=====|-----|\*\*\*\*\*", text)[0]
    text = '.'.join([s for s in text.split('.')
                     if len(s) > 10 and len(s) < 10000])
    text = text.lower().translate(str.maketrans(FILTS, ' '*len(FILTS)))
    return text


def char_trans(char):
    """ Return custom mapping for ASCII int values """
    return MAPS[char-33] + 1


def text_to_chars(text, sent_lim=200, doc_lim=30):
    """ Split a text to (30 sentences x 200 characters) """
    out = []
    sents = clean_text(text).split('.')
    for i, sent in enumerate(sents):
        if i == doc_lim:
            break
        sent = sent.encode('ascii', errors='ignore')
        out.append(sequence.pad_sequences([[char_trans(c) for c in sent]],
                                          sent_lim)[0])
    if i != doc_lim:
        out = sequence.pad_sequences([out], doc_lim)[0]
    else:
        out = np.array(out)
    return out
