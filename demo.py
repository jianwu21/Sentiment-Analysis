import pickle

from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error

from sentiment.utilities import download_data, texts_to_sequences
from sentiment.model import make_fastsent

N_FOLDS = 10

# Download the data, if they exist the function returns them anyway
data = download_data()

# Preprocess and convert words and bigrams to integer IDs
# We're using the 20000 most frequent words and 20000 most frequent bigrams
# from the IMDB dataset here: http://ai.stanford.edu/~amaas/data/sentiment/
X = texts_to_sequences(data.text)
y = data.ratings.values

# We're gonna need the reviewers so we can split evenly
z = data.reviewers.values

clf = KerasRegressor(make_fastsent, epochs=4, batch_size=32, shuffle=True,
                     verbose=0)

scores = []
cv = StratifiedKFold(N_FOLDS)

for i, (train_idx, test_idx) in enumerate(cv.split(X, z)):
    print("\rDoin' fold n.%2d/%-2d" % (i+1, N_FOLDS), end='')
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    _clf = clone(clf)
    _clf.fit(X_train, y_train)
    y_pred = _clf.predict(X_test)
    scores.append(mean_absolute_error(y_test, y_pred))

print("\nRESULTS FOR %d FOLDS:" % N_FOLDS)
print("avg mae: %.4f (%.4f), min: %.4f, max: %.4f" % (np.mean(scores),
                                                      np.std(scores),
                                                      np.min(scores),
                                                      np.max(scores)))
