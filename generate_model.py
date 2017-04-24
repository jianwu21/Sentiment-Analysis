import timeit

from keras.wrappers.scikit_learn import KerasRegressor

from sentiment.utilities import dataset_split
from sentiment.utilities import download_data
from sentiment.utilities import load_imdb
from sentiment.utilities import texts_to_sequences
from sentiment.model import mean_absolute_error
from sentiment.model import make_fastsent

N_FOLDS = 4
USE_BIGRAMS = True
MAXLEN = 1000
EPOCHS = 2
BATCH_SIZE = 32
LR = 0.01
EMBEDDING_DIMS = 10
DROPOUT = 0.5
L2_REG = 3e-8
N_DENSE = 60

MAX_FEATURES = 40000 if USE_BIGRAMS else 20000

begin_time = timeit.default_timer()

# Download the data, if they exist the function returns them anyway
data = download_data()

# Preprocess and convert words and bigrams to integer IDs
# We're using the 20000 most frequent words and 20000 most frequent bigrams
# from the IMDB dataset here: http://ai.stanford.edu/~amaas/data/sentiment/
X = texts_to_sequences(data.text, pad=MAXLEN, bigrams=USE_BIGRAMS)
y = data.ratings.values

# We're gonna need the reviewers so we can split evenly
z = data.reviewers.values

# idc_all = (idc_train, idc_val, idc_test)
idc_all = dataset_split(z, holdout=0.3, validation=0.0)

clf = make_fastsent(embedding_dims=EMBEDDING_DIMS, dropout=DROPOUT,
                    l2_reg=L2_REG, n_dense=N_DENSE, maxlen=MAXLEN,
                    max_features=MAX_FEATURES)

# Load the imdb data, hope you have them :)
data2 = load_imdb()
X_imdb = texts_to_sequences(data2.text, pad=MAXLEN, bigrams=USE_BIGRAMS)
y_imdb = data2.ratings.values

# Fit first the imdb data for an epoch with double the batch size
clf.fit(X_imdb, y_imdb, epochs=1, batch_size=BATCH_SIZE*2,
        shuffle=True, verbose=1)

# Fit our train data
clf.fit(X[idc_all[0]], y[idc_all[0]], epochs=2, batch_size=BATCH_SIZE,
        shuffle=True, verbose=1)

# Evaluate
y_pred = clf.predict(X[idc_all[1]])
mae = mean_absolute_error(y[idc_all[1]], y_pred)
print("Mean absolute error for held-out set: %.4f" % mae)

# Create and fit final model
clf = make_fastsent(embedding_dims=EMBEDDING_DIMS, dropout=DROPOUT,
                    l2_reg=L2_REG, n_dense=N_DENSE, maxlen=MAXLEN,
                    max_features=MAX_FEATURES)

clf.fit(X_imdb, y_imdb, epochs=1, batch_size=BATCH_SIZE*2,
        shuffle=True, verbose=1)
clf.fit(X, y, epochs=2, batch_size=BATCH_SIZE,
        shuffle=True, verbose=1)

clf.save('last_model.h5')
print("Saved model in 'last_model.h5'")

end_time = timeit.default_timer()
print("Script run in %.1f minutes" %((end_time-begin_time) / 60))
