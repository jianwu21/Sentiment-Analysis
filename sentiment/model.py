from keras import backend as K
from keras.models import Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Input
from keras.layers.core import Layer
from keras.layers.noise import GaussianDropout
from keras.optimizers import RMSprop
from keras.wrappers.scikit_learn import KerasRegressor
from keras.regularizers import l2
import numpy as np


class AveragePooling(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(AveragePooling, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if mask is not None:
            mask = K.cast(mask, 'float32')
            s = mask.sum(axis=1, keepdims=True)
            if K.equal(s, K.zeros_like(s)):
                return K.mean(x, axis=1)
            else:
                return K.cast(x.sum(axis=1) / mask.sum(axis=1, keepdims=True), K.floatx())
        else:
            return K.mean(x, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def compute_mask(self, input, input_mask=None):
        return None


def mean_absolute_error(y_true, y_pred):
    return np.absolute(y_pred - y_true).mean()


def make_fastsent(max_features=40000, maxlen=1000, embedding_dims=50,
                  n_dense=0, dropout=0.0, lr=0.01):
    x_in = Input((maxlen, ))

    embed = Embedding(max_features + 1, embedding_dims, input_length=maxlen,
                      mask_zero=True, embeddings_regularizer=l2(1e-6))(x_in)

    pool = AveragePooling()(embed)

    if n_dense:
        pool = GaussianDropout(dropout)(pool)
        pool = Dense(n_dense, activation='tanh')(pool)

    out = Dense(1, activation='sigmoid', kernel_regularizer=l2(1e-7))(pool)
    model = Model(inputs=x_in, outputs=out)
    model.compile(loss='mse', metrics=['mae'],
                  optimizer=RMSprop(lr))
    return model
