from keras.engine.topology import InputSpec
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Input
from keras.layers.core import Layer
from keras.models import Model
from keras.optimizers import RMSprop
from keras.regularizers import l2
import numpy as np
import theano
from theano import tensor as T


def mean_absolute_error(y_true, y_pred):
    return np.absolute(y_pred - y_true).mean()


class AveragePooling(Layer):
    """
    This is a custom Keras layer. This pooling layer accepts the temporal
    sequence output by a recurrent layer and performs temporal pooling,
    looking at only the non-masked portion of the sequence. The pooling
    layer converts the entire variable-length hidden vector sequence
    into a single hidden vector, and then feeds its output to the Dense
    layer.

    input shape: (nb_samples, nb_timesteps, nb_features)
    output shape: (nb_samples, nb_features)
    """
    def __init__(self, **kwargs):
        super(AveragePooling, self).__init__(**kwargs)
        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3)]

    def call(self, x, mask=None): #mask: (nb_samples, nb_timesteps)
        if mask is None:
            mask = T.mean(T.ones_like(x), axis=-1)
        ssum = T.sum(x,axis=-2) #(nb_samples, np_features)
        rcnt = T.sum(mask,axis=-1,keepdims=True) #(nb_samples)
        rcnt = T.tile(rcnt,x.shape[-1])
        return (ssum/rcnt).astype(theano.config.floatX)
        #return rcnt

    def compute_mask(self, input, mask):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


def make_fastsent(max_features=40000, maxlen=1000, embedding_dims=9,
                  n_dense=50, dropout=0.0, lr=0.01, l2_reg=2e-8):
    x_in = Input((maxlen, ))

    embed = Embedding(max_features + 1, embedding_dims, input_length=maxlen,
                      mask_zero=True, embeddings_regularizer=l2(2e-8))(x_in)

    pool = AveragePooling()(embed)

    if n_dense:
        pool = Dropout(dropout)(pool)
        pool = Dense(n_dense, activation='tanh')(pool)

    pool = Dropout(dropout)(pool)
    out = Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_reg))(pool)
    model = Model(inputs=x_in, outputs=out)
    model.compile(loss='mse', metrics=['mae'],
                  optimizer=RMSprop(lr))
    return model
