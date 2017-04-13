from keras.models import Model
from keras.layers import Dense, Dropout, GRU, Embedding, Input
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop
from keras.regularizers import l2
import numpy as np


def mean_absolute_error(y_true, y_pred):
    return np.absolute(predictions - truth).mean()


def create_kwargs(layer, kwargs):
    if layer in ('encoder', 'analyzer'):
        kwargs_out = {'return_sequences': True}
        kwargs_out.update({k: kwargs[layer + '_' + k]
                           for k in ('units', 'activation', 'dropout',
                                    'recurrent_dropout')})
        kwargs_out.update(
            {k + '_' + 'regularizer': l2(kwargs[layer + '_l2_' + k])
             for k in ('activity', 'kernel', 'bias')})
    else:
        kwargs_out = {'units': kwargs['dense_units']}
        kwargs_out.update(
            {k + '_' + 'regularizer': l2(kwargs[layer + '_l2_' + k])
             for k in ('activity', 'kernel', 'bias')})
    return kwargs_out


def god_model(input_shape, **kwargs):
    """ Bi-directional GRUs for sentiment analysis """
    def_kwargs = {"encoder_layers": 2,
                  "encoder_units": 128,
                  "encoder_activation": 'relu',
                  "encoder_dropout": 0.5,
                  "encoder_recurrent_dropout": 0.2,
                  "encoder_l2_activity": 1e-3,
                  "encoder_l2_kernel": 0.0,
                  "encoder_l2_bias": 0.0,
                  "encoder_final_dropout": 0.5,
                  "analyzer_layers": 2,
                  "analyzer_units": 64,
                  "analyzer_dropout": 0.5,
                  "analyzer_recurrent_dropout": 0.2,
                  "analyzer_activation": 'relu',
                  "analyzer_l2_activity": 1e-3,
                  "analyzer_l2_kernel": 0.0,
                  "analyzer_l2_bias": 0.0,
                  "analyzer_final_dropout": 0.5,
                  "dense_units": 128,
                  "dense_dropout": 0.5,
                  "dense_l2_activity": 1e-3,
                  "dense_l2_kernel": 0.0,
                  "dense_l2_bias": 0.0}

    for key, val in def_kwargs.items():
        if key not in kwargs.keys():
            kwargs.update({key: val})

    # Char to sentence model #################################################
    sent_in = Input((input_shape[-1],), name='characters_input')

    # Embed ASCII int values to a 10 dimensional space
    embed = Embedding(input_dim=50, output_dim=10,
                      input_length=input_shape[-1],
                      mask_zero=True,
                      name='charaters_embedding')(sent_in)

    # Initialize GRU layers
    gru_kwargs = create_kwargs('encoder', kwargs)
    gru = embed
    for i in range(kwargs['encoder_layers']):
        if i == kwargs['encoder_layers'] - 1:
            gru_kwargs['return_sequences'] = False
        gru = Bidirectional(GRU(**gru_kwargs,
                                name='encoder_gru_%d' % (i+1)))(gru)

    # Apply dropout and create model
    sent_out = Dropout(kwargs['encoder_final_dropout'],
                       name='encoder_dropout')(gru)
    encoder = Model(inputs=sent_in, outputs=sent_out)
    encoder.summary()

    # Sentence to doc model ##################################################
    doc_in = Input((input_shape[-2], input_shape[-1],),
                   name='sentences_input')

    # We apply the char to sent model for each sentence in input
    sentences = TimeDistributed(encoder, name='sentences_iterator')(doc_in)

    # Initialize GRU layers again
    gru_kwargs = create_kwargs('analyzer', kwargs)
    gru = sentences
    for i in range(kwargs['analyzer_layers']):
        if i == kwargs['analyzer_layers'] - 1:
            gru_kwargs['return_sequences'] = False
        gru = Bidirectional(GRU(**gru_kwargs,
                                name='analyzer_gru_%d' % (i+1)))(gru)

    doc_out = Dropout(kwargs['analyzer_final_dropout'],
                      name='analyzer_dropout')(gru)

    dense_kwargs = create_kwargs('dense', kwargs)
    doc_out = Dense(**dense_kwargs, name='dense')(doc_out)
    doc_out = Dense(1, activation='relu', name='rating_out')(doc_out)

    model = Model(inputs=doc_in, outputs=doc_out)
    model.compile(optimizer=RMSprop(clipnorm=1), loss=['mae'], metrics=['mse'])
    model.summary()
    return model


def training_session(model, X, y, idc_all, batch_size=32):
    """ Run a training session for given model and data.
    Train on training set, check validation set for convergence and
    finally report score for test set.
    """
    idc_train, idc_val, idc_test = idc_all
    model.fit(
        X[idc_train], y[idc_train], validation_data=(X[idc_val], y[idc_val]),
        epochs=5, verbose=1, batch_size=batch_size,
        callbacks=[EarlyStopping(min_delta=1e-3, verbose=1, patience=0)])
    y_pred = model.predict(X[idc_test])
    n_test = len(idc_test)
    err = mean_absolute_error(y[idc_test], y_pred)
    print("Error on held-out set: %.4g" % err)
    return err, model
