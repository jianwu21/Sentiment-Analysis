from sklearn.metrics import mean_absolute_error
from keras.models import Model
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Input
from keras.layers import GlobalAveragePooling1D
from keras.layers.merge import Concatenate
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop
from keras.layers import Embedding


# HELPER FUNCTIONS ###########################################################
def conv1d(x_in, n_filters, window):
    """ Return 1D convolutional layer.
    It shifts a number of filters over a number of words to extract
    relevant spatial features.
    Arguments:
        x_in (Keras layer): layer input
        n_filters (int): How many filters to extract
        window (int): How many words should the filters see
    Returns:
        Keras layer
    """
    return Conv1D(n_filters, window, activation='relu', padding='same')(x_in)


def poolconv1d(x_in, n_filters, window, dropout):
    """ Reduce a 1D convolutional layer to half and apply another on top.
    It extracts the max value for every two filters and reduces the
    spatial component to half.
    Arguments:
        x_in (Keras layer): layer input
        n_filters (int): How many filters to extract
        window (int): How many words should the filters see
        dropout (float): Dropout value
    Returns:
        Keras layer
    """
    pool = MaxPooling1D(2, padding='same')(x_in)
    drop = Dropout(dropout)(pool)
    return Conv1D(n_filters, window, activation='relu')(drop)


def concat_pool(tensors, dropout):
    """ Concatenate some convolutional layers and average them over
    the spatial dimension, i.e. keep the most activated filters.
    Arguments:
        tensors (list): List of keras layers to pool
        dropout (float): Dropout value
    """
    merge = Concatenate(axis=-1)(tensors)
    pool = GlobalAveragePooling1D()(merge)
    return Dropout(dropout)(pool)


def concat(tensors):
    """ It's just convenience"""
    return Concatenate()(tensors)


# PUBLIC FUNCTIONS ###########################################################
def make_model(input_shape, conv_size, n_dense, dropout=0.5, clipnorm=1,
               mode='multi'):
    """ Make a convolutional sentiment analyzer.
    Specifically, the model comprises of:
        INPUT
            (n_samples n_x words x 1
            The input contains each document as a sequence of integers,
            where integers are given by a hash-mapping or simillar technique.

        EMBEDDING
            We embed the integer values to a 400D space.

        CONVOLUTIONAL LAYERS
            We apply consecutive convolutional layers. For each pass we
            apply 2 layers, one of window of 2 words and one of 5, both of
            conv_size[i] filters. Before each additional layer we pool the
            previous layer using Max Pooling ad apply Dropout. Finally,
            we Average Pool each temporal dimension.

        DENSE
            We feed each averaged temporal dimension in a fully-connected
            layer of size n_dense.

        OUTPUT
            We can have either a continuous value or 3 different classes
            for output.
    Arguments:
        input_shape (tuple): as given by X_train.shape
        conv_size (tuple): Size and shape of layer, i.e. (32, 32)
            for a 2-level deep network with 32 filters for each level
        n_dense (int): Number of nodes for the dense layer
        dropout(float): Dropout value
        clipnorm (float): Threshold weight's norm for clipping
        mode (string): one of ['single', 'multi'].
            If 'multi', the model needs two y-values, one for the
            regression task and one for the classification task
    Returns:
        model (Keras model) if mode == 'multi'
        model_regression, model_classification if mode =='single'
    """
    x_in = Input((input_shape[1],))
    embed = Embedding(6000, 400, input_length=input_shape[1])(x_in)

    conv_depth = len(conv_size)

    # Create the first layer
    conv_layers = [[conv1d(embed, conv_size[0], 2),
                    conv1d(embed, conv_size[0], 5)]]

    for size in conv_size[1:]:
        # Create an extra layer with window 3 for each layer in previous level
        temp = [poolconv1d(x, size, 3, dropout) for x in conv_layers[-1]]
        conv_layers.append(temp)

    # Merge layers for each level
    if conv_depth == 1:
        merged = concat_pool(conv_layers[0], dropout)
    else:
        merged = concat([concat_pool(l, dropout) for l in conv_layers])

    # Fully connected layer
    dense = Dense(n_dense, activation='relu')(merged)

    # Outputs
    y_clf = Dense(3, activation='softmax', name='class')(dense)
    y_reg = Dense(1, activation='relu', name='rating')(Concatenate()([dense,
                                                                      y_clf]))

    # Optimizer
    opt = RMSprop(clipnorm=clipnorm)

    if mode == 'single':
        model_clf = Model(inputs=x_in, outputs=y_clf)
        model_clf.compile(optimizer=opt, loss='binary_crossentropy',
                          metrics=['accuracy'])

        model_reg = Model(inputs=x_in, outputs=y_reg)
        model_reg.compile(optimizer=opt, loss='mae', metrics=['mse'])

        return model_reg, model_clf

    else:
        if mode != 'multi':
            print("Unrecognized option ", mode, " for 'mode' argument, "
                  "returning multi")
        model = Model(inputs=x_in, outputs=[y_reg, y_clf])
        model.compile(optimizer=opt, loss=['mae', 'binary_crossentropy'],
                      metrics={'class': 'accuracy'})
        return model


def training_session(model, data, mode='multi'):
    """ Run a training session for given model and data.
    Train on training set, check validation set for convergence and
    finally report score for test set.
    Arguments:
        model (Keras model)
        data (tuple): ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        mode (str): one of ['single', 'multi']
    Returns:
        error_value (float), model (Keras model)
    """
    if mode == 'single':
        def func(x): return x
    else:
        def func(x): return x[0], x[1]

    train_set, val_set, test_set = data
    (X_train, y_train), (X_val, y_val) = train_set, val_set
    model.fit(X_train, func(y_train), validation_data=(X_val, func(y_val)),
              batch_size=64, epochs=3, shuffle=True, verbose=1,
              callbacks=[EarlyStopping(min_delta=1e-6, verbose=1, patience=0)])
    y_pred = model.predict(test_set[0])
    err = mean_absolute_error(test_set[1], y_pred)
    print("Error on held-out set: %.4g" % err)
    return err, model
