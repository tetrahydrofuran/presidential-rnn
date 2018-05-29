from keras.optimizers import Adam
from keras import Model
from keras.layers import GRU, Dense, Input, Lambda, Bidirectional, Flatten
import keras.backend as K
import numpy as np


def __init_shared_layers(X, y):
    # Shapes
    if len(X.shape) >= 2:
        input_len = int(X.shape[1])
    else:
        input_len = 1
    nb_classes = np.max(X) + 1
    output_len = y.shape[1]  # should be same as X.shape[1]?

    # Input and output layers
    seq_input = Input((input_len,), dtype=np.int32, name='character_input')
    x_ohe = Lambda(K.one_hot, arguments={'num_classes': nb_classes}, output_shape=(input_len, nb_classes))

    output = Dense(units=output_len, activation='softmax')

    return seq_input, x_ohe, output


def rnn_hidden_elu(X, y, bidirectional=False):
    seq_input, x_ohe, output = __init_shared_layers(X, y)

    # Network architecture
    x = x_ohe(seq_input)
    gru = GRU(units=256, dropout=0.2, recurrent_dropout=0.2)
    if bidirectional:
        x = Bidirectional(gru(x))
    else:
        x = gru(x)

    x = Dense(units=512, activation='elu')(x)
    x = output(x)

    opt = Adam(lr=0.01)
    model = Model(seq_input, x)
    model.compile(optimizer=opt, loss='categorical_crossentropy')

    return model


def rnn_basic(X, y, bidirectional=False):
    seq_input, x_ohe, output = __init_shared_layers(X, y)

    x = x_ohe(seq_input)
    gru = GRU(units=256, dropout=0.2, recurrent_dropout=0.2)
    if bidirectional:
        x = Bidirectional(gru(x))
    else:
        x = gru(x)

    x = output(x)
    opt = Adam(lr=0.01)
    model = Model(seq_input, x)
    model.compile(optimizer=opt, loss='categorical_crossentropy')

    return model


def rnn_flanked_elu(X, y, bidirectional=False):
    seq_input, x_ohe, output = __init_shared_layers(X, y)

    x = x_ohe(seq_input)
    x = Dense(units=256, activation='elu')(x)
    gru = GRU(units=256, dropout=0.2, recurrent_dropout=0.2)
    if bidirectional:
        x = Bidirectional(gru(x))
    else:
        x = gru(x)

    x = Dense(units=512, activation='elu')(x)
    x = output(x)

    opt = Adam(lr=0.01)
    model = Model(seq_input, x)
    model.compile(optimizer=opt, loss='categorical_crossentropy')

    return model
