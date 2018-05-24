import lib
from keras.optimizers import Adam
from keras import Model
from keras.layers import GRU, Dense, Embedding, Input


def rnn_hidden_elu(X, y):
    if len(X.shape) >= 2:
        embedding_input_length = int(X.shape[1])
    else:
        embedding_input_length = 1

    embedding_input_dim = lib.char_size
    embedding_output_dim = int(min((embedding_input_dim + 1) / 2, 50))

    # Input and output layers
    seq_input = Input((embedding_input_dim, ), name='character_input')
    embed = Embedding(input_dim=embedding_input_dim,
                      output_dim=embedding_output_dim,
                      input_length=embedding_input_length,
                      trainable=True,
                      name='character_embedding')
    output_len = y.shape[1]  # should be same as X.shape[1]?
    output = Dense(units=output_len, activation='softmax')

    # Network architecture
    x = embed(seq_input)
    x = GRU(units=256, dropout=0.2, recurrent_dropout=0.2)(x)
    x = Dense(units=512, activation='elu')(x)
    x = output(x)

    opt = Adam(lr=0.01)
    model = Model(seq_input, x)
    model.compile(optimizer=opt, loss='categorical_crossentropy')

    return model

