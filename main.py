import lib
import pandas as pd
from sklearn.externals import joblib
import logging
import models
import numpy as np
import os
from keras_callbacks import SentenceGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import RMSprop


def main():
    # Param setup
    lib.set_pred_len(140)
    lib.set_window(40)
    lib.set_test(True)
    lib.set_model_params(batch=4096, epochs=200)

    # extract(reprocess=True, corpus=True)
    # extract(reprocess=True, corpus=False)
    series = joblib.load('tweets-series.pkl')
    corpus = lib.__combine_tweet_corpus(series)
    x, y = transform(series, corpus)



def extract(reprocess=True, corpus=True):
    logging.debug('extract()')
    if reprocess:
        if corpus:
            df = pd.read_csv('../data/text.csv')
            dfc = lib.get_remarks_corpus(df, combined=False)
            combined = lib.get_remarks_corpus(df, combined=True)
        else:
            df = pd.read_csv('../data/tweets.csv')
            dfc = lib.get_tweet_corpus(df, combined=False)
            combined = lib.get_tweet_corpus(df, combined=True)

    else:
        if corpus:
            combined = joblib.load('../data/clean/remarks.pkl')
            dfc = joblib.load('../data/clean/remarks-series.pkl')
        else:
            combined = joblib.load('../data/clean/tweets.pkl')
            dfc = joblib.load('../data/clean/tweets-series.pkl')

    return dfc, combined


def transform(series, corpus):
    lib.map_corpus(corpus, sprintable=True)
    x_agg = list()
    y_agg = list()

    if lib.is_test:
        series = series.head(5).copy()

    # Iterate through individual observations
    for text in series:
        # Generate x and y for observations
        observation_x, observation_y = lib.get_input_target(text, false_y=False)
        x_agg.extend(observation_x)
        y_agg.extend(observation_y)

    x = np.matrix(x_agg)
    y = np.matrix(y_agg)
    return x, y


def model(x, y, model_index, bidirectional):
    if model_index == 0:
        model = models.rnn_basic(x, y, bidirectional)
    elif model_index == 1:
        model = models.rnn_hidden_elu(x, y, bidirectional)
    elif model_index == 2:
        model = models.double_rnn_hidden_elu(x, y, bidirectional)
    else:
        model = models.rnn_flanked_elu(x, y, bidirectional)

    if lib.is_test:
        epochs = 2
    else:
        epochs = lib.num_epochs

    tf_path = os.path.join(os.path.expanduser('~/log-dir'), lib.get_batch())
    mc_path = os.path.join(os.path.expanduser('~/mc-dir'), lib.get_batch() + '_epoch_{epoch:03d}_loss_{loss:.2f}.h5py')
    cb_path = 'callbacks.csv'
    sent_gen = SentenceGenerator(output_path=cb_path, verbose=1)

    optimizer = RMSprop()
    callbacks = [TensorBoard(log_dir=tf_path),
                 ModelCheckpoint(filepath=mc_path),
                 sent_gen]

    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    model.fit(x, y, batch_size=lib.batch_size, epochs=epochs, callbacks=callbacks)


if __name__ == "__main__":
    main()
