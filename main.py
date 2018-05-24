import lib
import pandas as pd
from sklearn.externals import joblib
import logging
import model


def main():
    # Param setup
    lib.set_pred_len(140)
    lib.set_window(40)

    series, corpus = extract(reprocess=False, corpus=True)
    transform(series, corpus)


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
    pass
    # TODO Tensorboard callback

if __name__ == "__main__":
    main()
