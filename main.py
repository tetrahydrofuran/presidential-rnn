import lib
import pandas as pd
from sklearn.externals import joblib
import logging


def main():
    extract(reprocess=False, corpus=True)
    extract(reprocess=False, corpus=False)


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


def transform():
    pass


if __name__ == "__main__":
    main()
