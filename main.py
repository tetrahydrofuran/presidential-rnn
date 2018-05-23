import lib
import pandas as pd
from sklearn.externals import joblib

def main():
    pass


def extract(reprocess=True, corpus=True):
    if reprocess:
        if corpus:
            df = pd.read_csv('../data/text.csv')
            df = lib.get_remarks_corpus(df, combined=False)
            combined = lib.get_remarks_corpus(df, combined=True)
            return df, combined
        else:
            df = pd.read_csv('../data/tweets.csv')
            df = lib.get_tweet_corpus(df, combined=False)
            combined = lib.get_tweet_corpus(df, combined=True)
            return df, combined
    else:
        if corpus:
            joblib.load('../data/clean/remarks.pkl')
            joblib.load('../data/clean/remarks-series.pkl')
        else:
            joblib.load('../data/clean/tweets.pkl')
            joblib.load('../data/clean/tweets-series.pkl')



if __name__ == "__main__":
    main()
