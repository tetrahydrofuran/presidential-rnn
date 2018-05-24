from sklearn.externals import joblib
from unidecode import unidecode
import re
import time
from collections import Counter
import numpy as np
import pandas as pd


def markov_main():
    # TODO placeholder locations
    df_tweet = joblib.load('tweets-series.pkl')
    df_remarks = joblib.load('remarks-series.pkl')
    df_tweet.apply(__prep_tweets)
    df_remarks.apply(__prep_remarks)
    ft_remarks = __generate_freq_table(df_remarks)
    ft_tweets = __generate_freq_table(df_tweet)

    joblib.dump(ft_remarks, 'ft_remarks.pkl')
    joblib.dump(ft_tweets, 'ft_tweets.pkl')


def __generate_freq_table(text_series):
    def time_since(since):
        s = time.time() - since
        m = s // 60
        s -= m * 60
        return '%dm %ds' % (m, s)

    start = time.time()

    before = []
    after = []
    for text in text_series:
        tokens = text.split(' ')
        before += tokens[:-1]
        after += tokens[1:]

    frequency_table = {}
    progress = 0
    while before:
        individual_word = before[0]
        same_word = []
        for index, word in enumerate(before):
            if individual_word == word:
                same_word.append(after[index])

        word_counter = Counter(same_word)
        words = []
        counts = []

        for item in list(word_counter.items()):
            words.append(item[0])
            counts.append(item[1])
        counts = np.divide(counts, sum(counts))  # probability
        frequency_table[before[0]] = {
            'words': words,
            'probs': counts
        }

        try:
            while True:
                before.remove(individual_word)
        except ValueError:
            progress += 1
            print('Progress: ', progress, len(before), progress / len(before) * 100, '%', time_since(start))
            continue
    return pd.DataFrame(frequency_table)


def __prep_tweets(text):
    text = unidecode(text)
    text = re.sub(r'\n', ' ', text)

    # Insert start tokens
    #    text = re.sub(r'<', ' ', text)
    #    text = '<' + text
    text = text.strip()
    text = re.sub(r'https?:[A-Za-z.\/0-9]+', '', text)  # remove hyperlinks
    text = re.sub(r'--', ' ', text)  # remove double dash
    text = re.sub(r'@\w+', '', text)  # remove usernames
    text = re.sub('\.\.\.\.?', '', text)  # remove ellipses
    text = re.sub(r'^\.', '', text)  # remove start periods
    text = re.sub(r'[^ A-Za-z0-9.,]', '', text)  # other punctuation
    text = re.sub(r'\.', ' .', text)  # space before periods

    # normalize ish
    tokens = text.split(' ')
    for word in tokens:
        if len(word) > 15:
            tokens.remove(word)
    text = ' '.join(tokens)
    return text.strip()


def __prep_remarks(text):
    return ' '.join(text)


if __name__ == '__main__':
    markov_main()
