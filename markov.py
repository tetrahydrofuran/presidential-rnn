from sklearn.externals import joblib
from unidecode import unidecode
import re
import time
from collections import Counter
import numpy as np
import pandas as pd
import sys


def markov_main():
    # TODO if os.path.isfile etc.
    prep_markov_chain()


def prep_markov_chain():
    # TODO placeholder locations
    df_tweet = joblib.load('tweets-series.pkl')
    df_remarks = joblib.load('remarks-series.pkl')
    df_tweet = df_tweet.apply(__prep_tweets)
    df_remarks = df_remarks.apply(__prep_remarks)

    ft_remarks = __generate_freq_table(df_remarks)
    joblib.dump(ft_remarks, 'ft_remarks.pkl')
    ft_remarks.to_csv('ft_remarks.csv')

    ft_tweets = __generate_freq_table(df_tweet)
    joblib.dump(ft_tweets, 'ft_tweets.pkl')
    ft_tweets.to_csv('ft_tweets.csv')


def __generate_freq_table(text_series):
    """
    Really slow implementation of creating a frequency table for a Markov chain.
    :param text_series: List of documents to 'train' Markov on
    :type text_series: pandas Series, list, or equivalent
    :return: pandas DataFrame of frequency distribution table
    """
    def time_since(since):
        """Simple timer"""
        s = time.time() - since
        m = s // 60
        s -= m * 60
        return '%dm %ds' % (m, s)

    def update_progress(progress):
        """Outputs progress to stdout"""
        progress += 1
        progress_percentage = og_length / len(before) * 100
        prog_str = f'Progress: {progress_percentage}%; Items: {progress}:{len(before)}; {time_since(start)} \n'
        sys.stdout.write(prog_str)
        sys.stdout.flush()
        return progress

    def checkpoint_progress():
        end = str(progress) + '.pkl'
        joblib.dump(after, 'after' + end)
        joblib.dump(before, 'before' + end)
        joblib.dump(frequency_table, 'freq' + end)

    start = time.time()


    # Generate list of words and their matching following word
    before = []
    after = []
    for text in text_series:
        tokens = text.split(' ')
        before += tokens[:-1]
        after += tokens[1:]

    og_length = len(after)
    # Init frequency_table and progress counter
    frequency_table = {}
    progress = 0

    # While there are still words in the 'before' list, find all the words in the 'after' list that correspond
    while before:
        individual_word = before[0]
        same_word = []
        for index, word in enumerate(before):
            if individual_word == word:
                same_word.append(after[index])

        # Obtain counts of each 'after' word
        word_counter = Counter(same_word)
        words = []
        counts = []

        for item in list(word_counter.items()):
            words.append(item[0])
            counts.append(item[1])

        # Determine frequency of each 'after' word, add to frequency_table
        counts = np.divide(counts, sum(counts))  # probability
        frequency_table[before[0]] = {
            'words': words,
            'probs': counts
        }

        # Remove each instance of word until none remain, at which point ValueError will be thrown
        # Progress will update, and will continue to the next word
        # I know this nested try except and intended exception throwing is kind of janky, but it does the job
        try:
            while True:
                index_to_remove = before.index(individual_word)
                before.pop(index_to_remove)
                after.pop(index_to_remove)
        except (ValueError, IndexError):
            try:
                progress = update_progress(progress)
                if progress % 50 == 0:
                    checkpoint_progress()

            except ZeroDivisionError:
                continue
            continue

    return pd.DataFrame.from_dict(frequency_table, orient='index').reset_index()


def __prep_tweets(text):
    text = unidecode(text)
    text = re.sub(r'\n', ' ', text)

    text = text.strip()
    text = re.sub(r'https?:[A-Za-z.\/0-9]+', '', text)  # remove hyperlinks
    text = re.sub(r'--', ' ', text)  # remove double dash
    text = re.sub(r'@\w+', '', text)  # remove usernames
    text = re.sub('\.\.\.\.?', '', text)  # remove ellipses
    text = re.sub(r'^\.', '', text)  # remove start periods
    text = re.sub(r'[^ A-Za-z0-9.,]', '', text)  # other punctuation
    text = re.sub(r',', ' ,', text)  # spaces before commas
    text = re.sub(r'\.', ' .', text)  # space before periods

    # normalize ish
    tokens = text.split(' ')
    for word in tokens:
        if len(word) > 15:
            tokens.remove(word)
    text = ' '.join(tokens)
    return text.strip()


def __prep_remarks(text):
    text = ' '.join(text)
    text = unidecode(text)
    text = re.sub(r'\.', ' .', text)  # space before periods
    text = re.sub(r',', ' ,', text)  # spaces before commas
    text = re.sub(r'[^ A-Za-z0-9.,]', '', text)  # other punctuation
    return text


if __name__ == '__main__':
    markov_main()
