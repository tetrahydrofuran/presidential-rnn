from unidecode import unidecode
import re
import logging
from sklearn.externals import joblib

char_to_index = {}
index_to_char = {}
data_size = 0
char_size = 0


# region Corpus Preparation
# region Remarks Corpus
def get_remarks_corpus(df, combined=False):
    df['text_filter'] = df['text'].apply(__extract_donald)
    df['text_filter'] = df['text_filter'].apply(unidecode)
    df['is_donald'] = df['text_filter'].apply(__get_donald)
    df = df[df['is_donald']]  # filter out not-Donald
    if combined:
        corpus = __combine_remark_corpus(df['text_filter'])
        logging.debug('get_remarks_corpus(): Dumping to remarks.pkl')
        joblib.dump(corpus, '../data/clean/remarks.pkl')
        return corpus
    else:
        corpus = df['text_filter']
        logging.debug('get_remarks_corpus(): Dumping to remarks-series.pkl')
        joblib.dump(corpus, '../data/clean/remarks-series.pkl')
        return corpus


# TODO: test to catch 'Applause and Laughter.'
# region Remarks Helpers
def __extract_donald(text):
    # Split and rejoin to remove special whitespace characters
    text = text.split()
    text = ' '.join(text)
    trump_speech = re.compile(r"(PRESIDENT TRUMP|THE PRESIDENT): (.*?) ([A-Z’']+ ?[A-Z'’]+?:|Q ?|END)")
    results = trump_speech.findall(text)

    list_results = []
    for result in results:
        list_results.append(__filter_annotations(result[1]))
    return list_results


def __filter_annotations(text):
    text = re.sub(r'\(\w+\.?\)', '', text)  # eg (Applause.), (Laughter.)
    text = re.sub(r'\(In progress\)', '', text)
    return text


def __get_donald(lists):
    if lists:  # Empty list means no Donald
        return True
    return False


def __combine_remark_corpus(text):
    logging.debug('combine_remark_corpus(): Concatenating text.')
    corpus = ''
    for speech in text:
        for paragraph in speech:
            corpus += paragraph
            corpus += ' '
    return corpus
# endregion
# endregion


# region Tweets Corpus
def get_tweet_corpus(df, combined=False):
    df = df[df['is_retweet'] == 'false']
    if combined:
        corpus = __combine_tweet_corpus(df['text'])
        logging.debug('get_tweet_corpus(): Dumping to tweets.pkl')
        joblib.dump(corpus, '../data/clean/tweets.pkl')
        return corpus
    else:
        corpus = df['text']
        logging.debug('get_tweet_corpus(): Dumping to tweets-list.pkl')
        joblib.dump(corpus, '../data/clean/tweets-series.pkl')
        return corpus


# TODO: start or end tokens
def __combine_tweet_corpus(text):
    logging.debug('combine_tweet_corpus(): Concatenating text.')
    corpus = ''
    for words in text:
        corpus += words + ' '
    return corpus
# endregion
# endregion


def map_corpus(corpus):
    """
    Creates set of characters, and creates mapping of unique characters to integer and inverse, assigned
    to global scope
    :param corpus:
    :return:
    """
    characters = list(set(corpus))

    global char_to_index
    global index_to_char
    global data_size
    global char_size

    data_size = len(corpus)
    char_size = len(characters)
    char_to_index = {ch: i for i, ch in enumerate(characters)}
    index_to_char = {i: ch for i, ch in enumerate(characters)}
    logging.debug('map_corpus(): Data size: ' + str(data_size))
    logging.debug('map_corpus(): Char size: ' + str(char_size))