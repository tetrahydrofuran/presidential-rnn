from unidecode import unidecode
import re
import logging
import numpy as np
from sklearn.externals import joblib
import datetime
import string


characters = None
char_to_index = None
index_to_char = None
data_size = None
char_size = None
batch_name = None
window_len = None
predict_len = None
is_test = None
batch_size = None
num_epochs = None


# region Corpus Preparation
# region Remarks Corpus
def get_remarks_corpus(df, combined=False):
    df['text_filter'] = df['text'].apply(__extract_donald)
    df['text_filter'] = df['text_filter'].apply(__apply_unidecode)
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


# region Remarks Helpers
def __apply_unidecode(lists):
    out = []
    for statement in lists:
        out.append(unidecode(statement))
    return out


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
    text = re.sub(r'\([A-Za-z0-9. ]+\)', '', text)  # eg (Applause.), (Laughter.)
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
    df = __clean_tweet_corpus(df)
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


def __combine_tweet_corpus(text):
    logging.debug('combine_tweet_corpus(): Concatenating text.')
    corpus = ''
    for words in text:
        corpus += words + ' '
    return corpus


def __clean_tweet_corpus(df):
    """Some items inputted incorrectly"""
    bad = []
    for i, text in enumerate(df['text']):
        if ',false,' in text:
            bad.append(i)
    return df.reset_index().drop(bad)
# endregion
# endregion


def map_corpus(corpus, sprintable=True):
    """
    Creates set of characters, and creates mapping of unique characters to integer and inverse, assigned
    to global scope
    :param corpus:
    :param sprintable:
    :return:
    """

    global characters
    global char_to_index
    global index_to_char
    global data_size
    global char_size

    if sprintable:
        characters = set(string.printable + '<>')
    else:
        characters = set(corpus + '<>')
    characters = sorted(list(characters))

    data_size = len(corpus)
    char_size = len(characters)
    char_to_index = {ch: i for i, ch in enumerate(characters)}
    index_to_char = {i: ch for i, ch in enumerate(characters)}
    logging.debug('map_corpus(): Data size: ' + str(data_size))
    logging.debug('map_corpus(): Char size: ' + str(char_size))


def sample(preds, temperature=1.0):
    pass


def get_input_target(text, false_y=False):
    chars = set(char_to_index.keys())
    char_indices = char_to_index
    indices_char = index_to_char
    sentences = list()
    next_chars = list()
    step = 3

    # Add start and end characters
    text = re.sub('<', ' ', text)
    text = re.sub('>', ' ', text)
    text = '<' + text + '>>>>>>>>>>>>'

    text = map(lambda x: x.lower(), text)
    text = map(lambda x: x if x in chars else ' ', text)
    text = ''.join(text)

    if false_y:
        # Add a padding character, so which will become the Y
        text +=' '
    else:
        # Replace multiple whitespaces with a single whitespace
        text = re.sub(r'\s+', ' ', text)


    # Cut the text in semi-redundant sequences of maxlen characters
    for index in range(0, len(text) - window_len, step):
        sentences.append(text[index: index + window_len])
        next_chars.append(text[index + window_len])

    # Convert all sequences into X and Y matrices
    x = np.zeros((len(sentences), window_len))
    y = np.zeros((len(sentences), len(chars)), dtype=bool)

    for index, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[index, t] = char_indices[char]
        y[index, char_indices[next_chars[index]]] = 1

    return x, y


# region Getters and Setters
def get_batch():
    global batch_name
    if batch_name is None:
        logging.info('Batch name not yet set. Setting batch name.')
        batch_name = str(datetime.datetime.utcnow()).replace(' ', '_').replace('/', '_').replace(':', '_')
        logging.info('Batch name: {}'.format(batch_name))
    return batch_name


def set_window(length):
    global window_len
    window_len = length


def set_pred_len(length):
    global predict_len
    predict_len = length


def set_test(test):
    global is_test
    is_test = test


def set_model_params(batch, epochs):
    global batch_size
    global num_epochs
    batch_size = batch
    num_epochs = epochs

# endregion
