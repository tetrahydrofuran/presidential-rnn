from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import LdaModel
from gensim import matutils


def perform_lda_iterations(num_topics, num_passes):
    """Performs LDA on up to a specified number of topics with a specified number of passes."""
    tw = joblib.load('tweets-series.pkl')
    rm = joblib.load('remarks-series.pkl')

    rm = rm.apply(__unlist)
    tcv = CountVectorizer(stop_words='english', token_pattern="\\b[a-z][a-z]+\\b")
    tdm = tcv.fit_transform(tw).transpose()

    rcv = CountVectorizer(stop_words='english', token_pattern="\\b[a-z][a-z]+\\b")
    rdm = rcv.fit_transform(rm).transpose()

    tc = matutils.Sparse2Corpus(tdm)
    rc = matutils.Sparse2Corpus(rdm)

    tid2word = dict((v, k) for k, v in tcv.vocabulary_.items())
    rid2word = dict((v, k) for k, v in rcv.vocabulary_.items())

    for i in range(2, 20):
        tlda = LdaModel(corpus=tc, num_topics=i, minimum_probability=0.03, id2word=tid2word, passes=20)
        print('Modeled topics at ', i)
        print(tlda.print_topics())

    for i in range(2, 20):
        rlda = LdaModel(corpus=rc, num_topics=i, minimum_probability=0.03, id2word=rid2word, passes=20)
        print('Modeled topics at ', i)
        print(rlda.print_topics())


def __unlist(lists):
    text = ''
    for statement in lists:
        text += statement + ' '
    return text
