from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import LdaModel
from gensim import matutils
from yellowbrick.text import FreqDistVisualizer
from yellowbrick.features import PCADecomposition
from yellowbrick import set_palette
from sklearn.cluster import DBSCAN


def dbscan_pca(corpus):
    set_palette('sns_deep')  # set visible colors
    cv = CountVectorizer(stop_words='english', token_pattern="\\b[a-z][a-z]+\\b")
    vec = cv.fit_transform(corpus).toarray()
    db = DBSCAN(eps=0.15, min_samples=10)
    clusters = db.fit(PCADecomposition(scale=False).fit_transform(vec))
    visualizer = PCADecomposition(scale=False, color=clusters)
    visualizer.fit_transform(vec)
    visualizer.poof()


def token_frequency_plot(corpus, n_features):
    """Generates plot of most common tokens"""
    corpus_vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english', token_pattern="\\b[a-z][a-z]+\\b")
    doc_matrix = corpus_vectorizer.fit_transform(corpus)
    features = corpus_vectorizer.get_feature_names()
    viz = FreqDistVisualizer(features=features, n=n_features)
    viz.fit_transform(doc_matrix)
    viz.poof()


def perform_lda_iterations(num_topics, num_passes):
    """Performs LDA on up to a specified number of topics with a specified number of passes."""
    tw = joblib.load('../data/clean/tweets-series.pkl')
    rm = joblib.load('../data/clean/remarks-series.pkl')

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
    """Collapses remarks corpus into a series of strings"""
    text = ''
    for statement in lists:
        text += statement + ' '
    return text
