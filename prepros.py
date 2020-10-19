import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')


def no_prepro(qs):
    return qs


def stop_words_prepro(qs):
    stop_words = set(stopwords.words('english'))
    res = []
    for q in qs:
        words = q.split(' ')
        words = [w for w in words if w not in stop_words]
        res.append(' '.join(words))
    return res