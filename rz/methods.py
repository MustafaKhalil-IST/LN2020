import random
import sys

from nltk import edit_distance
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


def predict_random(all_ls, train_qs, train_ls, new_qs):
    res = []
    for q in new_qs:
        # filling with random
        predicted_l = random.choice(all_ls)
        # filling with constant
        # predicted_l = all_ls[0]
        res.append(predicted_l)
    return res


def predict_levenshtein(all_ls, train_qs, train_ls, new_qs):
    idx = 0
    count = len(new_qs)
    train_qs = [q.split(' ') for q in train_qs]
    new_qs = [q.split(' ') for q in new_qs]
    res = []
    for q in new_qs:
        if idx % 10 == 0:
            sys.stderr.write(f'entry {idx} of {count}\n')
        min_dist = 999999
        candidate = None
        for tl, tq in zip(train_ls, train_qs):
            dist = edit_distance(q, tq)
            if dist < min_dist:
                min_dist = dist
                candidate = tl
        res.append(candidate)
        idx += 1
    return res


def extract_vocab(qs):
    res = set()
    for q in qs:
        words = q.split(" ")
        for w in words:
            res.add(w)
    return res


def tfidf_transform(train_qs, new_qs):
    vocab = extract_vocab(train_qs)
    count_vectorizer = CountVectorizer(vocabulary=vocab)
    tfidf = TfidfTransformer(norm='l2')
    train_vectors = count_vectorizer.fit_transform(train_qs)
    train_vectors = tfidf.fit_transform(train_vectors)
    new_vectors = tfidf.transform(count_vectorizer.transform(new_qs))
    return train_vectors, new_vectors

def predict_knn(all_ls, train_qs, train_ls, new_qs):
    train_vectors, new_vectors = tfidf_transform(train_qs, new_qs)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_vectors, train_ls)
    predicted_ls = knn.predict(new_vectors)
    return predicted_ls


def predict_svm(all_ls, train_qs, train_ls, new_qs):
    train_vectors, new_vectors = tfidf_transform(train_qs, new_qs)
    s = svm.SVC()
    s.fit(train_vectors, train_ls)
    predicted_ls = s.predict(new_vectors)
    return predicted_ls
