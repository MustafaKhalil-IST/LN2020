from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from preprocess import combine_prepros
from nltk import edit_distance


class Model:
    def __init__(self, approach, coarse):
        self.approach = approach
        self.tfidf = TfidfTransformer(norm='l2')
        self.count_vectorizer = CountVectorizer(analyzer='word', stop_words='english')
        self.train_questions = None
        self.coarse = coarse
        self.coarse_labels = None
        self.fine_labels = None
        self.classes = None

    def preprocess(self, questions, prepros):
        return combine_prepros(questions, prepros)

    def extract_vocab(self, questions):
        res = set()
        for q in questions:
            words = q.split(" ")
            for w in words:
                res.add(w)
        return res

    def tfidf_transform(self, train_qs, new_qs):
        vocab = self.extract_vocab(train_qs)
        self.count_vectorizer = CountVectorizer(vocabulary=vocab)
        self.tfidf = TfidfTransformer(norm='l2')
        train_vectors = self.count_vectorizer.fit_transform(train_qs)
        train_vectors = self.tfidf.fit_transform(train_vectors)
        new_vectors = self.tfidf.transform(self.count_vectorizer.transform(new_qs))
        return train_vectors, new_vectors

    def vectorize(self, questions):
        train_terms = self.count_vectorizer.fit_transform(questions)
        self.tfidf.fit(train_terms)
        train_matrix = self.tfidf.transform(train_terms)
        return train_matrix

    def train(self, train_file_name):
        def parse_train_file(file_name):
            coarse_ls, fine_ls, qs = [], [], []
            with open(file_name, 'r') as f:
                for line in f.readlines():
                    line = line.replace('\n', '')
                    parts = line.split(' ')
                    fine_l, question = parts[0], ' '.join(parts[1:])
                    coarse_l = fine_l.split(':')[0]
                    coarse_ls.append(coarse_l)
                    fine_ls.append(fine_l)
                    qs.append(question)
            return coarse_ls, fine_ls, qs

        coarse_labels, fine_labels, questions = parse_train_file(train_file_name)

        le = LabelEncoder()

        self.train_questions = questions
        if self.coarse:
            self.coarse_labels = le.fit_transform(coarse_labels)
        else:
            self.fine_labels = le.fit_transform(fine_labels)
        self.classes = le.classes_

    def predict(self, dev_file_name, strategy='rf', prepros=[]):
        with open(dev_file_name, "r") as f:
            dev_questions = f.readlines()

        self.train_questions = self.preprocess(self.train_questions, prepros)
        dev_questions = self.preprocess(dev_questions, prepros)

        train_vectors, dev_vectors = self.tfidf_transform(self.train_questions, dev_questions)

        train_labels = self.coarse_labels if self.coarse else self.fine_labels

        if strategy == 'rf':
            predicted_labels = self.rff_strategy(dev_vectors, train_labels, train_vectors)
        elif strategy == 'dt':
            predicted_labels = self.dt_strategy(dev_vectors, train_labels, train_vectors)
        elif strategy.startswith('plots'):
            predicted_labels = self.knn_strategy(dev_vectors, train_labels, train_vectors,
                                                 n_nbrs=int(strategy.split('-')[1]))
        elif strategy == 'svm':
            predicted_labels = self.svm_strategy(dev_vectors, train_labels, train_vectors)
        elif strategy == 'levenshtein':
            predicted_labels = self.levenshtein_strategy(dev_questions, self.train_questions, train_labels)
        else:
            predicted_labels = self.rff_strategy(dev_vectors, train_labels, train_vectors)

        return predicted_labels

    def rff_strategy(self, dev_vectors, train_labels, train_vectors):
        rff = RandomForestClassifier()
        rff.fit(train_vectors, train_labels)
        predicted_labels = rff.predict(dev_vectors)
        return predicted_labels

    def dt_strategy(self, dev_vectors, train_labels, train_vectors):
        dt = DecisionTreeClassifier()
        dt.fit(train_vectors, train_labels)
        predicted_labels = dt.predict(dev_vectors)
        return predicted_labels

    def knn_strategy(self, dev_vectors, train_labels, train_vectors, n_nbrs=5):
        knn = KNeighborsClassifier(n_neighbors=n_nbrs)
        knn.fit(train_vectors, train_labels)
        predicted_labels = knn.predict(dev_vectors)
        return predicted_labels

    def levenshtein_strategy(self, dev_questions, train_questions, train_labels):
        predicted_labels = []
        for question in dev_questions:
            closer_question, closer_distance = None, 100000
            for i, train_question in enumerate(train_questions):
                distance = edit_distance(train_question, question)
                if distance < closer_distance:
                    closer_question, closer_distance = (train_question, i), distance
            predicted_labels.append(train_labels[closer_question[1]])
        return predicted_labels

    def svm_strategy(self, dev_vectors, train_labels, train_vectors):
        knn = SVC()
        knn.fit(train_vectors, train_labels)
        predicted_labels = knn.predict(dev_vectors)
        return predicted_labels
