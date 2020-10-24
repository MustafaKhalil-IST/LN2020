
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
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
            questions = f.readlines()

        all_questions = self.train_questions + questions

        all_questions = self.preprocess(all_questions, prepros)

        all_vectors = self.vectorize(all_questions)

        train_vectors = all_vectors[:len(self.train_questions)]
        dev_vectors = all_vectors[len(self.train_questions):]
        train_labels = self.coarse_labels if self.coarse else self.fine_labels

        if strategy == 'rf':
            predicted_labels = self.rff_strategy(dev_vectors, train_labels, train_vectors)
        elif strategy == 'dt':
            predicted_labels = self.dt_strategy(dev_vectors, train_labels, train_vectors)
        elif strategy == 'knn':
            predicted_labels = self.knn_strategy(dev_vectors, train_labels, train_vectors)
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

    def knn_strategy(self, dev_vectors, train_labels, train_vectors):
        knn = KNeighborsClassifier(n_neighbors=5)
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


        pass