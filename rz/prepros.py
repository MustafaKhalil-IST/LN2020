from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import string


def lowering(question):
    return question.lower()


def removing_ponctuations(question):
    return "".join([c for c in question if c not in string.punctuation])


def tokenizing(question):
    tokens = word_tokenize(question)
    return " ".join([token for token in tokens])


def stop_words(question):
    words = question.split(" ")
    stop_words = stopwords.words('english')
    return " ".join([word for word in words if word not in stop_words])


def stemming(question):
    words = question.split(" ")
    porter = PorterStemmer()
    return " ".join([porter.stem(word) for word in words])


def combine_prepros(questions, preprocessors):
    if 'token' in preprocessors:
        questions = [tokenizing(question) for question in questions]
    if 'lower' in preprocessors:
        questions = [lowering(question) for question in questions]
    if 'ponc' in preprocessors:
        questions = [removing_ponctuations(question) for question in questions]
    if 'stem' in preprocessors:
        questions = [stemming(question) for question in questions]
    if 'stop' in preprocessors:
        questions = [stop_words(question) for question in questions]
    return questions
