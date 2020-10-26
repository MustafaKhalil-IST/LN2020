from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import re


def clean(question):
    def removing_extra_spaces(question):
        return re.sub(r"\s\s+", r" ", question)

    def removing_ponctuations(question):
        return "".join([c for c in question if c not in string.punctuation])

    def removing_breakline(question):
        return question[:-1] if question[-1] == '\n' else question
    return removing_extra_spaces(removing_ponctuations(removing_breakline(question)))


def lowering(question):
    return question.lower()


def tokenizing(question):
    tokens = word_tokenize(question)
    return " ".join([token for token in tokens])


def stemming(question):
    stop_words = stopwords.words('english')
    for word in ['what', 'which', 'why', 'who', 'where', 'whom', 'how']:
        stop_words.remove(word)
    words = question.split(" ")
    filtered_words = [word for word in words if word not in stop_words]
    porter = PorterStemmer()
    return " ".join([porter.stem(word) for word in filtered_words])


def combine_prepros(questions, preprocessors):
    if 'token' in preprocessors:
        questions = [tokenizing(clean(question)) for question in questions]
    if 'lower' in preprocessors:
        questions = [lowering(clean(question)) for question in questions]
    if 'stem' in preprocessors:
        questions = [stemming(clean(question)) for question in questions]
    return questions
