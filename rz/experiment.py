import sys

from rz.file_ops import *
from rz.common import do_evaluate, train_and_predict
from rz.methods import predict_levenshtein, predict_random, predict_knn, predict_svm


def do_experiment(coarse, prepros, method):
    sys.stderr.write(f'coarse is {coarse}\n')
    sys.stderr.write(f'prepro is {prepros}\n')
    sys.stderr.write(f'method is {method}\n')
    new_qs, predicted_ls = train_and_predict(coarse, 'TRAIN.txt', 'DEV-questions.txt', prepros, method)
    true_coarse_ls, true_fine_ls = parse_l_file("DEV-labels.txt")
    true_ls = true_coarse_ls if coarse else true_fine_ls
    do_evaluate(true_ls, predicted_ls)


# coarse 16
# do_experiment(True, [], predict_random)

# coarse 71
# do_experiment(True, ['stop'], predict_levenshtein)

# coarse 72, fine 60
# do_experiment(True, ['stop_wh'], predict_levenshtein)

# coarse 70
# do_experiment(True, ['token', 'lower', 'ponc', 'stem', 'stop_wh'], predict_levenshtein)

# coarse 59
# do_experiment(True, ['token', 'lower', 'ponc', 'stem', 'stop_wh'], predict_knn)

# coarse 68, fine 56
# do_experiment(False, [], predict_knn)

# coarse 79
# do_experiment(True, ['token', 'lower', 'ponc', 'stem', 'stop_wh'], predict_svm)

# coarse 82, fine 69
# do_experiment(False, [], predict_svm)
