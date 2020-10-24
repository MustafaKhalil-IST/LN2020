from rz.file_ops import *
from rz.common import do_evaluate, train_and_predict
from rz.methods import predict_levenshtein, predict_random


def do_experiment(coarse, prepros, method):
    print(f'coarse is {coarse}')
    print(f'prepro is {prepros}')
    print(f'method is {method}')
    new_qs, predicted_ls = train_and_predict(coarse, 'TRAIN.txt', 'DEV-questions.txt', prepros, method)
    true_coarse_ls, true_fine_ls = parse_l_file("DEV-labels.txt")
    true_ls = true_coarse_ls if coarse else true_fine_ls
    do_evaluate(true_ls, predicted_ls)


# accuracy 16
# do_experiment(True, [], predict_random)

# accuracy 71
# do_experiment(True, ['stop'], predict_levenshtein)

# accuracy 72
# do_experiment(True, ['stop_wh'], predict_levenshtein)

# acc 70
# do_experiment(True, ['token', 'lower', 'ponc', 'stem', 'stop_wh'], predict_levenshtein)
