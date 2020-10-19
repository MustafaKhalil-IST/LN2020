from file_ops import *
from common import do_evaluate, train_and_predict
from methods import predict_random, predict_levenshtein


def do_experiment(coarse, method):
    print(f'coarse is {coarse}')
    print(f'method is {method}')
    new_qs, predicted_ls = train_and_predict(coarse, 'TRAIN.txt', 'DEV-questions.txt', method)
    true_coarse_ls, true_fine_ls = parse_l_file("DEV-labels.txt")
    true_ls = true_coarse_ls if coarse else true_fine_ls
    do_evaluate(true_ls, predicted_ls)

# accuracy 16
do_experiment(True, predict_random)

# accuracy 66, but takes forever to process
do_experiment(True, predict_levenshtein)
