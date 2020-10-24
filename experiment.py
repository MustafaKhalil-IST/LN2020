from file_ops import *
from common import do_evaluate, train_and_predict
from methods import predict_random, predict_levenshtein


def do_experiment(coarse, prepros, method):
    print(f'coarse is {coarse}')
    print(f'prepro is {prepros}')
    print(f'method is {method}')
    new_qs, predicted_ls = train_and_predict(coarse, 'TRAIN.txt', 'DEV-questions.txt', prepros, method)
    true_coarse_ls, true_fine_ls = parse_l_file("DEV-labels.txt")
    true_ls = true_coarse_ls if coarse else true_fine_ls
    do_evaluate(true_ls, predicted_ls)


# accuracy 16
do_experiment(True, ['lower', 'ponc'], predict_levenshtein)

# accuracy 71, but takes forever to process
# do_experiment(True, stop_words_prepro, predict_levenshtein)
