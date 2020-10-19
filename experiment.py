from file_ops import *
from predict import do_evaluate, train_and_predict

coarse = False

new_qs, predicted_ls = train_and_predict(coarse, 'TRAIN.txt', 'DEV-questions.txt')

true_coarse_ls, true_fine_ls = parse_l_file("DEV-labels.txt")
true_ls = true_coarse_ls if coarse else true_fine_ls
do_evaluate(true_ls, predicted_ls)
