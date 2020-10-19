import sys

from file_ops import parse_l_file
from predict import do_evaluate

args = sys.argv
true_l_fname = args[1]
predicted_l_fname = args[2]

true_coarse_ls, true_fine_ls = parse_l_file(true_l_fname)

coarse = None
predicted_ls = []
with open(predicted_l_fname, 'r') as f:
    for line in f.readlines():
        curr_coarse = ':' not in line
        if coarse is None:
            coarse = curr_coarse
        elif coarse != curr_coarse:
            raise Exception("predicted labels contain both coarse and fine")
        predicted_ls.append(line)

do_evaluate(true_coarse_ls if coarse else true_fine_ls, predicted_ls)
