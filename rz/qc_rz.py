import sys

from rz.methods import predict_random, predict_levenshtein

from rz.common import train_and_predict

args = sys.argv
coarse = args[1] == '-coarse'
train_fname = args[2]
target_q_fname = args[3]

_, predicted_ls = train_and_predict(coarse, train_fname, target_q_fname, ['stop_wh'], predict_levenshtein)

for l in predicted_ls:
    print(l)
