import sys

from common import train_and_predict
from methods import predict_random
from prepros import no_prepro

args = sys.argv
coarse = args[1] == '-coarse'
train_fname = args[2]
target_q_fname = args[3]

_, predicted_ls = train_and_predict(coarse, train_fname, target_q_fname, no_prepro, predict_random)

for l in predicted_ls:
    print(l)
