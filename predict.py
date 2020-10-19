import random

from file_ops import parse_train_file, parse_q_file

random.seed(421)

COARSE_LABELS = ['ABBR', 'DESC', 'ENTY', 'HUM', 'LOC', 'NUM']
FINE_LABELS = [
    'ABBR:abb', 'ABBR:exp',
    'DESC:def', 'DESC:desc', 'DESC:manner', 'DESC:reason',
    'ENTY:animal', 'ENTY:body', 'ENTY:color', 'ENTY:cremat', 'ENTY:currency', 'ENTY:dismed', 'ENTY:event', 'ENTY:food',
    'ENTY:instru', 'ENTY:lang', 'ENTY:letter', 'ENTY:other', 'ENTY:plant', 'ENTY:product', 'ENTY:religion',
    'ENTY:sport', 'ENTY:substance', 'ENTY:symbol', 'ENTY:techmeth', 'ENTY:termeq', 'ENTY:veh', 'ENTY:word',
    'HUM:desc', 'HUM:gr', 'HUM:ind', 'HUM:title',
    'LOC:city', 'LOC:country', 'LOC:mount', 'LOC:other', 'LOC:state',
    'NUM:code', 'NUM:count', 'NUM:date', 'NUM:dist', 'NUM:money', 'NUM:ord', 'NUM:other', 'NUM:perc', 'NUM:period',
    'NUM:speed', 'NUM:temp', 'NUM:volsize', 'NUM:weight']


def train_and_predict(coarse, train_fname, target_q_fname):
    train_coarse_ls, train_fine_ls, train_qs = parse_train_file(train_fname)
    train_ls = train_coarse_ls if coarse else train_fine_ls
    new_qs = parse_q_file(target_q_fname)
    all_ls = COARSE_LABELS if coarse else FINE_LABELS
    return do_train_and_predict(all_ls, train_qs, train_ls, new_qs)


def do_train_and_predict(all_ls, train_qs, train_ls, new_qs):
    res = []
    for q in new_qs:
        # filling with random
        predicted_l = random.choice(all_ls)
        # filling with constant
        # predicted_l = all_ls[0]
        res.append(predicted_l)
    return new_qs, res


def do_evaluate(true_ls, predicted_ls):
    tot_c = len(true_ls)
    if tot_c != len(predicted_ls):
        raise Exception("true and predicted label counts don't match")
    correct_c = 0
    for true_l, predicted_l in zip(true_ls, predicted_ls):
        if true_l == predicted_l:
            correct_c += 1
    res = round(100.0 * correct_c / tot_c)
    print(f'Accuracy is {res}%')
