import random

from file_ops import parse_train_file, parse_q_file
from preprocess import combine_prepros

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


def train_and_predict(coarse, train_fname, prepros, method):
    train_coarse_ls, train_fine_ls, train_qs = parse_train_file(train_fname)
    train_ls = train_coarse_ls if coarse else train_fine_ls
    new_qs = parse_q_file(target_q_fname)
    all_ls = COARSE_LABELS if coarse else FINE_LABELS
    train_qs = combine_prepros(train_qs, prepros)
    new_qs = combine_prepros(new_qs, prepros)
    return new_qs, method(all_ls, train_qs, train_ls, new_qs)

