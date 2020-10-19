import random
from nltk import edit_distance


def predict_random(all_ls, train_qs, train_ls, new_qs):
    res = []
    for q in new_qs:
        # filling with random
        predicted_l = random.choice(all_ls)
        # filling with constant
        # predicted_l = all_ls[0]
        res.append(predicted_l)
    return res


def predict_levenshtein(all_ls, train_qs, train_ls, new_qs):
    idx = 0
    count = len(new_qs)
    train_qs = [q.split(' ') for q in train_qs]
    new_qs = [q.split(' ') for q in new_qs]
    res = []
    for q in new_qs:
        print(f'entry {idx} of {count}')
        min_dist = 999999
        candidate = None
        for tl, tq in zip(train_ls, train_qs):
            dist = edit_distance(q, tq)
            if dist < min_dist:
                min_dist = dist
                candidate = tl
        res.append(candidate)
        idx += 1
    return res

def knn():
    # TODO: k nearest neighbors, we will use sklearn to do that
    pass
