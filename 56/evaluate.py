import sys


def parse_l_file(fname):
    coarse_ls, fine_ls = [], []
    with open(fname, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            fine_l = line.split(' ')[0]
            coarse_l = fine_l.split(':')[0]
            coarse_ls.append(coarse_l)
            fine_ls.append(fine_l)
    return coarse_ls, fine_ls


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


args = sys.argv
true_l_fname = args[1]
predicted_l_fname = args[2]

true_coarse_ls, true_fine_ls = parse_l_file(true_l_fname)

coarse = None
predicted_ls = []
with open(predicted_l_fname, 'r') as f:
    for line in f.readlines():
        line = line.replace('\n', '')
        curr_coarse = ':' not in line
        if coarse is None:
            coarse = curr_coarse
        elif coarse != curr_coarse:
            raise Exception("predicted labels contain both coarse and fine")
        predicted_ls.append(line)

do_evaluate(true_coarse_ls if coarse else true_fine_ls, predicted_ls)
