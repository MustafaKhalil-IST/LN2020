
def count_labels(fname):
    coarse_l_set, fine_l_set = set(), set()
    with open(fname, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            fine_l = line.split(' ')[0]
            coarse_l = fine_l.split(':')[0]
            coarse_l_set.add(coarse_l)
            fine_l_set.add(fine_l)
    coarse_l_set = list(coarse_l_set)
    coarse_l_set.sort()
    print(f'coarse labels ({len(coarse_l_set)}): {coarse_l_set}')
    fine_l_set = list(fine_l_set)
    fine_l_set.sort()
    print(f'fine labels ({len(fine_l_set)}): {fine_l_set}')


def parse_train_file(fname):
    coarse_ls, fine_ls, qs = [], [], []
    with open(fname, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            parts = line.split(' ')
            fine_l, question = parts[0], ' '.join(parts[1:])
            coarse_l = fine_l.split(':')[0]
            coarse_ls.append(coarse_l)
            fine_ls.append(fine_l)
            qs.append(question)
    return coarse_ls, fine_ls, qs


def parse_q_file(fname):
    qs = []
    with open(fname, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            qs.append(line)
    return qs


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
