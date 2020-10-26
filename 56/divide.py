
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


_, fine_ls, qs = parse_train_file('DEV.txt')

with open('DEV-questions.txt', 'w') as f:
    for question in qs:
        f.write(question + '\n')

with open('DEV-labels.txt', 'w') as f:
    for fine_l in fine_ls:
        f.write(fine_l + '\n')
