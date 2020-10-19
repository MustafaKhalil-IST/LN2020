
from file_ops import parse_train_file

_, fine_ls, qs = parse_train_file('DEV.txt')

with open('DEV-questions.txt', 'w') as f:
    for question in qs:
        f.write(question + '\n')

with open('DEV-labels.txt', 'w') as f:
    for fine_l in fine_ls:
        f.write(fine_l + '\n')
