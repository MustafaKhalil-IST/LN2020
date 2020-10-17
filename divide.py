import sys

questions, labels = [], []
with open(sys.argv[1], 'r') as f:
    for line in f.readlines():
        label, question = line.split(' ')[0], ' '.join(line.split(' ')[1:]).replace('\n', '')
        questions.append(question)
        labels.append(label)
f.close()

with open('DEV-questions.txt', 'w') as f:
    for question in questions:
        f.write(question + '\n')
f.close()

with open('DEV-labels.txt', 'w') as f:
    for label in labels:
        f.write(label + '\n')
f.close()
