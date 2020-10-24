from file_ops import parse_l_file
from model import Model


def do_evaluate(true_ls, predicted_ls):
    tot_c = len(true_ls)
    if tot_c != len(predicted_ls):
        raise Exception("true and predicted label counts don't match")
    correct_c = 0
    for true_l, predicted_l in zip(true_ls, predicted_ls):
        if true_l == predicted_l:
            correct_c += 1
    res = round(100.0 * correct_c / tot_c)
    return res


prepros = {
    'no': [],
    'lower': ['lower'],
    'stem': ['stem'],
    'token': ['token'],
    'lower_stem': ['lower', 'stem'],
    'lower_token': ['lower', 'token'],
    'stem_token': ['stem', 'token'],
    'lower_stem_token': ['lower', 'stem', 'token'],
}

strategies = ['knn', 'rff', 'dt']

train_file_name = 'TRAIN.txt'
dev_questions_file_name = 'DEV-questions.txt'

model = Model("closest", coarse=True)
true_coarse_labels, true_fine_labels = parse_l_file('DEV-labels.txt')

for strategy in strategies:
    for prepro in prepros:
        model.train(train_file_name)
        predicted_labels_numbers = model.predict(dev_questions_file_name, strategy=strategy, prepros=prepros[prepro])
        predicted_labels = [model.classes[e] for e in predicted_labels_numbers]
        acc = do_evaluate(true_coarse_labels, predicted_labels)
        print("Strategy={} - Prepros={} - Acc={}".format(strategy, prepro, acc))
