import sys
from model import Model

args = sys.argv
coarse = args[1] == '-coarse'
train_file_name = args[2]
dev_questions_file_name = args[3]

model = Model("closest", coarse)

model.train(train_file_name)

predicted_labels = model.predict(dev_questions_file_name, strategy='svm' if coarse else 'rff',
                                 prepros=['stem'] if coarse else ['no'])

for label in predicted_labels:
    print(model.classes[label])
