#!/bin/bash

python qc.py -coarse TRAIN.txt DEV-questions.txt > predicted-labels-coarse.txt
python qc.py -fine TRAIN.txt DEV-questions.txt > predicted-labels-fine.txt
python ./evaluate.py DEV-labels.txt predicted-labels-coarse.txt
python ./evaluate.py DEV-labels.txt predicted-labels-fine.txt
