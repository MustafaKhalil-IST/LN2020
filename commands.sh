#!/bin/bash

python qc.py -coarse TRAIN.txt DEV-questions.txt > develop56-coarse.txt
python qc.py -fine TRAIN.txt DEV-questions.txt > develop56-fine.txt
python ./evaluate.py DEV-labels.txt develop56-coarse.txt
python ./evaluate.py DEV-labels.txt develop56-fine.txt
