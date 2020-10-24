#!/bin/bash

python qc.py -coarse TRAIN.txt DEV-questions.txt > develop56-coarse.txt
python qc.py -fine   TRAIN.txt DEV-questions.txt > develop56-fine.txt
python ./evaluate.py DEV-labels.txt develop56-coarse.txt
python ./evaluate.py DEV-labels.txt develop56-fine.txt

python qc.py -coarse TRAIN.txt TEST.txt > test56-coarse.txt
python qc.py -fine   TRAIN.txt TEST.txt > test56-fine.txt

# server
docker run -it --rm --name mp1 -v "$PWD":/usr/src/myapp -w /usr/src/myapp python:3 bash
pip install nltk, numpy
pip install sklearn
export PYTHONPATH="${PYTHONPATH}:$PWD"
