# LN2020
## Group 56

To generate labels for coarse dev dataset run:
````batch
python qc.py -coarse TRAIN.txt DEV-questions.txt > develop56-coarse.txt
````


To generate labels for fine dev dataset run:
````batch
python qc.py -fine   TRAIN.txt DEV-questions.txt > develop56-fine.txt
````

To evaluate predicted coarse labels run:
````batch
python ./evaluate.py DEV-labels.txt develop56-coarse.txt
````

To evaluate predicted fine labels run:
````batch
python ./evaluate.py DEV-labels.txt develop56-fine.txt
````

To generate labels for coarse test dataset run:
````batch
python qc.py -coarse TRAIN.txt TEST.txt > test56-coarse.txt
````

To generate labels for fine test dataset run:
````batch
python qc.py -fine TRAIN.txt TEST.txt > test56-fine.txt
````
