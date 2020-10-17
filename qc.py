import sys

def process_coarse(fname):
    print('process_coarse')
    with open(fname, 'r') as f:
        print(f.readlines())

def process_fine(fname):
    print('process_fine')
    with open(fname, 'r') as f:
        print(f.readlines())

args = sys.argv

if args[1] == '-coarse':
    process_coarse(args[2])

if args[1] == '-fine':
    process_fine(args[2])