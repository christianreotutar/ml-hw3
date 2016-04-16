import sys
from collapsed import CollapsedSampler

'''
    Main program
'''
def main():
    train_lines, test_lines, output_file_path, \
        k, l, a, b, num_iters, num_burn_in = read_input()
    cs = CollapsedSampler(train_lines, test_lines, output_file_path, \
        k, l, a, b, num_iters, num_burn_in)
    cs.algorithm()


'''
    Reads in the input
    1 input train file
    2 input test file
    3 name of output file
    4 number of topics (K)
    5 value of lambda
    6 value of alpha
    7 value of beta
    8 number of total iterations
    9 number of samples to use as burn-in
'''
def read_input():

    if (len(sys.argv) < 10):
        raise Exception("Not enough params. Correct usage: ./collapsed-sampler input-train.txt input-test.txt output.txt 10 0.5 0.1 0.01 1100 1000")

    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]
    output_file_path = sys.argv[3]
    k = int(sys.argv[4])
    l = float(sys.argv[5])          # CORRECT
    a = float(sys.argv[6])
    b = float(sys.argv[7])
    num_iters = int(sys.argv[8])
    num_burn_in = int(sys.argv[9])

    f = open(train_file_path)
    train_lines = []
    for line in f.readlines(): 
        train_lines.append(line.strip().split())

    f = open(test_file_path)
    test_lines = []
    for line in f.readlines():
        test_lines.append(line.strip().split())

    return train_lines, test_lines, output_file_path, k, l, a, b, num_iters, num_burn_in


if __name__ == "__main__":    
    main()
