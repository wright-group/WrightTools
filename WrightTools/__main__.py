import WrightTools as wt
import argparse

def wt_tree():
    #print('hello world')
    parser = argparse.ArgumentParser(description='Print a given data tree.')
    parser.add_argument('path', type=str)
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose or not')
    parser.add_argument('--depth', '-d', type=int, default=9, help='depth to print (int)')
    args = parser.parse_args()
    data = wt.open(args.path)
    data.print_tree(verbose=args.verbose, depth=args.depth)

# don't forget to black, and copy formatting from other files