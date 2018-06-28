import WrightTools as wt
import argparse

def wt_tree():
    parser = argparse.ArgumentParser(description='Print a given data tree.')
    parser.add_argument('path', type=str)
    parser.add_argument('--verbose', '-v', action='store_true', help='If verbose. Default is false')
    parser.add_argument('--depth', '-d', '-l', type=int, default=9, help='depth to print (int)')
    parser.add_argument('internal_path', nargs='?', default=/)
    args = parser.parse_args()
    obj = wt.open(args.path, edit_local=true)
    obj=obj[int_path]   # Set the path to local if changed
    if isinstance(obj, wt.Data):
       obj.print_tree(verbose=args.verbose)
    else
       obj.print_tree(verbose=args.verbose, depth=args.depth)
    
    #a = 1 if cond else 3
    #obj.print_tree(verbose=args.verbose) if isinstance(obj, wt.Data) else obj.print_tree(verbose=args.verbose, depth=args.depth)

# don't forget to black, and copy formatting from other files