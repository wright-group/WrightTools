# --- import --------------------------------------------------------------------------------------


import WrightTools as wt
import argparse
from math import log10, floor


# --- define --------------------------------------------------------------------------------------
# entry points from terminal

# wt-tree

# Print a wt5 file tree
def wt_tree():
    parser = argparse.ArgumentParser(description="Print a given data tree.")

    # Add arguments
    parser.add_argument("path", type=str)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose? False by default")
    parser.add_argument("--depth", "-d", "-L", type=int, default=9, help="depth to print (int)")
    parser.add_argument("internal_path", nargs="?", default="/")
    args = parser.parse_args()

    # Create the data/collection object
    obj = wt.open(args.path, edit_local=True)[args.internal_path]

    # Print the tree
    # If the wt5 is a data object, it doesn't take depth as a parameter
    if isinstance(obj, wt.Data):
        obj.print_tree(verbose=args.verbose)
    else:
        obj.print_tree(verbose=args.verbose, depth=args.depth)


# wt-convert

def round_sig(value, sig):
    result = round(value, sig-int(floor(log10(abs(value))))-1)
    if result - int(result) == 0:
        result = int(result)
    return result

def numDigits(n): 
    return len(list(filter(lambda m:m.isdigit(), str(n))))

def convert_helper(value, orig, dest):
    return round_sig(wt.units.converter(float(value), orig, dest), numDigits(value))

def wt_convert():
    parser = argparse.ArgumentParser(description="Converts data units.")
    parser.add_argument('args', nargs='*')
    argsList = parser.parse_args().args

    # Perhaps more efficient way to do this is to loop backwards so I start with the units
    # Also use the other printer I used for the first assignment
    # Also try ternary operator
    # also functions maybe if once I make everything better there are repeats
    units = ["nm", "wn", "eV", "meV", "Hz", "THz", "GHz"]

    unitArgs = []
    for arg in argsList:
        if arg in units:
            unitArgs.append(arg)

    valueArgs = [x for x in argsList if x not in unitArgs]

    orig = unitArgs[0]

    # No destination units (so report all)
    if len(unitArgs) == 1:
        for value in valueArgs:
            if len(valueArgs) != 1:
                print(value, orig, "is")
            for dest in units:
                if dest != orig: # Don't print original units too
                    if len(valueArgs) != 1:
                        print("  ", end='')
                    print(convert_helper(value, orig, dest), dest)

    # Destination units
    else:
        dest = unitArgs[1]
        for value in valueArgs:
            if len(valueArgs) == 1:
                print(convert_helper(value, orig, dest))
            else:
                print(value, unitArgs[0], "is", convert_helper(value, orig, dest), dest)