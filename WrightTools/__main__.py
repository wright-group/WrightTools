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


# Rounds a value to a given number of sig figs
def round_sig(value, sig):
    result = round(value, sig - int(floor(log10(abs(value)))) - 1)
    if result - int(result) == 0:
        result = int(result)
    return result


# Determines the number of sig figs of a number
def sigFigs(n):
    integral, _, fractional = n.partition(".")
    if fractional:
        return len((integral + fractional).lstrip("0"))
    else:
        return len(integral.strip("0"))


# Performs the conversion accounting for sig figs and scientific notation
def convert_helper(value, orig, dest):
    result = round_sig(wt.units.converter(float(value), orig, dest), sigFigs(value))
    if result > 1000000:
        return "{:.2e}".format(result)
    else:
        return result


def wt_convert():
    # Setup
    parser = argparse.ArgumentParser(description="Converts data units.")
    parser.add_argument("args", nargs="*",
        help='Pass data as "wt-convert [value] [original unit] [destination unit]". You may also' +
             ' pass multiple values to receive conversions for each, and/or add no destination' +
             ' unit to receive conversions to every unit.')
    argsList = parser.parse_args().args
    units = ["nm", "wn", "eV", "meV", "Hz", "THz", "GHz"]
    unitArgs = []
    for arg in argsList:
        if arg in units:
            unitArgs.append(arg)
    valueArgs = [x for x in argsList if x not in unitArgs]
    orig = unitArgs[0]

    # Loop through and print
    dest = True if len(unitArgs) != 1 else False
    for value in valueArgs:
        if len(valueArgs) != 1:
            print(value, orig, "is", end=" ")
            if not dest:
                print()

        # Destination unit case
        if dest:
            print(convert_helper(value, orig, unitArgs[1]), end=" ")
            if len(valueArgs) != 1:
                print(unitArgs[1])

        # No destination unit case (print all conversions)
        else:
            for unit in units:
                if unit != orig:
                    if len(valueArgs) != 1:
                        print("  ", end="")
                    print(convert_helper(value, orig, unit), unit)
