# --- import --------------------------------------------------------------------------------------


import click
import WrightTools as wt

# --- define --------------------------------------------------------------------------------------


@click.group()
@click.version_option(wt.__version__)
def cli():
    pass


@cli.command(name="tree", help="Print a given data tree.")
@click.argument("path", nargs=1)
@click.option("--internal_path", default="/", help="specify a path internal to the file.  Defaults to root")
@click.option("--depth", "-d", "-L", type=int, default=9, help="Depth to print.")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Print a more detailed tree.")
def tree(path, internal_path, depth=9, verbose=False):
    # open the object
    obj = wt.open(path)[internal_path]

    # If the object is a data object, it doesn't take depth as a parameter
    if isinstance(obj, wt.Data):
        obj.print_tree(verbose=verbose)
    else:
        obj.print_tree(verbose=verbose, depth=depth)


@cli.command(name="load", help="Open a python cli with the object pre-loaded.")
@click.argument("path")
def load(path):
    import code
    d = wt.open(d)
    ...


@cli.command(name="crawl", help="Crawl a directory and survey the wt5 objects found.")
@click.argument("directory", nargs=-1, default=None)
@click.option("--recursive", "-r", is_flag=True, help="explore all levels of the directory")
@click.option("--pausing", "-p", is_flag=True, help="pause at each file and give option to interact.")
def crawl(directory=(), recursive=False, pausing=False):
    import glob, os

    if not directory:
        directory = os.getcwd()
    else:
        directory = directory[0]

    if pausing:
        import code
        shell = code.InteractiveConsole()

    for pathname in glob.iglob("**/*.wt5", root_dir=directory, recursive=recursive):
        print(pathname)
        d = wt.open(os.path.join(directory, pathname))
        if isinstance(d, wt.Data):
            d.print_tree(verbose=False)
        elif isinstance(d, wt.Collection):
            d.print_tree(depth=1, verbose=False)

        if pausing:
            msg = shell.raw_input("Continue [y]/n/interact? ")
            if msg == "n":
                break
            elif msg == "interact":
                lines = ["import WrightTools as wt"]
                if isinstance(d, wt.Data):
                    lines += [
                        "import matplotlib.pyplot as plt",
                        f"path = r'{os.path.join(directory, pathname)}'",
                        "d = wt.open(path)",
                    ]
                elif isinstance(d, wt.Collection):
                    lines.append(f"c = wt.open(r'{os.path.join(directory, pathname)}')")

                [shell.push(line) for line in lines]                
                shell.interact(banner="\n".join([">>> "+line for line in lines]))
            else:
                continue
        print("-"*100)
        d.close()


@cli.command(name="convert")
@click.argument("number", type=float, nargs=1)
@click.argument("unit", nargs=1)
@click.argument("destination_unit", default=None, nargs=-1)
def convert(number, unit, destination_unit=None):
    """Convert numbers to different units."""

    if int(number) == number:
        number = int(number)
    sig_figs = len(str(number))
    sig_figs -= 1 if "." in str(number) else 0

    def fmt(new):
        exponent = int(f"{new:e}".split("e")[1])
        if exponent > 6 or exponent < -3:
            return f"{new:{sig_figs}e}"
        else: # if a "normal" size number
            if sig_figs-exponent <= 0:
                return f"{int(round(new, sig_figs-exponent))}"
            else:
                return f"{round(new, sig_figs-exponent)}"


    if len(destination_unit):  # units provided
        destination_unit = destination_unit[0]
        if not wt.units.is_valid_conversion(unit, destination_unit):
            raise wt.exceptions.UnitsError(
                wt.units.get_valid_conversions(unit), destination_unit
            )
        new = wt.units.convert(number, unit, destination_unit)
        print(f"{number} {unit} = {fmt(new)} {destination_unit}")
    else:
        valid_units = wt.units.get_valid_conversions(unit)
        for d_unit in valid_units:
            new = wt.units.convert(number, unit, d_unit)
            print(f"{fmt(new)} {d_unit}")


if __name__ == "__main__":
    cli()

